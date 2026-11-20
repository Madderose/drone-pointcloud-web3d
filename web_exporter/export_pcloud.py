#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export PLY point clouds, DJI SRT trajectories, and POIs into progressive `.pcloud` (`CHPC` v1).

Binary `.pcloud` format (gzip level 9, no `.gz` suffix to prevent Apache double-decompression):
  - 48-byte Little-Endian header (`<4s3I8f`):
      magic   : 4s  (b"CHPC")
      version : u32 (1)
      count   : u32 (number of points)
      flags   : u32 (0)
      bbox    : 8 x float32 (minX, minY, minZ, maxX, maxY, maxZ, zP02, zP98)
  - Randomly shuffled payload (for progressive `setDrawRange` LOD):
      xyz_u16 : uint16[count * 3] quantized to [0, 65535]
      rgb_u8  : uint8[count * 3]
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.common import EnuOrigin, parse_dji_srt


def _split_spec_preserving_windows_drive(spec: str, sep: str = ":") -> list[str]:
    """Split a colon-separated spec string while keeping `C:\\...` drive prefixes intact."""
    if sep != ":":
        return spec.split(sep)
    tokens: list[str] = []
    cur: list[str] = []
    i = 0
    while i < len(spec):
        ch = spec[i]
        if (
            ch == ":"
            and len(cur) == 1
            and cur[0].isalpha()
            and i + 1 < len(spec)
            and spec[i + 1] in ("\\", "/")
        ):
            cur.append(ch)
        elif ch == ":":
            tokens.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    tokens.append("".join(cur))
    return tokens


def parse_layer_spec(spec: str) -> dict:
    """Parse `--layer id:label:ply_path[:role[:target_pts[:voxel_m]]]` (or `|` separated)."""
    sep = "|" if "|" in spec else ":"
    parts = _split_spec_preserving_windows_drive(spec, sep=sep)
    if len(parts) < 3:
        raise ValueError(
            f"Invalid --layer spec '{spec}'. Expected id:label:ply_path[:role[:target_pts[:voxel_m]]]"
        )
    layer_id = parts[0].strip()
    label = parts[1].strip() or layer_id
    ply_path = Path(parts[2].strip())
    role = parts[3].strip() if len(parts) > 3 and parts[3].strip() else "primary"
    target_pts = int(parts[4]) if len(parts) > 4 and parts[4].strip() else 1_400_000
    voxel_m = float(parts[5]) if len(parts) > 5 and parts[5].strip() else 0.04
    return {
        "id": layer_id,
        "label": label,
        "file": f"{layer_id}.pcloud",
        "ply": ply_path,
        "role": role,
        "target_points": target_pts,
        "voxel_m": voxel_m,
    }


def parse_trajectory_spec(spec: str) -> dict:
    """Parse `--trajectory-srt id:label:srt_path[:color[:step_frames]]`."""
    sep = "|" if "|" in spec else ":"
    parts = _split_spec_preserving_windows_drive(spec, sep=sep)
    if len(parts) < 3:
        raise ValueError(
            f"Invalid --trajectory-srt spec '{spec}'. Expected id:label:srt_path[:color[:step_frames]]"
        )
    return {
        "id": parts[0].strip(),
        "label": parts[1].strip() or parts[0].strip(),
        "srt": Path(parts[2].strip()),
        "color": parts[3].strip() if len(parts) > 3 and parts[3].strip() else "#14B8A6",
        "step_frames": int(parts[4]) if len(parts) > 4 and parts[4].strip() else 15,
    }


def downsample_and_shuffle(
    pcd: o3d.geometry.PointCloud,
    target_pts: int,
    initial_voxel_m: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Voxel-downsample gently toward `target_pts` and randomly permute for LOD streaming."""
    voxel_m = float(initial_voxel_m)
    down = pcd.voxel_down_sample(voxel_m) if voxel_m > 0 else pcd

    while len(down.points) < int(target_pts * 0.85) and voxel_m > 0.015 and len(pcd.points) > target_pts:
        voxel_m = round(voxel_m * 0.72, 4)
        down = pcd.voxel_down_sample(voxel_m)

    pts = np.asarray(down.points, dtype=np.float32)
    if len(pts) == 0:
        raise ValueError("Point cloud has 0 points after downsampling.")

    if down.has_colors():
        cols = np.clip(np.round(np.asarray(down.colors) * 255.0), 0, 255).astype(np.uint8)
    else:
        cols = np.full((len(pts), 3), 200, dtype=np.uint8)

    rng = np.random.default_rng(seed)
    if len(pts) > target_pts:
        idx = rng.choice(len(pts), size=target_pts, replace=False)
        rng.shuffle(idx)
        pts = pts[idx]
        cols = cols[idx]
    else:
        perm = rng.permutation(len(pts))
        pts = pts[perm]
        cols = cols[perm]

    return pts, cols, voxel_m


def write_pcloud(out_path: Path, pts: np.ndarray, cols: np.ndarray) -> dict:
    """Quantize points to uint16, pack CHPC v1 header + payload, and write gzipped `.pcloud`."""
    count = int(len(pts))
    pts_f32 = np.asarray(pts, dtype=np.float32)
    mins = pts_f32.min(axis=0).astype(np.float32)
    maxs = pts_f32.max(axis=0).astype(np.float32)
    spans = np.maximum(maxs - mins, np.float32(1e-5))
    zp02, zp98 = np.percentile(pts_f32[:, 2], [2.0, 98.0]).astype(np.float32)

    q_xyz = np.clip(np.round((pts_f32 - mins) / spans * 65535.0), 0, 65535).astype("<u2")
    rgb_u8 = np.ascontiguousarray(cols, dtype=np.uint8)

    header = struct.pack(
        "<4s3I8f",
        b"CHPC",
        1,
        count,
        0,
        float(mins[0]),
        float(mins[1]),
        float(mins[2]),
        float(maxs[0]),
        float(maxs[1]),
        float(maxs[2]),
        float(zp02),
        float(zp98),
    )
    raw = header + np.ascontiguousarray(q_xyz).tobytes() + rgb_u8.tobytes()
    compressed = gzip.compress(raw, compresslevel=9)

    tmp_path = out_path.with_suffix(".pcloud.tmp")
    tmp_path.write_bytes(compressed)
    tmp_path.replace(out_path)

    return {
        "points": count,
        "bytes": len(compressed),
        "raw_bytes": len(raw),
        "bbox_enu": {
            "min": [round(float(x), 3) for x in mins],
            "max": [round(float(x), 3) for x in maxs],
            "z_p02": round(float(zp02), 3),
            "z_p98": round(float(zp98), 3),
        },
    }


def auto_presets_from_bbox(layers_meta: list[dict]) -> list[dict]:
    """Generate clean default camera presets from the combined ENU bounding box."""
    if not layers_meta:
        return []
    bmin = np.min([l["bbox_enu"]["min"] for l in layers_meta], axis=0)
    bmax = np.max([l["bbox_enu"]["max"] for l in layers_meta], axis=0)
    center = 0.5 * (bmin + bmax)
    span = float(max(12.0, np.linalg.norm(bmax[:2] - bmin[:2])))
    return [
        {
            "id": "oblique_34",
            "label": "3/4 Oblique",
            "cam_enu": [
                round(float(center[0] - 0.55 * span), 2),
                round(float(center[1] - 0.65 * span), 2),
                round(float(center[2] + 0.45 * span), 2),
            ],
            "target_enu": [round(float(v), 2) for v in center],
        },
        {
            "id": "zenith",
            "label": "Top-Down Zenith",
            "cam_enu": [
                round(float(center[0]), 2),
                round(float(center[1] - 0.5), 2),
                round(float(center[2] + 1.05 * span), 2),
            ],
            "target_enu": [round(float(v), 2) for v in center],
        },
        {
            "id": "overview",
            "label": "Wide Overview",
            "cam_enu": [
                round(float(center[0] + 0.85 * span), 2),
                round(float(center[1] - 0.85 * span), 2),
                round(float(center[2] + 0.75 * span), 2),
            ],
            "target_enu": [round(float(v), 2) for v in center],
        },
    ]


def export_web_package(
    out_dir: Path,
    layer_specs: list[dict],
    site_title: str = "Drone 3D Point Cloud",
    subtitle: str = "Georeferenced ENU Photogrammetry",
    trajectory_specs: list[dict] | None = None,
    pois: list[dict] | None = None,
    presets: list[dict] | None = None,
    origin: EnuOrigin | None = None,
) -> Path:
    """Build `.pcloud` layers, `trajectories.json`, and `manifest.json` inside `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    version_ts = int(time.time())

    pcd_cache: dict[Path, o3d.geometry.PointCloud] = {}
    layers_out: list[dict] = []

    for spec in layer_specs:
        ply_path = Path(spec["ply"])
        if not ply_path.exists():
            raise FileNotFoundError(f"Point cloud PLY not found: {ply_path}")
        if ply_path not in pcd_cache:
            pcd_cache[ply_path] = o3d.io.read_point_cloud(str(ply_path))
        pcd = pcd_cache[ply_path]

        pts, cols, vox_used = downsample_and_shuffle(
            pcd,
            target_pts=int(spec.get("target_points", 1_400_000)),
            initial_voxel_m=float(spec.get("voxel_m", 0.04)),
        )
        file_name = spec.get("file") or f"{spec['id']}.pcloud"
        meta = write_pcloud(out_dir / file_name, pts, cols)

        layer_entry = {
            "id": spec["id"],
            "label": spec["label"],
            "file": file_name,
            "role": spec.get("role", "primary"),
            "voxel_cm": round(vox_used * 100.0, 1),
            "version_ts": version_ts,
            **meta,
        }
        if spec.get("replaces"):
            layer_entry["replaces"] = spec["replaces"]
        layers_out.append(layer_entry)
        print(f"  {file_name}: {meta['points']:,} pts, {meta['bytes']:,} bytes (voxel {vox_used*100:.1f} cm)")

    traj_specs = trajectory_specs or []
    flights_out: list[dict] = []
    derived_origin = origin

    for ts in traj_specs:
        srt_path = Path(ts["srt"])
        if not srt_path.exists():
            print(f"WARN: trajectory SRT missing, skipping: {srt_path}")
            continue
        rows = parse_dji_srt(srt_path)
        if not rows:
            continue
        if derived_origin is None:
            r0 = rows[0]
            derived_origin = EnuOrigin(r0["lat"], r0["lon"], r0["abs_alt"] - r0["rel_alt"])
        step = max(1, int(ts.get("step_frames", 15)))
        pts_list = []
        for idx, r in enumerate(rows):
            if idx % step != 0:
                continue
            xe, yn, zu = derived_origin.to_enu(r["lat"], r["lon"], r["abs_alt"])
            pts_list.append(
                {
                    "f": r["frame_cnt"],
                    "enu": [round(xe, 3), round(yn, 3), round(zu, 3)],
                    "rel_alt": round(r["rel_alt"], 2),
                }
            )
        flights_out.append(
            {
                "id": ts["id"],
                "label": ts["label"],
                "color": ts.get("color", "#14B8A6"),
                "step_frames": step,
                "points": pts_list,
            }
        )

    origin_dict = derived_origin.as_dict() if derived_origin else {"lat0": 0.0, "lon0": 0.0, "abs_alt_ref": 0.0}
    pois_out = pois or []

    traj_file_name = None
    if flights_out or pois_out:
        traj_file_name = "trajectories.json"
        traj_payload = {
            "enu_origin": origin_dict,
            "flights": flights_out,
            "pois": pois_out,
        }
        (out_dir / traj_file_name).write_text(
            json.dumps(traj_payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    final_presets = presets if presets else auto_presets_from_bbox(layers_out)
    active_pts = sum(l["points"] for l in layers_out if l["role"] != "preview")
    if active_pts == 0:
        active_pts = sum(l["points"] for l in layers_out)
    total_mb = round(sum(l["bytes"] for l in layers_out) / (1024 * 1024), 2)

    manifest = {
        "version": 1,
        "version_ts": version_ts,
        "site": site_title,
        "subtitle": subtitle,
        "coordinate_frame": "ENU (X=East, Y=North, Z=Up) -> Three.js (X, Z, -Y)",
        "enu_origin": origin_dict,
        "layers": layers_out,
        "pending_layers": [],
        "trajectories_file": traj_file_name,
        "presets": final_presets,
        "summary": {
            "active_points": active_pts,
            "total_download_mb": total_mb,
            "flights_count": len(flights_out),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest written -> {out_dir / 'manifest.json'} ({active_pts:,} active pts, {total_mb} MB)")
    return out_dir


def _resolve_cfg_path(base_dir: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (base_dir / p).resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for `.pcloud` exporter."""
    parser = argparse.ArgumentParser(
        description="Convert PLY clouds + DJI SRT trajectories + POIs into progressive gzipped `.pcloud` & manifest.json."
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config file (reads web_export section).")
    parser.add_argument("--out", type=Path, default=None, help="Output data directory (e.g. viewer/data).")
    parser.add_argument("--site", default="Drone 3D Point Cloud", help="Site title displayed in viewer header.")
    parser.add_argument("--subtitle", default="Georeferenced ENU Photogrammetry", help="Viewer subtitle.")
    parser.add_argument(
        "--layer",
        action="append",
        default=[],
        help="Layer spec: id:label:ply_path[:role[:target_pts[:voxel_m]]] (repeatable).",
    )
    parser.add_argument(
        "--trajectory-srt",
        action="append",
        default=[],
        help="Trajectory spec: id:label:srt_path[:color[:step_frames]] (repeatable).",
    )
    parser.add_argument("--pois-json", type=Path, default=None, help="Optional JSON file containing POI list.")
    parser.add_argument("--presets-json", type=Path, default=None, help="Optional JSON file containing camera presets.")
    parser.add_argument("--origin-lat", type=float, default=None, help="Optional ENU origin latitude.")
    parser.add_argument("--origin-lon", type=float, default=None, help="Optional ENU origin longitude.")
    parser.add_argument("--origin-alt", type=float, default=None, help="Optional ENU origin altitude (m).")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args(argv)

    layer_specs: list[dict] = []
    traj_specs: list[dict] = []
    pois: list[dict] = []
    presets: list[dict] = []
    out_dir = args.out
    site = args.site
    subtitle = args.subtitle
    origin = None

    if args.origin_lat is not None and args.origin_lon is not None:
        origin = EnuOrigin(args.origin_lat, args.origin_lon, args.origin_alt or 0.0)

    if args.config is not None:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        base = args.config.resolve().parent
        proj = cfg.get("project", {})
        site = proj.get("name", site)
        subtitle = proj.get("subtitle", subtitle)
        enu_cfg = cfg.get("enu_origin", {})
        if origin is None and str(enu_cfg.get("mode", "auto")).lower() == "manual":
            origin = EnuOrigin(float(enu_cfg["lat0"]), float(enu_cfg["lon0"]), float(enu_cfg.get("abs_alt_ref", 0.0)))
        wcfg = cfg.get("web_export", {})
        if out_dir is None and wcfg.get("out_dir"):
            out_dir = _resolve_cfg_path(base, wcfg["out_dir"])
        for litem in wcfg.get("layers", []):
            layer_specs.append(
                {
                    "id": litem["id"],
                    "label": litem.get("label", litem["id"]),
                    "file": litem.get("file", f"{litem['id']}.pcloud"),
                    "ply": _resolve_cfg_path(base, litem["ply"]),
                    "role": litem.get("role", "primary"),
                    "replaces": litem.get("replaces"),
                    "target_points": int(litem.get("target_points", 1_400_000)),
                    "voxel_m": float(litem.get("voxel_m", 0.04)),
                }
            )
        for titem in wcfg.get("trajectories", []):
            traj_specs.append(
                {
                    "id": titem["id"],
                    "label": titem.get("label", titem["id"]),
                    "srt": _resolve_cfg_path(base, titem["srt"]),
                    "color": titem.get("color", "#14B8A6"),
                    "step_frames": int(titem.get("step_frames", 15)),
                }
            )
        pois.extend(wcfg.get("pois", []))
        presets.extend(wcfg.get("presets", []))

    for raw_l in args.layer:
        layer_specs.append(parse_layer_spec(raw_l))
    for raw_t in args.trajectory_srt:
        traj_specs.append(parse_trajectory_spec(raw_t))
    if args.pois_json is not None:
        raw_p = json.loads(args.pois_json.read_text(encoding="utf-8"))
        pois = raw_p["pois"] if isinstance(raw_p, dict) and "pois" in raw_p else raw_p
    if args.presets_json is not None:
        raw_pr = json.loads(args.presets_json.read_text(encoding="utf-8"))
        presets = raw_pr["presets"] if isinstance(raw_pr, dict) and "presets" in raw_pr else raw_pr

    if out_dir is None:
        raise SystemExit("Error: specify --out <dir> or web_export.out_dir in --config.")
    if not layer_specs:
        raise SystemExit("Error: specify at least one --layer or web_export.layers entry.")

    export_web_package(
        out_dir=out_dir,
        layer_specs=layer_specs,
        site_title=site,
        subtitle=subtitle,
        trajectory_specs=traj_specs,
        pois=pois,
        presets=presets,
        origin=origin,
    )


if __name__ == "__main__":
    main()
