#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1: Extract frames, build conservative sky masks, ENU priors, and spatio-temporal pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.common import (
    EnuOrigin,
    compute_sky_mask_and_stats,
    parse_dji_srt,
    read_exif_gps,
)


def parse_video_spec(spec: str, idx: int) -> dict:
    """Parse `--videos` item formatted as `path[:start[:end[:stride[:prefix[:tag]]]]]`.

    On Windows, drive prefixes like `C:\\...` are preserved cleanly.
    """
    drive = ""
    rest = spec
    if len(spec) >= 2 and spec[1] == ":" and spec[0].isalpha():
        drive = spec[:2]
        rest = spec[2:]

    parts = rest.split(":")
    video_path = Path(drive + parts[0])
    start_frame = int(parts[1]) if len(parts) > 1 and parts[1] != "" else 0
    end_frame = int(parts[2]) if len(parts) > 2 and parts[2] != "" else -1
    stride = int(parts[3]) if len(parts) > 3 and parts[3] != "" else 15
    default_tag = f"{video_path.stem}_s{idx}"
    prefix = parts[4] if len(parts) > 4 and parts[4] != "" else f"v{idx}_{video_path.stem[:10]}"
    tag = parts[5] if len(parts) > 5 and parts[5] != "" else default_tag
    return {
        "path": video_path,
        "srt": None,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "stride": max(1, stride),
        "prefix": prefix,
        "tag": tag,
    }


def resolve_srt_path(video_path: Path, explicit_srt: str | None = None) -> Path:
    """Locate `.SRT` telemetry sidecar alongside a video file."""
    if explicit_srt:
        p = Path(explicit_srt)
        if p.exists():
            return p
    for ext in (".SRT", ".srt"):
        cand = video_path.with_suffix(ext)
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No .SRT telemetry sidecar found for video: {video_path}")


def determine_origin(
    video_specs: list[dict],
    still_paths: list[Path],
    origin_lat: float | None,
    origin_lon: float | None,
    origin_alt: float | None,
) -> EnuOrigin:
    """Resolve ENU tangent origin from explicit CLI flags or the first telemetry record."""
    if origin_lat is not None and origin_lon is not None:
        return EnuOrigin(
            lat0=float(origin_lat),
            lon0=float(origin_lon),
            abs_alt_ref=float(origin_alt if origin_alt is not None else 0.0),
        )

    for v in video_specs:
        srt_path = resolve_srt_path(v["path"], v.get("srt"))
        rows = parse_dji_srt(srt_path)
        if rows:
            first = rows[0]
            ground_alt = first["abs_alt"] - first["rel_alt"]
            return EnuOrigin(
                lat0=first["lat"],
                lon0=first["lon"],
                abs_alt_ref=float(origin_alt if origin_alt is not None else ground_alt),
            )

    for sp in still_paths:
        gps = read_exif_gps(sp)
        return EnuOrigin(
            lat0=gps["lat"],
            lon0=gps["lon"],
            abs_alt_ref=float(origin_alt if origin_alt is not None else gps["abs_alt"]),
        )

    raise RuntimeError("Could not infer ENU origin: no SRT telemetry or EXIF GPS inputs provided.")


def save_frame_and_mask(
    img_bgr: np.ndarray,
    img_dir: Path,
    mask_dir: Path,
    rel_name: str,
    width: int,
    height: int,
    jpeg_quality: int = 96,
) -> dict:
    """Resize BGR frame with Lanczos4, compute sky mask, and write JPEG + PNG mask."""
    resized = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_LANCZOS4)
    keep_mask, stats = compute_sky_mask_and_stats(resized)

    img_out = img_dir / rel_name
    mask_out = mask_dir / f"{rel_name}.png"
    img_out.parent.mkdir(parents=True, exist_ok=True)
    mask_out.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(img_out), resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    cv2.imwrite(str(mask_out), keep_mask)
    return stats


def extract_video_segments_for_file(
    mp4_path: Path,
    srt_rows: list[dict],
    segments: list[dict],
    origin: EnuOrigin,
    img_dir: Path,
    mask_dir: Path,
    width: int,
    height: int,
    jpeg_quality: int,
    backlit_lap_thresh: float,
    meta_list: list[dict],
) -> None:
    """Single sequential pass over an MP4 file extracting one or more frame segments."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {mp4_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_map: dict[int, tuple[str, str, int]] = {}
    counts: dict[str, int] = {}
    for seg in segments:
        start = max(0, seg["start_frame"])
        end = total if seg["end_frame"] < 0 else min(total, seg["end_frame"])
        stride = max(1, seg["stride"])
        out_counter = 0
        for fi in range(start, end, stride):
            frame_map[fi] = (seg["prefix"], seg["tag"], out_counter)
            out_counter += 1
        counts[seg["tag"]] = out_counter

    if not frame_map:
        cap.release()
        return

    max_idx = max(frame_map.keys())
    idx = 0
    while idx <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in frame_map:
            prefix, tag, out_counter = frame_map[idx]
            rel_name = f"video/{prefix}_{out_counter:04d}.jpg"
            stats = save_frame_and_mask(frame, img_dir, mask_dir, rel_name, width, height, jpeg_quality)
            srt = srt_rows[min(idx, len(srt_rows) - 1)]
            xe, yn, zu = origin.to_enu(srt["lat"], srt["lon"], srt["abs_alt"])
            t_sec = idx / fps
            meta_list.append(
                {
                    "image_name": rel_name,
                    "source": tag,
                    "orig_file": mp4_path.name,
                    "orig_frame_idx": idx,
                    "t_sec": round(t_sec, 3),
                    "enu": [xe, yn, zu],
                    "is_backlit": stats["sharpness_lap"] < backlit_lap_thresh,
                    **stats,
                }
            )
        idx += 1

    cap.release()
    for seg in segments:
        print(f"  [{seg['tag']}] extracted {counts[seg['tag']]} frames -> video/{seg['prefix']}_XXXX.jpg")


def build_pairs(meta_list: list[dict], temporal_window: int = 12, spatial_k: int = 45) -> list[tuple[str, str]]:
    """Build deduplicated unordered match pairs from temporal neighbors + spatial ENU k-NN."""
    pairs: set[tuple[str, str]] = set()

    def _add(a: str, b: str) -> None:
        if a != b:
            pairs.add((a, b) if a < b else (b, a))

    video_files = sorted({m["orig_file"] for m in meta_list if m["image_name"].startswith("video/")})
    for vf in video_files:
        seq = sorted(
            [m for m in meta_list if m["orig_file"] == vf],
            key=lambda m: m["orig_frame_idx"],
        )
        for i in range(len(seq)):
            for j in range(i + 1, min(len(seq), i + 1 + temporal_window)):
                _add(seq[i]["image_name"], seq[j]["image_name"])

    if spatial_k > 0 and len(meta_list) > 1:
        names = [m["image_name"] for m in meta_list]
        coords = np.array([m["enu"] for m in meta_list], dtype=np.float64)
        diffs = coords[:, None, :] - coords[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        k = min(spatial_k + 1, len(names))
        for i in range(len(names)):
            nn_idx = np.argsort(dists[i])[:k]
            for j in nn_idx:
                if i != int(j):
                    _add(names[i], names[int(j)])

    return sorted(pairs)


def render_contact_sheet(meta_list: list[dict], img_dir: Path, mask_dir: Path, out_path: Path) -> None:
    """Render a 12-frame diagnostic contact sheet highlighting masked sky pixels."""
    if not meta_list:
        return
    picks = np.linspace(0, len(meta_list) - 1, min(12, len(meta_list)), dtype=int)
    fig, axes = plt.subplots(3, 4, figsize=(20, 10), dpi=130)
    for ax in axes.flat:
        ax.axis("off")
    for ax, p_idx in zip(axes.flat, picks):
        it = meta_list[int(p_idx)]
        im_rgb = cv2.cvtColor(cv2.imread(str(img_dir / it["image_name"])), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(str(mask_dir / f"{it['image_name']}.png"), cv2.IMREAD_GRAYSCALE)
        overlay = im_rgb.copy()
        sky_px = msk == 0
        overlay[sky_px] = (0.5 * overlay[sky_px] + 0.5 * np.array([255, 40, 160])).astype(np.uint8)
        ax.imshow(overlay)
        tag = "LOW_SHARP" if it["is_backlit"] else "OK"
        ax.set_title(
            f"{it['image_name'].split('/')[-1]} [{tag}]\n"
            f"sharp={it['sharpness_lap']:.0f} sky={it['sky_masked_pct']:.1f}%",
            fontsize=9,
        )
    plt.suptitle("Sampled Frames & Conservative Sky-Mask Audit (Magenta = Masked Sky)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def prepare_dataset(
    video_specs: list[dict],
    still_paths: list[Path],
    out_dir: Path,
    width: int = 2048,
    height: int = 1152,
    jpeg_quality: int = 96,
    origin_lat: float | None = None,
    origin_lon: float | None = None,
    origin_alt: float | None = None,
    temporal_window: int = 12,
    gps_neighbors: int = 45,
    backlit_lap_thresh: float = 1800.0,
    make_contact_sheet: bool = True,
) -> Path:
    """Execute Stage 1 dataset preparation and return the dataset directory."""
    img_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    origin = determine_origin(video_specs, still_paths, origin_lat, origin_lon, origin_alt)
    print(f"ENU origin: lat0={origin.lat0:.6f}, lon0={origin.lon0:.6f}, abs_alt_ref={origin.abs_alt_ref:.3f} m")

    meta_list: list[dict] = []

    # Group segments sharing the same video path for a single sequential read pass
    by_video: dict[Path, list[dict]] = {}
    for v in video_specs:
        vp = Path(v["path"]).resolve()
        by_video.setdefault(vp, []).append(v)

    for vp, segs in by_video.items():
        srt_path = resolve_srt_path(vp, segs[0].get("srt"))
        srt_rows = parse_dji_srt(srt_path)
        if not srt_rows:
            raise RuntimeError(f"Parsed 0 telemetry rows from {srt_path}")
        extract_video_segments_for_file(
            vp,
            srt_rows,
            segs,
            origin,
            img_dir,
            mask_dir,
            width,
            height,
            jpeg_quality,
            backlit_lap_thresh,
            meta_list,
        )

    for idx, sp in enumerate(still_paths):
        gps = read_exif_gps(sp)
        img = cv2.imread(str(sp))
        if img is None:
            raise RuntimeError(f"Failed to read still image: {sp}")
        rel_name = f"stills/still_{idx:03d}_{sp.stem}.jpg"
        stats = save_frame_and_mask(img, img_dir, mask_dir, rel_name, width, height, jpeg_quality)
        xe, yn, zu = origin.to_enu(gps["lat"], gps["lon"], gps["abs_alt"])
        meta_list.append(
            {
                "image_name": rel_name,
                "source": "stills",
                "orig_file": sp.name,
                "orig_frame_idx": 0,
                "t_sec": 0.0,
                "enu": [xe, yn, zu],
                "is_backlit": stats["sharpness_lap"] < backlit_lap_thresh,
                **stats,
            }
        )
    if still_paths:
        print(f"  [stills] extracted {len(still_paths)} still images -> stills/")

    meta_path = out_dir / "frames_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **origin.as_dict(),
                "width": width,
                "height": height,
                "total_images": len(meta_list),
                "frames": meta_list,
            },
            f,
            indent=2,
        )

    priors_path = out_dir / "gps_priors_enu.txt"
    with open(priors_path, "w", encoding="utf-8") as f:
        for item in meta_list:
            xe, yn, zu = item["enu"]
            f.write(f"{item['image_name']} {xe:.6f} {yn:.6f} {zu:.6f}\n")

    pairs = build_pairs(meta_list, temporal_window=temporal_window, spatial_k=gps_neighbors)
    pairs_path = out_dir / "pairs.txt"
    with open(pairs_path, "w", encoding="utf-8") as f:
        for a, b in pairs:
            f.write(f"{a} {b}\n")

    if make_contact_sheet:
        render_contact_sheet(meta_list, img_dir, mask_dir, out_dir / "contact_sheet_masks.jpg")

    print(f"Dataset ready: {len(meta_list)} images, {len(pairs)} pairs -> {out_dir}")
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for Stage 1."""
    parser = argparse.ArgumentParser(
        description="Prepare drone video frames, stills, sky masks, ENU GPS priors, and COLMAP pairs."
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        default=[],
        help="Video specs formatted as path[:start[:end[:stride[:prefix[:tag]]]]].",
    )
    parser.add_argument("--stills", nargs="*", default=[], help="Optional still JPEG paths with EXIF GPS.")
    parser.add_argument("--out", required=True, type=Path, help="Output dataset directory.")
    parser.add_argument("--width", type=int, default=2048, help="Resized output width (default: 2048).")
    parser.add_argument("--height", type=int, default=1152, help="Resized output height (default: 1152).")
    parser.add_argument("--jpeg-quality", type=int, default=96, help="Output JPEG quality (default: 96).")
    parser.add_argument("--origin-lat", type=float, default=None, help="Explicit WGS84 origin latitude.")
    parser.add_argument("--origin-lon", type=float, default=None, help="Explicit WGS84 origin longitude.")
    parser.add_argument("--origin-alt", type=float, default=None, help="Explicit WGS84 origin altitude (m).")
    parser.add_argument("--temporal-window", type=int, default=12, help="Sequential neighbor window (default: 12).")
    parser.add_argument("--gps-neighbors", type=int, default=45, help="Spatial ENU k-NN pairs per image (default: 45).")
    parser.add_argument("--no-contact-sheet", action="store_true", help="Skip rendering contact_sheet_masks.jpg.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args(argv)
    if not args.videos and not args.stills:
        raise SystemExit("Error: provide at least one --videos or --stills input.")
    video_specs = [parse_video_spec(s, i) for i, s in enumerate(args.videos)]
    still_paths = [Path(s) for s in args.stills]
    prepare_dataset(
        video_specs=video_specs,
        still_paths=still_paths,
        out_dir=args.out,
        width=args.width,
        height=args.height,
        jpeg_quality=args.jpeg_quality,
        origin_lat=args.origin_lat,
        origin_lon=args.origin_lon,
        origin_alt=args.origin_alt,
        temporal_window=args.temporal_window,
        gps_neighbors=args.gps_neighbors,
        make_contact_sheet=not args.no_contact_sheet,
    )


if __name__ == "__main__":
    main()
