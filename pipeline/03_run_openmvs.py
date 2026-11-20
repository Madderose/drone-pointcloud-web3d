#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3: OpenMVS Dense MVS + Delaunay Visibility-Carved Mesh Proximity Filtering."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def resolve_openmvs_exe(openmvs_bin: Path | None, base_name: str) -> str:
    """Resolve OpenMVS CLI executable on Windows (.exe) or Linux/macOS."""
    candidates = [f"{base_name}.exe", base_name] if os.name == "nt" else [base_name, f"{base_name}.exe"]
    if openmvs_bin is not None:
        for cand in candidates:
            p = openmvs_bin / cand
            if p.exists():
                return str(p)
    for cand in candidates:
        found = shutil.which(cand)
        if found:
            return found
    raise FileNotFoundError(
        f"OpenMVS executable '{base_name}' not found. Pass --openmvs-bin <dir> or add it to PATH."
    )


def run_openmvs(
    openmvs_bin: Path | None,
    base_name: str,
    args: list[str],
    cwd: Path,
    check: bool = True,
) -> tuple[int, float]:
    """Run an OpenMVS CLI tool with working-dir logs surfaced on error."""
    exe = resolve_openmvs_exe(openmvs_bin, base_name)
    env = os.environ.copy()
    if openmvs_bin is not None:
        sep = ";" if os.name == "nt" else ":"
        env["PATH"] = str(openmvs_bin) + sep + env.get("PATH", "")

    print(f"\n>>> [OpenMVS {base_name}] {' '.join(args[:6])} ...", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [exe] + args,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    dt = time.time() - t0
    if proc.returncode != 0:
        logs = sorted(cwd.glob(f"{base_name}*.log"), key=lambda x: x.stat().st_mtime)
        log_tail = logs[-1].read_text(encoding="utf-8", errors="replace")[-2500:] if logs else ""
        std_tail = (proc.stdout + "\n" + proc.stderr)[-1500:]
        print(
            f"!!! [OpenMVS {base_name}] FAILED rc={proc.returncode} ({dt:.1f}s)\n{std_tail}\n{log_tail}",
            flush=True,
        )
        if check:
            raise RuntimeError(f"OpenMVS {base_name} failed rc={proc.returncode}")
    else:
        print(f"<<< [OpenMVS {base_name}] DONE in {dt:.1f}s", flush=True)
    return proc.returncode, dt


def crop_mesh_to_bounds(
    mesh: o3d.geometry.TriangleMesh, bmin: np.ndarray, bmax: np.ndarray
) -> o3d.geometry.TriangleMesh:
    """Crop a triangle mesh cleanly to axis-aligned ENU bounds."""
    verts = np.asarray(mesh.vertices)
    if len(verts) == 0 or len(mesh.triangles) == 0:
        return o3d.geometry.TriangleMesh()
    in_roi = np.all((verts >= bmin) & (verts <= bmax), axis=1)
    tris = np.asarray(mesh.triangles)
    tri_keep = np.all(in_roi[tris], axis=1)
    sub = o3d.geometry.TriangleMesh()
    sub.vertices = mesh.vertices
    sub.triangles = o3d.utility.Vector3iVector(tris[tri_keep])
    if mesh.has_vertex_colors():
        sub.vertex_colors = mesh.vertex_colors
    sub.remove_unreferenced_vertices()
    sub.compute_vertex_normals()
    return sub


def run_openmvs_stage(
    undist_dir: Path,
    out_dir: Path,
    openmvs_bin: Path | None = None,
    max_threads: int = 16,
    resolution_level: int = 1,
    number_views: int = 8,
    number_views_fuse: int = 4,
    fusion_filter: int = 2,
    bbox_min: np.ndarray | None = None,
    bbox_max: np.ndarray | None = None,
    hard_min: np.ndarray | None = None,
    hard_max: np.ndarray | None = None,
    percentile_low: float = 0.2,
    percentile_high: float = 99.8,
    mesh_proximity_m: float = 0.20,
    voxel_size_m: float = 0.05,
    sor_neighbors: int = 24,
    sor_std_ratio: float = 2.0,
    ror_min_points: int = 8,
    ror_radius_m: float = 0.35,
    export_mesh: bool = True,
) -> Path:
    """Execute OpenMVS dense reconstruction, visibility-carved filtering, and PLY export."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_mvs = out_dir / "scene.mvs"
    dense_mvs = out_dir / "scene_dense.mvs"
    dense_ply = out_dir / "scene_dense.ply"
    mesh_mvs = out_dir / "scene_dense_mesh.mvs"
    mesh_ply = out_dir / "scene_dense_mesh.ply"

    # 1. InterfaceCOLMAP (-w undistorted workspace required by OpenMVS)
    rc, t_if = run_openmvs(
        openmvs_bin,
        "InterfaceCOLMAP",
        [
            "-w", str(undist_dir),
            "-i", str(undist_dir),
            "-o", str(scene_mvs),
            "--image-folder", "images",
        ],
        cwd=out_dir,
        check=False,
    )
    if rc != 0 or not scene_mvs.exists():
        _, t_if = run_openmvs(
            openmvs_bin,
            "InterfaceCOLMAP",
            [
                "-w", str(undist_dir),
                "-i", str(undist_dir / "sparse"),
                "-o", str(scene_mvs),
                "--image-folder", str(undist_dir / "images"),
            ],
            cwd=out_dir,
            check=True,
        )

    # 2. DensifyPointCloud
    _, t_dens = run_openmvs(
        openmvs_bin,
        "DensifyPointCloud",
        [
            "-w", str(undist_dir),
            "-i", str(scene_mvs),
            "-o", str(dense_mvs),
            "--max-threads", str(max_threads),
            "--resolution-level", str(resolution_level),
            "--number-views", str(number_views),
            "--number-views-fuse", str(number_views_fuse),
            "--fusion-filter", str(fusion_filter),
            "--estimate-colors", "2",
            "--estimate-normals", "2",
            "--filter-point-cloud", "1",
            "--remove-dmaps", "1",
        ],
        cwd=out_dir,
    )

    # 3. ReconstructMesh (Delaunay visibility carving)
    _, t_mesh = run_openmvs(
        openmvs_bin,
        "ReconstructMesh",
        [
            "-w", str(undist_dir),
            "-i", str(dense_mvs),
            "-o", str(mesh_mvs),
            "--max-threads", str(max_threads),
            "--remove-spurious", "20",
            "--remove-spikes", "1",
            "--close-holes", "30",
            "--smooth", "2",
        ],
        cwd=out_dir,
    )

    # 4. Crop bounds (explicit bbox or auto-percentile inside hard bounds)
    pcd_raw = o3d.io.read_point_cloud(str(dense_ply))
    pts_all = np.asarray(pcd_raw.points, dtype=np.float64)
    if len(pts_all) == 0:
        raise RuntimeError(f"DensifyPointCloud produced 0 points in {dense_ply}")

    if bbox_min is not None and bbox_max is not None:
        env_min = np.asarray(bbox_min, dtype=np.float64)
        env_max = np.asarray(bbox_max, dtype=np.float64)
    else:
        h_min = np.array([-200.0, -400.0, -80.0]) if hard_min is None else np.asarray(hard_min, dtype=np.float64)
        h_max = np.array([300.0, 200.0, 80.0]) if hard_max is None else np.asarray(hard_max, dtype=np.float64)
        in_hard = np.all((pts_all >= h_min) & (pts_all <= h_max), axis=1)
        pts_hard = pts_all[in_hard] if np.sum(in_hard) >= 500 else pts_all
        env_min = np.maximum(np.percentile(pts_hard, percentile_low, axis=0), h_min)
        env_max = np.minimum(np.percentile(pts_hard, percentile_high, axis=0), h_max)

    print(f"Crop ENU min={env_min.round(2).tolist()} max={env_max.round(2).tolist()}", flush=True)
    in_env = np.where(np.all((pts_all >= env_min) & (pts_all <= env_max), axis=1))[0]
    pcd_env = pcd_raw.select_by_index(in_env)

    mesh_raw = o3d.io.read_triangle_mesh(str(mesh_ply))
    mesh_env = crop_mesh_to_bounds(mesh_raw, env_min, env_max)

    # 5. Visibility-carved mesh proximity filtering via Open3D RaycastingScene
    if mesh_proximity_m > 0 and len(mesh_env.triangles) > 0 and len(pcd_env.points) > 0:
        t_mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(mesh_env)
        rc_scene = o3d.t.geometry.RaycastingScene()
        rc_scene.add_triangles(t_mesh_o3d)
        query = o3d.core.Tensor(np.asarray(pcd_env.points, dtype=np.float32))
        dists = rc_scene.compute_distance(query).numpy()
        keep_idx = np.where(dists <= mesh_proximity_m)[0]
        pcd_vis = pcd_env.select_by_index(keep_idx)
    else:
        pcd_vis = pcd_env

    # 6. Voxel downsample + SOR + ROR + normals
    pcd_clean = pcd_vis.voxel_down_sample(voxel_size=voxel_size_m) if voxel_size_m > 0 else pcd_vis
    if len(pcd_clean.points) > 100 and sor_neighbors > 0:
        pcd_clean, _ = pcd_clean.remove_statistical_outlier(nb_neighbors=sor_neighbors, std_ratio=sor_std_ratio)
    if len(pcd_clean.points) > 100 and ror_min_points > 0:
        pcd_clean, _ = pcd_clean.remove_radius_outlier(nb_points=ror_min_points, radius=ror_radius_m)
    if len(pcd_clean.points) > 50:
        norm_rad = max(0.4, voxel_size_m * 6.0)
        pcd_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=norm_rad, max_nn=35))

    out_ply = out_dir / "consolidated_cloud.ply"
    o3d.io.write_point_cloud(str(out_ply), pcd_clean)

    out_mesh = out_dir / "consolidated_mesh.ply"
    if export_mesh and len(mesh_env.vertices) > 0 and len(pcd_clean.points) > 0:
        mv = np.asarray(mesh_env.vertices)
        tree = cKDTree(np.asarray(pcd_clean.points))
        _, nn_idx = tree.query(mv, k=1, workers=-1)
        if pcd_clean.has_colors():
            mesh_env.vertex_colors = o3d.utility.Vector3dVector(np.asarray(pcd_clean.colors)[nn_idx])
        mesh_env.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out_mesh), mesh_env)

    info = {
        "interface_s": round(t_if, 1),
        "densify_s": round(t_dens, 1),
        "mesh_s": round(t_mesh, 1),
        "env_min": [round(float(x), 3) for x in env_min],
        "env_max": [round(float(x), 3) for x in env_max],
        "raw_dense_points": len(pcd_raw.points),
        "cropped_points": len(pcd_env.points),
        "mesh_filtered_points": len(pcd_vis.points),
        "consolidated_points": len(pcd_clean.points),
        "mesh_vertices": len(mesh_env.vertices),
        "mesh_triangles": len(mesh_env.triangles),
        "out_ply": str(out_ply),
    }
    with open(out_dir / "mvs_summary.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print("OpenMVS summary:", json.dumps(info, indent=2), flush=True)
    return out_ply


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for Stage 3."""
    parser = argparse.ArgumentParser(
        description="Run OpenMVS dense reconstruction + Delaunay visibility-carved mesh proximity filtering."
    )
    parser.add_argument("--undistorted", required=True, type=Path, help="COLMAP undistorted workspace directory.")
    parser.add_argument("--out", required=True, type=Path, help="Output OpenMVS working directory.")
    parser.add_argument("--openmvs-bin", type=Path, default=None, help="Directory containing OpenMVS executables.")
    parser.add_argument("--max-threads", type=int, default=16, help="Thread count for OpenMVS (default: 16).")
    parser.add_argument("--resolution-level", type=int, default=1, help="OpenMVS resolution level (default: 1).")
    parser.add_argument("--number-views", type=int, default=8, help="Max views per depth map (default: 8).")
    parser.add_argument("--number-views-fuse", type=int, default=4, help="Min consistent views to fuse (default: 4).")
    parser.add_argument("--fusion-filter", type=int, default=2, help="OpenMVS fusion filter level (default: 2).")
    parser.add_argument(
        "--bbox",
        nargs=6,
        type=float,
        default=None,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        help="Optional explicit ENU crop bounding box; defaults to auto-percentile clamping.",
    )
    parser.add_argument("--mesh-proximity", type=float, default=0.20, help="Max distance to carved mesh (m, default: 0.20).")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="Output voxel downsample size (m, default: 0.05).")
    parser.add_argument("--no-mesh-export", action="store_true", help="Skip exporting vertex-colored consolidated_mesh.ply.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args(argv)
    bmin = np.array(args.bbox[:3], dtype=np.float64) if args.bbox else None
    bmax = np.array(args.bbox[3:], dtype=np.float64) if args.bbox else None
    run_openmvs_stage(
        undist_dir=args.undistorted,
        out_dir=args.out,
        openmvs_bin=args.openmvs_bin,
        max_threads=args.max_threads,
        resolution_level=args.resolution_level,
        number_views=args.number_views,
        number_views_fuse=args.number_views_fuse,
        fusion_filter=args.fusion_filter,
        bbox_min=bmin,
        bbox_max=bmax,
        mesh_proximity_m=args.mesh_proximity,
        voxel_size_m=args.voxel_size,
        export_mesh=not args.no_mesh_export,
    )


if __name__ == "__main__":
    main()
