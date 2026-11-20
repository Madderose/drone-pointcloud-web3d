#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2: COLMAP GPU SIFT SfM + Dual Mapper Selection + Single Global Rigid Sim(3) ENU Alignment."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.common import ransac_irls_sim3


def resolve_colmap_bin(colmap_bin: str) -> str:
    """Resolve COLMAP executable or .bat launcher across Windows and Linux."""
    cand = Path(colmap_bin)
    if cand.exists():
        return str(cand)
    found = shutil.which(colmap_bin)
    if found:
        return found
    raise FileNotFoundError(
        f"COLMAP binary not found ({colmap_bin}). Pass --colmap-bin /path/to/colmap or COLMAP.bat."
    )


def run_colmap(colmap_exe: str, args: list[str], step_name: str, check: bool = True) -> tuple[int, float]:
    """Invoke a COLMAP CLI subcommand and return (returncode, elapsed_seconds)."""
    cmd = [colmap_exe] + args
    print(f"\n>>> [{step_name}] {' '.join(args[:4])} ...", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    if proc.returncode != 0:
        tail = (proc.stdout + "\n" + proc.stderr)[-2500:]
        print(f"!!! [{step_name}] FAILED rc={proc.returncode} ({dt:.1f}s):\n{tail}", flush=True)
        if check:
            raise RuntimeError(f"COLMAP {step_name} failed with rc={proc.returncode}")
    else:
        print(f"<<< [{step_name}] DONE in {dt:.1f}s", flush=True)
    return proc.returncode, dt


def evaluate_model_dir(model_root: Path, src_by_name: dict[str, str]) -> list[dict]:
    """Inspect all sub-models (0, 1, ...) inside a COLMAP sparse directory."""
    candidates: list[dict] = []
    if not model_root.exists():
        return candidates
    for sub in sorted(model_root.iterdir()):
        if not sub.is_dir():
            continue
        if not ((sub / "images.bin").exists() or (sub / "images.txt").exists()):
            continue
        rec = pycolmap.Reconstruction(str(sub))
        n_reg = rec.num_reg_images()
        n_pts = rec.num_points3D()
        reproj = rec.compute_mean_reprojection_error() if n_pts > 0 else 999.0
        track = rec.compute_mean_track_length() if n_pts > 0 else 0.0
        sources = set()
        for im in rec.images.values():
            if im.has_pose:
                nm = im.name.replace("\\", "/")
                if nm in src_by_name:
                    sources.add(src_by_name[nm])
        candidates.append(
            {
                "path": sub,
                "n_reg": n_reg,
                "n_sources": len(sources),
                "sources": sorted(sources),
                "n_pts": n_pts,
                "reproj": reproj,
                "track": track,
            }
        )
    return candidates


def run_sfm(
    dataset_dir: Path,
    sfm_dir: Path,
    colmap_bin: str = "colmap",
    camera_model: str = "OPENCV",
    single_camera_per_folder: bool = True,
    max_image_size: int = 2048,
    max_num_features: int = 8192,
    use_gpu: bool = True,
    estimate_affine_shape: bool = False,
    domain_size_pooling: bool = False,
    guided_matching: bool = True,
    min_track_len: int = 3,
    max_reproj_error: float = 2.0,
    reference_sparse: Path | None = None,
    sim3_inlier_thresh: float = 2.0,
) -> Path:
    """Run Stage 2 SfM and return the undistorted MVS workspace path."""
    colmap_exe = resolve_colmap_bin(colmap_bin)
    sfm_dir.mkdir(parents=True, exist_ok=True)
    img_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    pairs_txt = dataset_dir / "pairs.txt"

    db_path = sfm_dir / "database.db"
    if db_path.exists():
        db_path.unlink()

    with open(dataset_dir / "frames_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    enu_by_name = {fr["image_name"]: np.array(fr["enu"], dtype=np.float64) for fr in meta["frames"]}
    src_by_name = {fr["image_name"]: fr["source"] for fr in meta["frames"]}

    # 1. Native GPU SIFT feature extraction (avoids ONNX CUDA EP DLL clashes)
    gpu_flag = "1" if use_gpu else "0"
    run_colmap(
        colmap_exe,
        [
            "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(img_dir),
            "--ImageReader.mask_path", str(mask_dir),
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera_per_folder", "1" if single_camera_per_folder else "0",
            "--ImageReader.default_focal_length_factor", "0.75",
            "--FeatureExtraction.type", "SIFT",
            "--FeatureExtraction.max_image_size", str(max_image_size),
            "--FeatureExtraction.use_gpu", gpu_flag,
            "--SiftExtraction.estimate_affine_shape", "1" if estimate_affine_shape else "0",
            "--SiftExtraction.domain_size_pooling", "1" if domain_size_pooling else "0",
            "--SiftExtraction.max_num_features", str(max_num_features),
        ],
        "feature_extractor_sift",
    )

    # 2. Match pairs.txt via matches_importer (or exhaustive_matcher if pairs.txt absent)
    if pairs_txt.exists():
        run_colmap(
            colmap_exe,
            [
                "matches_importer",
                "--database_path", str(db_path),
                "--match_list_path", str(pairs_txt),
                "--match_type", "pairs",
                "--FeatureMatching.type", "SIFT_BRUTEFORCE",
                "--FeatureMatching.use_gpu", gpu_flag,
                "--FeatureMatching.guided_matching", "1" if guided_matching else "0",
            ],
            "matches_importer_pairs",
        )
    else:
        run_colmap(
            colmap_exe,
            [
                "exhaustive_matcher",
                "--database_path", str(db_path),
                "--FeatureMatching.type", "SIFT_BRUTEFORCE",
                "--FeatureMatching.use_gpu", gpu_flag,
                "--FeatureMatching.guided_matching", "1" if guided_matching else "0",
            ],
            "exhaustive_matcher_sift",
        )

    # 3. Dual mapper selection: global_mapper + incremental mapper
    sparse_global = sfm_dir / "sparse_global"
    sparse_inc = sfm_dir / "sparse_inc"
    for d in (sparse_global, sparse_inc):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    run_colmap(
        colmap_exe,
        [
            "global_mapper",
            "--database_path", str(db_path),
            "--image_path", str(img_dir),
            "--output_path", str(sparse_global),
            "--GlobalMapper.ba_refine_focal_length", "1",
            "--GlobalMapper.ba_refine_principal_point", "0",
            "--GlobalMapper.ba_refine_extra_params", "1",
        ],
        "global_mapper",
        check=False,
    )

    run_colmap(
        colmap_exe,
        [
            "mapper",
            "--database_path", str(db_path),
            "--image_path", str(img_dir),
            "--output_path", str(sparse_inc),
            "--Mapper.ba_refine_focal_length", "1",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params", "1",
            "--Mapper.ba_global_max_num_iterations", "100",
        ],
        "incremental_mapper",
        check=False,
    )

    all_cands = []
    for label, root in [("global_mapper", sparse_global), ("incremental_mapper", sparse_inc)]:
        for c in evaluate_model_dir(root, src_by_name):
            c["mapper"] = label
            all_cands.append(c)

    if not all_cands:
        raise RuntimeError("Both global_mapper and incremental_mapper produced 0 valid models.")

    print("\n=== SfM CANDIDATE MODELS ===")
    for c in all_cands:
        print(
            f"  [{c['mapper']}] {c['path'].name}: reg={c['n_reg']}/{len(enu_by_name)}, "
            f"sources={c['n_sources']}, pts3D={c['n_pts']}, reproj={c['reproj']:.3f}px"
        )

    all_cands.sort(key=lambda c: (-c["n_sources"], -c["n_reg"], c["reproj"]))
    best = all_cands[0]
    print(f"Selected winner SfM model: [{best['mapper']}] {best['path']}")

    # 4. Point filtering + Bundle Adjustment
    sparse_filt = sfm_dir / "sparse_filtered"
    sparse_ba = sfm_dir / "sparse_ba"
    for d in (sparse_filt, sparse_ba):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    run_colmap(
        colmap_exe,
        [
            "point_filtering",
            "--input_path", str(best["path"]),
            "--output_path", str(sparse_filt),
            "--min_track_len", str(min_track_len),
            "--max_reproj_error", str(max_reproj_error),
        ],
        "point_filtering",
    )

    run_colmap(
        colmap_exe,
        [
            "bundle_adjuster",
            "--input_path", str(sparse_filt),
            "--output_path", str(sparse_ba),
            "--BundleAdjustment.refine_focal_length", "1",
            "--BundleAdjustment.refine_principal_point", "0",
            "--BundleAdjustment.refine_extra_params", "1",
        ],
        "bundle_adjuster",
    )

    # 5. Single global rigid Sim(3) alignment (either to reference_sparse shared cameras or ENU GPS)
    rec = pycolmap.Reconstruction(str(sparse_ba))
    alignment_mode = "gps_enu"
    shared_ref_rmse = None

    if reference_sparse is not None and reference_sparse.exists():
        ref_rec = pycolmap.Reconstruction(str(reference_sparse))
        ref_centers = {
            im.name.replace("\\", "/"): np.asarray(im.projection_center(), dtype=np.float64)
            for im in ref_rec.images.values()
            if im.has_pose
        }
        src_pts, dst_pts = [], []
        for im in rec.images.values():
            if not im.has_pose:
                continue
            nm = im.name.replace("\\", "/")
            if nm in ref_centers:
                src_pts.append(np.asarray(im.projection_center(), dtype=np.float64))
                dst_pts.append(ref_centers[nm])
        if len(src_pts) >= 10:
            alignment_mode = "reference_sparse_sim3"
            s, r_mat, t_vec, inl = ransac_irls_sim3(
                np.array(src_pts),
                np.array(dst_pts),
                inlier_thresh=min(0.5, sim3_inlier_thresh),
                huber_delta=0.25,
            )
            sim3 = pycolmap.Sim3d(s, pycolmap.Rotation3d(r_mat), t_vec)
            rec.transform(sim3)
            post_shared = [
                float(np.linalg.norm(np.asarray(im.projection_center(), dtype=np.float64) - ref_centers[im.name.replace("\\", "/")]))
                for im in rec.images.values()
                if im.has_pose and im.name.replace("\\", "/") in ref_centers
            ]
            shared_ref_rmse = round(float(np.sqrt(np.mean(np.array(post_shared) ** 2))), 4)
        else:
            print(
                f"WARN: Only {len(src_pts)} shared cameras with {reference_sparse}; falling back to GPS ENU Sim(3)."
            )

    if alignment_mode == "gps_enu":
        sfm_centers, gps_centers = [], []
        for im in rec.images.values():
            if not im.has_pose:
                continue
            nm = im.name.replace("\\", "/")
            if nm in enu_by_name:
                sfm_centers.append(np.asarray(im.projection_center(), dtype=np.float64))
                gps_centers.append(enu_by_name[nm])
        s, r_mat, t_vec, inl = ransac_irls_sim3(
            np.array(sfm_centers),
            np.array(gps_centers),
            inlier_thresh=sim3_inlier_thresh,
        )
        sim3 = pycolmap.Sim3d(s, pycolmap.Rotation3d(r_mat), t_vec)
        rec.transform(sim3)

    errs_all = []
    errs_by_src: dict[str, list[float]] = {}
    counts_by_src: dict[str, int] = {}
    for im in rec.images.values():
        if not im.has_pose:
            continue
        nm = im.name.replace("\\", "/")
        src = src_by_name.get(nm, "unknown")
        counts_by_src[src] = counts_by_src.get(src, 0) + 1
        if nm in enu_by_name:
            err = float(np.linalg.norm(np.asarray(im.projection_center(), dtype=np.float64) - enu_by_name[nm]))
            errs_all.append(err)
            errs_by_src.setdefault(src, []).append(err)

    errs_arr = np.array(errs_all) if errs_all else np.array([0.0])
    sparse_enu_0 = sfm_dir / "sparse_enu" / "0"
    if sparse_enu_0.parent.exists():
        shutil.rmtree(sparse_enu_0.parent)
    sparse_enu_0.mkdir(parents=True)
    rec.write(str(sparse_enu_0))
    rec.export_PLY(str(sfm_dir / "sparse_enu_points.ply"))

    summary = {
        "alignment_mode": alignment_mode,
        "selected_mapper": best["mapper"],
        "total_images": len(enu_by_name),
        "registered_images": rec.num_reg_images(),
        "registered_pct": round(100.0 * rec.num_reg_images() / max(1, len(enu_by_name)), 1),
        "registered_by_source": counts_by_src,
        "num_points3D": rec.num_points3D(),
        "mean_reproj_error_px": round(float(rec.compute_mean_reprojection_error()), 4),
        "mean_track_length": round(float(rec.compute_mean_track_length()), 3),
        "sim3_scale": round(float(s), 6),
        "sim3_inliers": int(np.sum(inl)),
        "shared_ref_rmse_m": shared_ref_rmse,
        "gps_rmse_m": round(float(np.sqrt(np.mean(errs_arr**2))), 3),
        "gps_median_err_m": round(float(np.median(errs_arr)), 3),
        "gps_rmse_by_source_m": {
            k: round(float(np.sqrt(np.mean(np.array(v) ** 2))), 3)
            for k, v in errs_by_src.items()
        },
    }
    with open(sfm_dir / "sfm_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== FINAL ENU SfM SUMMARY ===")
    print(json.dumps(summary, indent=2))

    # 6. Export undistorted PINHOLE workspace for OpenMVS
    undist_dir = sfm_dir / "undistorted"
    if undist_dir.exists():
        shutil.rmtree(undist_dir)
    run_colmap(
        colmap_exe,
        [
            "image_undistorter",
            "--image_path", str(img_dir),
            "--input_path", str(sparse_enu_0),
            "--output_path", str(undist_dir),
            "--output_type", "COLMAP",
            "--max_image_size", str(max_image_size),
        ],
        "image_undistorter",
    )
    print(f"Undistorted MVS workspace ready at: {undist_dir}")
    return undist_dir


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for Stage 2."""
    parser = argparse.ArgumentParser(
        description="Run COLMAP GPU SIFT SfM, dual mapper selection, rigid Sim(3) ENU alignment, and undistortion."
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Dataset directory produced by Stage 1.")
    parser.add_argument("--out", required=True, type=Path, help="Output SfM working directory.")
    parser.add_argument("--colmap-bin", default="colmap", help="Path to colmap binary or COLMAP.bat.")
    parser.add_argument("--camera-model", default="OPENCV", help="COLMAP camera model (default: OPENCV).")
    parser.add_argument("--max-image-size", type=int, default=2048, help="Max image dimension (default: 2048).")
    parser.add_argument("--max-num-features", type=int, default=8192, help="Max SIFT features per image (default: 8192).")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU SIFT extraction/matching.")
    parser.add_argument("--min-track-len", type=int, default=3, help="COLMAP point_filtering min track length.")
    parser.add_argument("--max-reproj-error", type=float, default=2.0, help="COLMAP point_filtering max reproj px.")
    parser.add_argument(
        "--reference-sparse",
        type=Path,
        default=None,
        help="Optional reference sparse_enu/0 folder to lock a wide scan onto a core reconstruction via shared cameras.",
    )
    parser.add_argument("--sim3-inlier-thresh", type=float, default=2.0, help="RANSAC Sim(3) inlier threshold (m).")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args(argv)
    run_sfm(
        dataset_dir=args.dataset,
        sfm_dir=args.out,
        colmap_bin=args.colmap_bin,
        camera_model=args.camera_model,
        max_image_size=args.max_image_size,
        max_num_features=args.max_num_features,
        use_gpu=not args.no_gpu,
        min_track_len=args.min_track_len,
        max_reproj_error=args.max_reproj_error,
        reference_sparse=args.reference_sparse,
        sim3_inlier_thresh=args.sim3_inlier_thresh,
    )


if __name__ == "__main__":
    main()
