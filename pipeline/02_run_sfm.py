#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2: COLMAP GPU SIFT SfM, Smart Early-Accept Gate, Snapshots & Rigid Sim(3) Alignment."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.common import ransac_irls_sim3
from pipeline.progress import ProgressTracker, make_colmap_parser, stream_command


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


def detect_colmap_capabilities(colmap_exe: str) -> dict[str, str | bool | None]:
    """Probe `colmap mapper -h` once to adapt flags cleanly across COLMAP 3.x and 4.x."""
    proc = subprocess.run(
        [colmap_exe, "mapper", "-h"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    help_txt = (proc.stdout or "") + "\n" + (proc.stderr or "")
    snap_flag: str | None = None
    if "--Mapper.snapshot_frames_freq" in help_txt:
        snap_flag = "--Mapper.snapshot_frames_freq"
    elif "--Mapper.snapshot_images_freq" in help_txt:
        snap_flag = "--Mapper.snapshot_images_freq"

    return {
        "has_log_target": "--log_target" in help_txt,
        "has_log_color": "--log_color" in help_txt,
        "snapshot_freq_flag": snap_flag,
    }


def _db_is_complete(db_path: Path, expected_images: int) -> bool:
    """Return True if database.db already has extracted features and verified matches for all images."""
    if not db_path.exists():
        return False
    con = None
    try:
        con = sqlite3.connect(str(db_path))
        n_img = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        n_kp = con.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0]
        n_tvg = con.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
        return n_img == expected_images and n_kp == expected_images and n_tvg > 0
    except Exception:
        return False
    finally:
        if con is not None:
            con.close()


def _run_colmap_logged(
    colmap_exe: str,
    caps: dict[str, str | bool | None],
    args: list[str],
    step_name: str,
    *,
    sfm_dir: Path,
    tracker: ProgressTracker | None,
    total_images: int | None = None,
    unit: str = "phase",
    snapshot_dir: Path | None = None,
    timeout_s: float | None = None,
    check: bool = True,
) -> tuple[int, float, bool]:
    """Stream a COLMAP subcommand live to logs/<step>.log and update ProgressTracker."""
    cmd = [colmap_exe] + args
    if caps.get("has_log_target"):
        cmd += ["--log_target", "stderr"]
    if caps.get("has_log_color"):
        cmd += ["--log_color", "0"]

    env = os.environ.copy()
    env["GLOG_logtostderr"] = "1"

    print(f"\n>>> [{step_name}] {' '.join(args[:4])} ...", flush=True)
    log_file = sfm_dir / "logs" / f"{step_name}.log"
    res = stream_command(
        cmd,
        stage="stage2_sfm",
        step_name=step_name,
        tracker=tracker,
        log_file=log_file,
        env=env,
        line_parser=make_colmap_parser(step_name, total_images=total_images),
        total_units=total_images if unit == "images" else None,
        unit=unit,
        snapshot_dir=snapshot_dir,
        timeout_s=timeout_s if (timeout_s and timeout_s > 0) else None,
        check=check,
    )
    if res.returncode == 0 and not res.timed_out:
        print(f"<<< [{step_name}] DONE in {res.elapsed_s:.1f}s (log: {log_file.name})", flush=True)
    elif res.timed_out:
        print(f"!!! [{step_name}] TIMED OUT after {res.elapsed_s:.1f}s (any flushed model preserved)", flush=True)
    return res.returncode, res.elapsed_s, res.timed_out


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
        try:
            rec = pycolmap.Reconstruction(str(sub))
        except Exception as exc:
            print(f"WARN: Skipping unreadable sub-model {sub}: {exc}", flush=True)
            continue
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
    *,
    snapshot_freq: int = 25,
    early_accept_min_reg: float = 0.95,
    early_accept_max_reproj: float = 1.2,
    global_mapper_timeout_s: float = 900.0,
    force_both_mappers: bool = False,
    resume: bool = False,
    tracker: ProgressTracker | None = None,
    progress_file: Path | None = None,
    stall_warn_s: float = 180.0,
) -> Path:
    """Run Stage 2 SfM and return the undistorted MVS workspace path."""
    t_stage0 = time.monotonic()
    colmap_exe = resolve_colmap_bin(colmap_bin)
    caps = detect_colmap_capabilities(colmap_exe)
    sfm_dir.mkdir(parents=True, exist_ok=True)
    img_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    pairs_txt = dataset_dir / "pairs.txt"

    owns_tracker = tracker is None
    if tracker is None:
        tracker = ProgressTracker(progress_file or (sfm_dir / "progress.json"), stall_warn_s=stall_warn_s)

    try:
        with open(dataset_dir / "frames_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        enu_by_name = {fr["image_name"]: np.array(fr["enu"], dtype=np.float64) for fr in meta["frames"]}
        src_by_name = {fr["image_name"]: fr["source"] for fr in meta["frames"]}
        total_images = len(enu_by_name)
        total_sources = max(1, len(set(src_by_name.values())))

        db_path = sfm_dir / "database.db"
        reused_db = resume and _db_is_complete(db_path, total_images)
        gpu_flag = "1" if use_gpu else "0"

        if reused_db:
            print(f"[resume] Reusing intact COLMAP database ({total_images} imgs): {db_path}", flush=True)
            tracker.finish_step("feature_extractor_sift", status="skipped", elapsed_s=0.0, message="Reused database.db")
            match_step_label = "matches_importer_pairs" if pairs_txt.exists() else "exhaustive_matcher_sift"
            tracker.finish_step(match_step_label, status="skipped", elapsed_s=0.0, message="Reused database.db")
        else:
            if db_path.exists():
                db_path.unlink()

            # 1. Native GPU SIFT feature extraction (avoids ONNX CUDA EP DLL clashes)
            _run_colmap_logged(
                colmap_exe,
                caps,
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
                sfm_dir=sfm_dir,
                tracker=tracker,
                total_images=total_images,
                unit="images",
            )

            # 2. Match pairs.txt via matches_importer (or exhaustive_matcher if pairs.txt absent)
            if pairs_txt.exists():
                _run_colmap_logged(
                    colmap_exe,
                    caps,
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
                    sfm_dir=sfm_dir,
                    tracker=tracker,
                    unit="blocks",
                )
            else:
                _run_colmap_logged(
                    colmap_exe,
                    caps,
                    [
                        "exhaustive_matcher",
                        "--database_path", str(db_path),
                        "--FeatureMatching.type", "SIFT_BRUTEFORCE",
                        "--FeatureMatching.use_gpu", gpu_flag,
                        "--FeatureMatching.guided_matching", "1" if guided_matching else "0",
                    ],
                    "exhaustive_matcher_sift",
                    sfm_dir=sfm_dir,
                    tracker=tracker,
                    unit="blocks",
                )

        # 3. Mapper execution: global_mapper first + smart early-accept gate
        sparse_global = sfm_dir / "sparse_global"
        sparse_inc = sfm_dir / "sparse_inc"
        for d in (sparse_global, sparse_inc):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

        _run_colmap_logged(
            colmap_exe,
            caps,
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
            sfm_dir=sfm_dir,
            tracker=tracker,
            timeout_s=global_mapper_timeout_s,
            check=False,
        )

        global_cands = evaluate_model_dir(sparse_global, src_by_name)
        for c in global_cands:
            c["mapper"] = "global_mapper"

        early_accepted = False
        min_reg_count = math.ceil(early_accept_min_reg * total_images)
        if global_cands and not force_both_mappers:
            global_cands.sort(key=lambda c: (-c["n_sources"], -c["n_reg"], c["reproj"]))
            top_g = global_cands[0]
            if (
                top_g["n_reg"] >= min_reg_count
                and top_g["reproj"] <= early_accept_max_reproj
                and top_g["n_sources"] >= total_sources
            ):
                early_accepted = True
                pct_g = 100.0 * top_g["n_reg"] / max(1, total_images)
                msg = (
                    f"[early-accept] global_mapper registered {top_g['n_reg']}/{total_images} "
                    f"({pct_g:.1f}%, reproj={top_g['reproj']:.3f}px, sources={top_g['n_sources']}/{total_sources}) "
                    "— skipping redundant incremental_mapper"
                )
                print(msg, flush=True)
                tracker.finish_step("incremental_mapper", status="skipped", elapsed_s=0.0, message=msg)

        inc_cands: list[dict] = []
        snapshot_dir: Path | None = None
        if not early_accepted:
            inc_args = [
                "mapper",
                "--database_path", str(db_path),
                "--image_path", str(img_dir),
                "--output_path", str(sparse_inc),
                "--Mapper.ba_refine_focal_length", "1",
                "--Mapper.ba_refine_principal_point", "0",
                "--Mapper.ba_refine_extra_params", "1",
                "--Mapper.ba_global_max_num_iterations", "100",
            ]
            if snapshot_freq > 0:
                snapshot_dir = sfm_dir / "snapshots"
                if snapshot_dir.exists():
                    shutil.rmtree(snapshot_dir)
                snapshot_dir.mkdir(parents=True)
                inc_args += ["--Mapper.snapshot_path", str(snapshot_dir)]
                if caps.get("snapshot_freq_flag"):
                    inc_args += [str(caps["snapshot_freq_flag"]), str(snapshot_freq)]

            _run_colmap_logged(
                colmap_exe,
                caps,
                inc_args,
                "incremental_mapper",
                sfm_dir=sfm_dir,
                tracker=tracker,
                total_images=total_images,
                unit="images",
                snapshot_dir=snapshot_dir,
                check=False,
            )
            for c in evaluate_model_dir(sparse_inc, src_by_name):
                c["mapper"] = "incremental_mapper"
                inc_cands.append(c)

        all_cands = global_cands + inc_cands
        if not all_cands:
            raise RuntimeError("Both global_mapper and incremental_mapper produced 0 valid models.")

        print("\n=== SfM CANDIDATE MODELS ===")
        for c in all_cands:
            print(
                f"  [{c['mapper']}] {c['path'].name}: reg={c['n_reg']}/{total_images}, "
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

        _run_colmap_logged(
            colmap_exe,
            caps,
            [
                "point_filtering",
                "--input_path", str(best["path"]),
                "--output_path", str(sparse_filt),
                "--min_track_len", str(min_track_len),
                "--max_reproj_error", str(max_reproj_error),
            ],
            "point_filtering",
            sfm_dir=sfm_dir,
            tracker=tracker,
        )

        _run_colmap_logged(
            colmap_exe,
            caps,
            [
                "bundle_adjuster",
                "--input_path", str(sparse_filt),
                "--output_path", str(sparse_ba),
                "--BundleAdjustment.refine_focal_length", "1",
                "--BundleAdjustment.refine_principal_point", "0",
                "--BundleAdjustment.refine_extra_params", "1",
            ],
            "bundle_adjuster",
            sfm_dir=sfm_dir,
            tracker=tracker,
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
                    float(
                        np.linalg.norm(
                            np.asarray(im.projection_center(), dtype=np.float64)
                            - ref_centers[im.name.replace("\\", "/")]
                        )
                    )
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

        # 6. Export undistorted PINHOLE workspace for OpenMVS
        undist_dir = sfm_dir / "undistorted"
        if undist_dir.exists():
            shutil.rmtree(undist_dir)
        _run_colmap_logged(
            colmap_exe,
            caps,
            [
                "image_undistorter",
                "--image_path", str(img_dir),
                "--input_path", str(sparse_enu_0),
                "--output_path", str(undist_dir),
                "--output_type", "COLMAP",
                "--max_image_size", str(max_image_size),
            ],
            "image_undistorter",
            sfm_dir=sfm_dir,
            tracker=tracker,
            total_images=rec.num_reg_images(),
            unit="images",
        )

        summary = {
            "alignment_mode": alignment_mode,
            "selected_mapper": best["mapper"],
            "early_accepted_global": early_accepted,
            "snapshot_dir": str(snapshot_dir) if snapshot_dir else None,
            "total_images": total_images,
            "registered_images": rec.num_reg_images(),
            "registered_pct": round(100.0 * rec.num_reg_images() / max(1, total_images), 1),
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
            "elapsed_s": round(time.monotonic() - t_stage0, 1),
        }
        with open(sfm_dir / "sfm_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("\n=== FINAL ENU SfM SUMMARY ===")
        print(json.dumps(summary, indent=2))
        print(f"Undistorted MVS workspace ready at: {undist_dir}")

        if owns_tracker:
            tracker.finish_pipeline(status="completed", message="Stage 2 SfM complete")
        return undist_dir
    except Exception as exc:
        if owns_tracker:
            tracker.finish_pipeline(status="failed", message=str(exc))
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for Stage 2."""
    parser = argparse.ArgumentParser(
        description="Run COLMAP GPU SIFT SfM, dual/early-accept mapper, snapshots, rigid Sim(3) ENU alignment, and undistortion."
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
    parser.add_argument("--snapshot-freq", type=int, default=25, help="Save incremental mapper snapshots every N images (0=off).")
    parser.add_argument(
        "--early-accept-min-reg",
        type=float,
        default=0.95,
        help="Skip incremental_mapper if global_mapper registers >= this fraction (default: 0.95).",
    )
    parser.add_argument(
        "--early-accept-max-reproj",
        type=float,
        default=1.2,
        help="Skip incremental_mapper only when global_mapper reproj error <= px (default: 1.2).",
    )
    parser.add_argument(
        "--global-mapper-timeout",
        type=float,
        default=900.0,
        help="Max seconds for global_mapper before terminating cleanly (0=unlimited, default: 900).",
    )
    parser.add_argument(
        "--force-both-mappers",
        action="store_true",
        help="Always run incremental_mapper even if global_mapper meets early-accept thresholds.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse existing intact database.db if complete.")
    parser.add_argument("--progress-file", type=Path, default=None, help="Path for live atomic progress.json.")
    parser.add_argument("--stall-warn-s", type=float, default=180.0, help="Stall warning threshold recorded in progress.json.")
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
        snapshot_freq=args.snapshot_freq,
        early_accept_min_reg=args.early_accept_min_reg,
        early_accept_max_reproj=args.early_accept_max_reproj,
        global_mapper_timeout_s=args.global_mapper_timeout,
        force_both_mappers=args.force_both_mappers,
        resume=args.resume,
        progress_file=args.progress_file,
        stall_warn_s=args.stall_warn_s,
    )


if __name__ == "__main__":
    main()
