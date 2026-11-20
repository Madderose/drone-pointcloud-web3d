#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end YAML orchestrator running Stages 1-4 (Dataset -> SfM -> OpenMVS -> Audit)."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

stage1 = importlib.import_module("pipeline.01_prepare_dataset")
stage2 = importlib.import_module("pipeline.02_run_sfm")
stage3 = importlib.import_module("pipeline.03_run_openmvs")
stage4 = importlib.import_module("pipeline.04_audit_cloud")


def _resolve_path(base_dir: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else (base_dir / p).resolve()


def run_pipeline(config_path: Path, from_stage: int = 1, to_stage: int = 4) -> None:
    """Load YAML config and execute requested pipeline stages sequentially."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.resolve().parent

    proj = cfg.get("project", {})
    workdir = _resolve_path(base_dir, proj.get("workdir", "./work"))
    assert workdir is not None
    dataset_dir = workdir / "dataset"
    sfm_dir = workdir / "sfm"
    mvs_dir = workdir / "mvs"
    eval_dir = workdir / "eval"

    if from_stage <= 1 <= to_stage:
        enu_cfg = cfg.get("enu_origin", {})
        is_manual = str(enu_cfg.get("mode", "auto")).lower() == "manual"
        inputs_cfg = cfg.get("inputs", {})
        prep_cfg = cfg.get("prepare", {})

        video_specs = []
        for idx, item in enumerate(inputs_cfg.get("videos", [])):
            vp = _resolve_path(base_dir, item["path"])
            srt_p = _resolve_path(base_dir, item.get("srt"))
            video_specs.append(
                {
                    "path": vp,
                    "srt": str(srt_p) if srt_p else None,
                    "start_frame": int(item.get("start_frame", 0)),
                    "end_frame": int(item.get("end_frame", -1)),
                    "stride": int(item.get("stride", 15)),
                    "prefix": item.get("prefix", f"v{idx}"),
                    "tag": item.get("tag", f"video_{idx}"),
                }
            )
        still_paths = [_resolve_path(base_dir, s) for s in inputs_cfg.get("stills", []) if s]

        stage1.prepare_dataset(
            video_specs=video_specs,
            still_paths=[p for p in still_paths if p is not None],
            out_dir=dataset_dir,
            width=int(prep_cfg.get("width", 2048)),
            height=int(prep_cfg.get("height", 1152)),
            jpeg_quality=int(prep_cfg.get("jpeg_quality", 96)),
            origin_lat=float(enu_cfg["lat0"]) if is_manual else None,
            origin_lon=float(enu_cfg["lon0"]) if is_manual else None,
            origin_alt=float(enu_cfg["abs_alt_ref"]) if is_manual else None,
            temporal_window=int(prep_cfg.get("temporal_window", 12)),
            gps_neighbors=int(prep_cfg.get("gps_neighbors", 45)),
            backlit_lap_thresh=float(prep_cfg.get("backlit_laplacian_thresh", 1800.0)),
            make_contact_sheet=bool(prep_cfg.get("render_contact_sheet", True)),
        )

    if from_stage <= 2 <= to_stage:
        sfm_cfg = cfg.get("sfm", {})
        ref_sparse = _resolve_path(base_dir, sfm_cfg.get("reference_sparse"))
        stage2.run_sfm(
            dataset_dir=dataset_dir,
            sfm_dir=sfm_dir,
            colmap_bin=str(sfm_cfg.get("colmap_bin", "colmap")),
            camera_model=str(sfm_cfg.get("camera_model", "OPENCV")),
            single_camera_per_folder=bool(sfm_cfg.get("single_camera_per_folder", True)),
            max_image_size=int(sfm_cfg.get("max_image_size", 2048)),
            max_num_features=int(sfm_cfg.get("max_num_features", 8192)),
            use_gpu=bool(sfm_cfg.get("use_gpu", True)),
            estimate_affine_shape=bool(sfm_cfg.get("estimate_affine_shape", False)),
            domain_size_pooling=bool(sfm_cfg.get("domain_size_pooling", False)),
            guided_matching=bool(sfm_cfg.get("guided_matching", True)),
            min_track_len=int(sfm_cfg.get("min_track_len", 3)),
            max_reproj_error=float(sfm_cfg.get("max_reproj_error", 2.0)),
            reference_sparse=ref_sparse,
            sim3_inlier_thresh=float(sfm_cfg.get("sim3_inlier_thresh", 2.0)),
        )

    if from_stage <= 3 <= to_stage:
        mvs_cfg = cfg.get("openmvs", {})
        omvs_bin = _resolve_path(base_dir, mvs_cfg.get("openmvs_bin")) if mvs_cfg.get("openmvs_bin") else None
        crop_mode = str(mvs_cfg.get("crop_mode", "auto")).lower()
        bmin = np.array(mvs_cfg["bbox_min"], dtype=np.float64) if crop_mode == "manual" and "bbox_min" in mvs_cfg else None
        bmax = np.array(mvs_cfg["bbox_max"], dtype=np.float64) if crop_mode == "manual" and "bbox_max" in mvs_cfg else None
        stage3.run_openmvs_stage(
            undist_dir=sfm_dir / "undistorted",
            out_dir=mvs_dir,
            openmvs_bin=omvs_bin,
            max_threads=int(mvs_cfg.get("max_threads", 16)),
            resolution_level=int(mvs_cfg.get("resolution_level", 1)),
            number_views=int(mvs_cfg.get("number_views", 8)),
            number_views_fuse=int(mvs_cfg.get("number_views_fuse", 4)),
            fusion_filter=int(mvs_cfg.get("fusion_filter", 2)),
            bbox_min=bmin,
            bbox_max=bmax,
            hard_min=np.array(mvs_cfg.get("hard_min", [-200.0, -400.0, -80.0]), dtype=np.float64),
            hard_max=np.array(mvs_cfg.get("hard_max", [300.0, 200.0, 80.0]), dtype=np.float64),
            percentile_low=float(mvs_cfg.get("percentile_low", 0.2)),
            percentile_high=float(mvs_cfg.get("percentile_high", 99.8)),
            mesh_proximity_m=float(mvs_cfg.get("mesh_proximity_m", 0.20)),
            voxel_size_m=float(mvs_cfg.get("voxel_size_m", 0.05)),
            sor_neighbors=int(mvs_cfg.get("sor_neighbors", 24)),
            sor_std_ratio=float(mvs_cfg.get("sor_std_ratio", 2.0)),
            ror_min_points=int(mvs_cfg.get("ror_min_points", 8)),
            ror_radius_m=float(mvs_cfg.get("ror_radius_m", 0.35)),
            export_mesh=bool(mvs_cfg.get("export_mesh", True)),
        )

    if from_stage <= 4 <= to_stage:
        aud_cfg = cfg.get("audit", {})
        base_ply = _resolve_path(base_dir, aud_cfg.get("baseline_ply"))
        out_rrd = (eval_dir / "audit_scene.rrd") if aud_cfg.get("export_rrd", False) else None
        stage4.run_audit(
            ply_path=mvs_dir / "consolidated_cloud.ply",
            out_dir=eval_dir,
            cell_size=float(aud_cfg.get("cell_size_m", 0.5)),
            min_pts=int(aud_cfg.get("min_pts_per_cell", 5)),
            plane_roi=aud_cfg.get("plane_roi"),
            baseline_ply=base_ply,
            out_rrd=out_rrd,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for end-to-end orchestrator."""
    parser = argparse.ArgumentParser(description="Run Stages 1-4 of drone-pointcloud-web3d from a YAML config.")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML pipeline configuration.")
    parser.add_argument("--from-stage", type=int, default=1, choices=[1, 2, 3, 4], help="Start stage (1..4).")
    parser.add_argument("--to-stage", type=int, default=4, choices=[1, 2, 3, 4], help="End stage (1..4).")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args(argv)
    if args.from_stage > args.to_stage:
        raise SystemExit("Error: --from-stage must be <= --to-stage.")
    run_pipeline(args.config, from_stage=args.from_stage, to_stage=args.to_stage)


if __name__ == "__main__":
    main()
