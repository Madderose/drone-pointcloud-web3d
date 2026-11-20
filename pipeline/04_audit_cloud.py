#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 4: Quantitative Geometric Audit, Cross-Section Plots & Optional Rerun Export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import open3d as o3d

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cell_z_stats(pts: np.ndarray, cell_size: float = 0.5, min_pts: int = 5) -> dict:
    """Measure per-cell vertical IQR, 5-95th percentile Z-span, and outlier cell ratios."""
    if len(pts) < min_pts:
        return {
            "n_cells": 0,
            "med_iqr_m": None,
            "med_span_m": None,
            "p90_span_m": None,
            "pct_gt_05m": None,
            "pct_gt_20m": None,
        }
    ix = np.floor(pts[:, 0] / cell_size).astype(np.int64)
    iy = np.floor(pts[:, 1] / cell_size).astype(np.int64)
    keys = ix * 1_000_000 + iy
    order = np.argsort(keys)
    sk, sz = keys[order], pts[order, 2]
    _, starts, counts = np.unique(sk, return_index=True, return_counts=True)

    iqrs, spans = [], []
    for s, c in zip(starts, counts):
        if c >= min_pts:
            p5, p25, p75, p95 = np.percentile(sz[s : s + c], [5, 25, 75, 95])
            iqrs.append(p75 - p25)
            spans.append(p95 - p5)
    if not spans:
        return {
            "n_cells": 0,
            "med_iqr_m": None,
            "med_span_m": None,
            "p90_span_m": None,
            "pct_gt_05m": None,
            "pct_gt_20m": None,
        }
    iqrs_a = np.array(iqrs)
    spans_a = np.array(spans)
    return {
        "n_cells": int(len(spans_a)),
        "med_iqr_m": round(float(np.median(iqrs_a)), 4),
        "med_span_m": round(float(np.median(spans_a)), 4),
        "p90_span_m": round(float(np.percentile(spans_a, 90)), 4),
        "pct_gt_05m": round(float(np.mean(spans_a > 0.5) * 100.0), 2),
        "pct_gt_20m": round(float(np.mean(spans_a > 2.0) * 100.0), 2),
    }


def xy_footprint_stats(pts: np.ndarray, cell_size: float = 1.0) -> dict:
    """Compute XY footprint area (m^2 and hectares) from occupied grid cells."""
    if len(pts) == 0:
        return {"occupied_cells": 0, "area_m2": 0.0, "area_ha": 0.0}
    ix = np.floor(pts[:, 0] / cell_size).astype(np.int64)
    iy = np.floor(pts[:, 1] / cell_size).astype(np.int64)
    unique_cells = len(set(zip(ix.tolist(), iy.tolist())))
    area_m2 = float(unique_cells) * (cell_size**2)
    return {
        "occupied_cells": int(unique_cells),
        "area_m2": round(area_m2, 1),
        "area_ha": round(area_m2 / 10000.0, 3),
    }


def ransac_plane_rmse(pts: np.ndarray, roi: list[float]) -> dict:
    """Measure planar thickness RMSE (cm) inside an ENU ROI [xmin, xmax, ymin, ymax, zmin, zmax]."""
    xmin, xmax, ymin, ymax, zmin, zmax = roi
    m = (
        (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax)
        & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
        & (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
    )
    sub = pts[m]
    if len(sub) < 40:
        return {"plane_pts": int(len(sub)), "plane_rmse_cm": None, "plane_p90_err_cm": None}
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sub)
    plane, _ = pcd.segment_plane(distance_threshold=0.08, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane
    norm = float(np.linalg.norm([a, b, c]))
    dists = np.abs(sub @ np.array([a, b, c]) + d) / max(norm, 1e-9)
    return {
        "plane_pts": int(len(sub)),
        "plane_rmse_cm": round(float(np.sqrt(np.mean(dists**2)) * 100.0), 2),
        "plane_p90_err_cm": round(float(np.percentile(dists, 90) * 100.0), 2),
    }


def render_audit_figure(
    pts: np.ndarray,
    cols: np.ndarray,
    out_png: Path,
    baseline_pts: np.ndarray | None = None,
    metrics: dict | None = None,
) -> None:
    """Render a 4-panel dark-theme orthophoto, elevation DSM, cross-section, and metric summary."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=150)
    order = np.argsort(pts[:, 2])
    x0, x1 = np.percentile(pts[:, 0], [0.5, 99.5])
    y0, y1 = np.percentile(pts[:, 1], [0.5, 99.5])
    z0, z1 = np.percentile(pts[:, 2], [2.0, 98.0])
    pad_x = max(2.0, 0.05 * (x1 - x0))
    pad_y = max(2.0, 0.05 * (y1 - y0))

    # Panel 1: RGB Orthophoto
    ax = axes[0, 0]
    ax.set_facecolor("#0F172A")
    ax.scatter(pts[order, 0], pts[order, 1], c=cols[order], s=0.8, rasterized=True)
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - pad_y, y1 + pad_y)
    ax.set_aspect("equal")
    ax.set_title(f"1. RGB Orthophoto ({len(pts):,} pts)", fontweight="bold")
    ax.set_xlabel("East X (m)")
    ax.set_ylabel("North Y (m)")
    ax.grid(True, alpha=0.25, color="#94A3B8")

    # Panel 2: DSM Elevation
    ax = axes[0, 1]
    ax.set_facecolor("#0F172A")
    sc = ax.scatter(pts[order, 0], pts[order, 1], c=pts[order, 2], cmap="turbo", vmin=z0, vmax=z1, s=0.8, rasterized=True)
    plt.colorbar(sc, ax=ax, label="ENU Elevation Z (m)")
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - pad_y, y1 + pad_y)
    ax.set_aspect("equal")
    ax.set_title("2. Digital Surface Model (ENU Z)", fontweight="bold")
    ax.set_xlabel("East X (m)")
    ax.set_ylabel("North Y (m)")
    ax.grid(True, alpha=0.25, color="#94A3B8")

    # Panel 3: Central E-W Vertical Cross-Section
    y_mid = float(np.median(pts[:, 1]))
    slab = max(1.5, 0.04 * (y1 - y0))
    ax = axes[1, 0]
    ax.set_facecolor("#0F172A")
    if baseline_pts is not None and len(baseline_pts) > 0:
        mb = (baseline_pts[:, 1] >= y_mid - slab) & (baseline_pts[:, 1] <= y_mid + slab)
        ax.scatter(
            baseline_pts[mb, 0],
            baseline_pts[mb, 2],
            c="#EF4444",
            s=2.0,
            alpha=0.45,
            label="Baseline Cloud",
            rasterized=True,
        )
    mc = (pts[:, 1] >= y_mid - slab) & (pts[:, 1] <= y_mid + slab)
    ax.scatter(pts[mc, 0], pts[mc, 2], c="#22D3EE", s=2.2, alpha=0.85, label="Consolidated Cloud", rasterized=True)
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_title(f"3. E-W Vertical Cross-Section (Y = {y_mid - slab:.1f}..{y_mid + slab:.1f} m)", fontweight="bold")
    ax.set_xlabel("East X (m)")
    ax.set_ylabel("Elevation Z (m)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25, color="#94A3B8")

    # Panel 4: Quantitative Summary Card
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("4. Quantitative Geometric Audit", fontweight="bold")
    if metrics is not None:
        lines = [
            f"Points:            {metrics['points']:,}",
            f"Footprint Area:    {metrics['footprint']['area_ha']:.3f} ha ({metrics['footprint']['area_m2']:,.0f} m2)",
            f"Cell Z-IQR (med):  {metrics['cell_z']['med_iqr_m']} m",
            f"Cell Z-Span (med): {metrics['cell_z']['med_span_m']} m",
            f"Cell Z-Span (p90): {metrics['cell_z']['p90_span_m']} m",
            f"Cells > 0.5 m:     {metrics['cell_z']['pct_gt_05m']} %",
        ]
        if metrics.get("plane_rmse"):
            pr = metrics["plane_rmse"]
            lines.append(f"Plane RMSE:        {pr['plane_rmse_cm']} cm (p90={pr['plane_p90_err_cm']} cm)")
        if metrics.get("baseline_cell_z"):
            bz = metrics["baseline_cell_z"]
            lines.append(f"Baseline Z-Span:   {bz['med_span_m']} m (>{0.5}m: {bz['pct_gt_05m']}%)")
        ax.text(
            0.05,
            0.85,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=11,
            va="top",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.7", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def export_rerun(pts: np.ndarray, cols: np.ndarray, out_rrd: Path) -> None:
    """Save point cloud into an interactive `.rrd` Rerun recording (requires optional rerun-sdk)."""
    try:
        import rerun as rr
    except ImportError as exc:
        raise RuntimeError("rerun-sdk is not installed; install `rerun-sdk` to use --rrd.") from exc

    rr.init("drone_pointcloud_audit", spawn=False)
    rr.save(str(out_rrd))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/point_cloud",
        rr.Points3D(pts, colors=(np.clip(cols, 0.0, 1.0) * 255.0).astype(np.uint8), radii=0.05),
        static=True,
    )
    print(f"Saved Rerun recording -> {out_rrd}")


def run_audit(
    ply_path: Path,
    out_dir: Path,
    cell_size: float = 0.5,
    min_pts: int = 5,
    plane_roi: list[float] | None = None,
    baseline_ply: Path | None = None,
    out_rrd: Path | None = None,
) -> dict:
    """Execute Stage 4 geometric audit, plot generation, and optional Rerun export."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points, dtype=np.float64)
    cols = np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors() else np.full_like(pts, 0.7)

    report: dict = {
        "input_ply": str(ply_path),
        "points": int(len(pts)),
        "bbox_min": [round(float(x), 3) for x in pts.min(axis=0)] if len(pts) else [],
        "bbox_max": [round(float(x), 3) for x in pts.max(axis=0)] if len(pts) else [],
        "footprint": xy_footprint_stats(pts, cell_size=1.0),
        "cell_z": cell_z_stats(pts, cell_size=cell_size, min_pts=min_pts),
    }
    if plane_roi is not None:
        report["plane_rmse"] = ransac_plane_rmse(pts, plane_roi)

    base_pts = None
    if baseline_ply is not None and baseline_ply.exists():
        b_pcd = o3d.io.read_point_cloud(str(baseline_ply))
        base_pts = np.asarray(b_pcd.points, dtype=np.float64)
        report["baseline_points"] = int(len(base_pts))
        report["baseline_cell_z"] = cell_z_stats(base_pts, cell_size=cell_size, min_pts=min_pts)
        if plane_roi is not None:
            report["baseline_plane_rmse"] = ransac_plane_rmse(base_pts, plane_roi)

    metrics_path = out_dir / "audit_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    png_path = out_dir / "audit_report.png"
    render_audit_figure(pts, cols, png_path, baseline_pts=base_pts, metrics=report)

    if out_rrd is not None:
        export_rerun(pts, cols, out_rrd)

    print(json.dumps(report, indent=2))
    print(f"Saved audit report -> {metrics_path} & {png_path}")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for Stage 4."""
    parser = argparse.ArgumentParser(
        description="Audit consolidated PLY surface sharpness, cell Z-spread, planar RMSE, and cross-sections."
    )
    parser.add_argument("--ply", required=True, type=Path, help="Input consolidated PLY point cloud.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for audit JSON & PNG.")
    parser.add_argument("--cell-size", type=float, default=0.5, help="Grid cell size in meters (default: 0.5).")
    parser.add_argument("--min-pts", type=int, default=5, help="Min points per cell for Z-spread (default: 5).")
    parser.add_argument(
        "--plane-roi",
        nargs=6,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Optional ENU ROI box for RANSAC planar RMSE (cm).",
    )
    parser.add_argument("--baseline-ply", type=Path, default=None, help="Optional baseline PLY for side-by-side plots.")
    parser.add_argument("--rrd", type=Path, default=None, help="Optional output .rrd Rerun recording path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args(argv)
    run_audit(
        ply_path=args.ply,
        out_dir=args.out,
        cell_size=args.cell_size,
        min_pts=args.min_pts,
        plane_roi=args.plane_roi,
        baseline_ply=args.baseline_ply,
        out_rrd=args.rrd,
    )


if __name__ == "__main__":
    main()
