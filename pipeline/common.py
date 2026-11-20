# -*- coding: utf-8 -*-
"""Shared telemetry, projection, sky-masking, and Sim(3) alignment utilities."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

R_EARTH_M = 6378137.0

# Matches DJI SRT subtitle blocks across DJI Mini / Air / Mavic series
_SRT_BLOCK_RE = re.compile(
    r"(?:FrameCnt:\s*(\d+).*?)?"
    r"\[latitude:\s*([-\d.]+)\]\s*"
    r"\[longitude:\s*([-\d.]+)\]\s*"
    r"(?:\[(?:rel_alt:\s*([-\d.]+)\s+)?abs_alt:\s*([-\d.]+)\]|"
    r"\[altitude:\s*([-\d.]+)\])",
    re.DOTALL | re.IGNORECASE,
)
_FOCAL_RE = re.compile(r"\[focal_len:\s*([\d.]+)\]", re.IGNORECASE)
_FRAME_CNT_RE = re.compile(r"FrameCnt:\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class EnuOrigin:
    """Local East-North-Up tangent-plane reference origin."""

    lat0: float
    lon0: float
    abs_alt_ref: float

    def to_enu(self, lat: float, lon: float, abs_alt: float) -> tuple[float, float, float]:
        """Project WGS84 (lat, lon, abs_alt) into local ENU meters (East, North, Up)."""
        x_e = math.radians(lon - self.lon0) * R_EARTH_M * math.cos(math.radians(self.lat0))
        y_n = math.radians(lat - self.lat0) * R_EARTH_M
        z_u = abs_alt - self.abs_alt_ref
        return x_e, y_n, z_u

    def as_dict(self) -> dict[str, float]:
        """Serialize origin to a JSON-friendly dictionary."""
        return {
            "lat0": float(self.lat0),
            "lon0": float(self.lon0),
            "abs_alt_ref": float(self.abs_alt_ref),
        }


def parse_dji_srt(srt_path: Path) -> list[dict]:
    """Parse DJI per-frame `.SRT` telemetry into raw WGS84 records.

    Supports DJI Mini/Air/Mavic bracketed telemetry tags (`[latitude: ...]`,
    `[longitude: ...]`, `[rel_alt: ... abs_alt: ...]` or `[altitude: ...]`).
    """
    text = srt_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\r?\n\s*\r?\n", text.strip())
    rows: list[dict] = []

    for idx, block in enumerate(blocks):
        m = _SRT_BLOCK_RE.search(block)
        if not m:
            continue
        fc_match = _FRAME_CNT_RE.search(block)
        frame_cnt = int(m.group(1) or (fc_match.group(1) if fc_match else idx + 1))
        focal_match = _FOCAL_RE.search(block)
        focal = float(focal_match.group(1)) if focal_match else 24.0
        lat = float(m.group(2))
        lon = float(m.group(3))
        rel_alt = float(m.group(4)) if m.group(4) is not None else 0.0
        abs_alt = float(m.group(5) if m.group(5) is not None else (m.group(6) or rel_alt))
        rows.append(
            {
                "frame_cnt": frame_cnt,
                "focal_35mm": focal,
                "lat": lat,
                "lon": lon,
                "rel_alt": rel_alt,
                "abs_alt": abs_alt,
            }
        )

    if not rows:
        for idx, m in enumerate(_SRT_BLOCK_RE.finditer(text)):
            frame_cnt = int(m.group(1) or (idx + 1))
            lat = float(m.group(2))
            lon = float(m.group(3))
            rel_alt = float(m.group(4)) if m.group(4) is not None else 0.0
            abs_alt = float(m.group(5) if m.group(5) is not None else (m.group(6) or rel_alt))
            rows.append(
                {
                    "frame_cnt": frame_cnt,
                    "focal_35mm": 24.0,
                    "lat": lat,
                    "lon": lon,
                    "rel_alt": rel_alt,
                    "abs_alt": abs_alt,
                }
            )
    return rows


def read_exif_gps(jpg_path: Path) -> dict:
    """Extract WGS84 GPS (lat, lon, abs_alt) from JPEG EXIF GPS IFD (0x8825)."""
    with Image.open(jpg_path) as im:
        exif = im.getexif()
        gps = exif.get_ifd(0x8825)
    if not gps or 2 not in gps or 4 not in gps:
        raise ValueError(f"No GPS EXIF IFD found in image: {jpg_path}")

    def _dms_to_deg(dms) -> float:
        return float(dms[0]) + float(dms[1]) / 60.0 + float(dms[2]) / 3600.0

    lat = _dms_to_deg(gps[2])
    if gps.get(1, "N") == "S":
        lat = -lat
    lon = _dms_to_deg(gps[4])
    if gps.get(3, "E") == "W":
        lon = -lon
    abs_alt = float(gps[6]) if 6 in gps else 0.0
    if gps.get(5, 0) == 1:
        abs_alt = -abs_alt
    return {"lat": lat, "lon": lon, "abs_alt": abs_alt}


def compute_sky_mask_and_stats(img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Build a conservative binary keep-mask (255=keep, 0=sky) and frame quality stats.

    Only top-edge-connected low-gradient bright/blue regions in the upper 45% of
    the frame are masked, with a 1-iteration 3x3 erosion at 640x360 so horizon
    treetops and roof ridges remain completely unmasked.
    """
    h, w = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (640, 360), interpolation=cv2.INTER_AREA)
    gray_s = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv_s = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    lap_var = float(cv2.Laplacian(gray_s, cv2.CV_64F).var())
    mean_v = float(np.mean(gray_s))
    std_v = float(np.std(gray_s))
    clip_pct = float(np.mean(gray_s > 245) * 100.0)

    b_s, _, r_s = cv2.split(small)
    grad_x = cv2.Sobel(gray_s, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_s, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(grad_x, grad_y)

    bright_desat = (hsv_s[:, :, 2] > 185) & (hsv_s[:, :, 1] < 95)
    blue_sky = (b_s.astype(np.int16) - r_s.astype(np.int16) > 18) & (hsv_s[:, :, 2] > 140)
    sky_cand = (bright_desat | blue_sky) & (grad_mag < 32.0)
    sky_cand[int(0.45 * 360) :, :] = False

    _, labels = cv2.connectedComponents(sky_cand.astype(np.uint8), connectivity=8)
    top_labels = set(np.unique(labels[0, :]).tolist()) - {0}
    if not top_labels:
        sky_small = np.zeros((360, 640), dtype=np.uint8)
    else:
        sky_small = np.isin(labels, list(top_labels)).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        sky_small = cv2.erode(sky_small, kernel, iterations=1)

    keep_small = ((1 - sky_small) * 255).astype(np.uint8)
    keep_full = cv2.resize(keep_small, (w, h), interpolation=cv2.INTER_NEAREST)
    sky_pct = float(np.mean(sky_small > 0) * 100.0)

    stats = {
        "sharpness_lap": round(lap_var, 2),
        "mean_luma": round(mean_v, 2),
        "std_contrast": round(std_v, 2),
        "clip_pct": round(clip_pct, 3),
        "sky_masked_pct": round(sky_pct, 2),
    }
    return keep_full, stats


def umeyama_sim3(
    src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute weighted 7-DoF Umeyama similarity transform: dst ~= s * (R @ src) + t."""
    n = len(src)
    if n < 3:
        raise ValueError(f"Umeyama Sim(3) requires at least 3 points, got {n}")
    w = np.ones(n, dtype=np.float64) / n if weights is None else np.asarray(weights, dtype=np.float64)
    w_sum = float(np.sum(w))
    w = w / max(w_sum, 1e-12)
    mu_s = np.sum(src * w[:, None], axis=0)
    mu_d = np.sum(dst * w[:, None], axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = (dc * w[:, None]).T @ sc
    u_mat, d_vec, vt_mat = np.linalg.svd(cov)
    s_mat = np.eye(3)
    if np.linalg.det(u_mat) * np.linalg.det(vt_mat) < 0:
        s_mat[2, 2] = -1.0
    r_mat = u_mat @ s_mat @ vt_mat
    var_s = float(np.sum(w * np.sum(sc**2, axis=1)))
    scale = float(np.sum(d_vec * np.diag(s_mat)) / max(var_s, 1e-12))
    t_vec = mu_d - scale * (r_mat @ mu_s)
    return scale, r_mat, t_vec


def ransac_irls_sim3(
    src: np.ndarray,
    dst: np.ndarray,
    inlier_thresh: float = 2.0,
    huber_delta: float = 1.0,
    num_iters: int = 3500,
    seed: int = 42,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate robust 7-DoF rigid Sim(3) via RANSAC followed by Huber IRLS."""
    rng = np.random.default_rng(seed)
    n = len(src)
    if n < 3:
        raise ValueError(f"RANSAC Sim(3) requires at least 3 correspondences, got {n}")

    best_inl = np.zeros(n, dtype=bool)
    for thresh in (inlier_thresh, inlier_thresh * 2.5):
        for _ in range(num_iters):
            idx = rng.choice(n, size=3, replace=False)
            s, r_mat, t_vec = umeyama_sim3(src[idx], dst[idx])
            if not (0.005 < s < 2000.0):
                continue
            err = np.linalg.norm(s * (src @ r_mat.T) + t_vec - dst, axis=1)
            inl = err < thresh
            if int(np.sum(inl)) > int(np.sum(best_inl)):
                best_inl = inl
        if int(np.sum(best_inl)) >= max(10, int(0.45 * n)):
            break

    if int(np.sum(best_inl)) < 4:
        best_inl = np.ones(n, dtype=bool)

    s, r_mat, t_vec = umeyama_sim3(src[best_inl], dst[best_inl])
    src_in, dst_in = src[best_inl], dst[best_inl]
    for _ in range(10):
        res = np.linalg.norm(s * (src_in @ r_mat.T) + t_vec - dst_in, axis=1)
        w = np.where(res <= huber_delta, 1.0, huber_delta / np.maximum(res, 1e-6))
        s, r_mat, t_vec = umeyama_sim3(src_in, dst_in, weights=w)

    return s, r_mat, t_vec, best_inl
