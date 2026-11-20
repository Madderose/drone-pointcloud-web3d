#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Atomic FTP uploader for the WebGL2 `.pcloud` viewer (`STOR *.tmp` -> `RNFR/RNTO`).

Reads credentials strictly from environment variables or a local `.env` file:
  FTP_HOST, FTP_PORT (default 21), FTP_USER, FTP_PASS, FTP_REMOTE_DIR, PUBLIC_URL
"""

from __future__ import annotations

import argparse
import ftplib
import io
import os
import sys
import urllib.request
from pathlib import Path

STATIC_FILES = ["index.html", "styles.css", "app.js", ".htaccess"]


def load_dotenv(env_path: Path) -> None:
    """Minimal dependency-free `.env` loader (does not overwrite existing env vars)."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def ensure_remote_dir(ftp: ftplib.FTP, path: str) -> None:
    """Recursively create and verify remote FTP directory."""
    parts = [p for p in path.split("/") if p]
    curr = ""
    for part in parts:
        curr += "/" + part
        try:
            ftp.cwd(curr)
        except ftplib.error_perm:
            ftp.mkd(curr)
            ftp.cwd(curr)


def atomic_upload(ftp: ftplib.FTP, remote_dir: str, local_path: Path, remote_name: str) -> int:
    """Upload file as `<name>.tmp` then atomically rename via `RNFR/RNTO`."""
    ftp.cwd(remote_dir)
    raw = local_path.read_bytes()
    tmp_name = f"{remote_name}.tmp"
    ftp.storbinary(f"STOR {tmp_name}", io.BytesIO(raw), blocksize=65536)
    try:
        ftp.rename(tmp_name, remote_name)
    except ftplib.error_perm:
        try:
            ftp.delete(remote_name)
        except ftplib.error_perm:
            pass
        ftp.rename(tmp_name, remote_name)
    return len(raw)


def verify_http(url: str) -> tuple[int, int]:
    """Perform HTTP GET health check and return (status_code, content_length)."""
    req = urllib.request.Request(url, headers={"User-Agent": "DronePointCloudDeployCheck/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = resp.read()
        return resp.status, len(body)


def deploy(viewer_dir: Path, data_dir: Path, env_file: Path | None, skip_verify: bool = False) -> None:
    """Upload static viewer assets and `.pcloud`/JSON payloads atomically over FTP."""
    if env_file is not None:
        load_dotenv(env_file)

    host = os.environ.get("FTP_HOST", "").strip()
    port = int(os.environ.get("FTP_PORT", "21"))
    user = os.environ.get("FTP_USER", "").strip()
    password = os.environ.get("FTP_PASS", "").strip()
    remote_dir = os.environ.get("FTP_REMOTE_DIR", "").strip().rstrip("/")
    public_url = os.environ.get("PUBLIC_URL", "").strip()

    if not all([host, user, password, remote_dir]):
        raise SystemExit(
            "Error: set FTP_HOST, FTP_USER, FTP_PASS, and FTP_REMOTE_DIR via environment or `.env`."
        )
    if not data_dir.exists():
        raise SystemExit(f"Error: data directory not found: {data_dir}")

    remote_data_dir = f"{remote_dir}/data"
    print(f"Connecting FTP {host}:{port} as {user}...")
    ftp = ftplib.FTP()
    ftp.connect(host, port, timeout=30)
    ftp.login(user, password)
    ftp.set_pasv(True)

    ensure_remote_dir(ftp, remote_dir)
    for fname in STATIC_FILES:
        lp = viewer_dir / fname
        if not lp.exists():
            raise FileNotFoundError(f"Missing static viewer file: {lp}")
        sz = atomic_upload(ftp, remote_dir, lp, fname)
        print(f"  [OK] {fname} -> {remote_dir}/{fname} ({sz:,} bytes)")

    ensure_remote_dir(ftp, remote_data_dir)
    data_files = sorted(p for p in data_dir.iterdir() if p.is_file() and not p.name.endswith(".tmp"))
    for lp in data_files:
        sz = atomic_upload(ftp, remote_data_dir, lp, lp.name)
        print(f"  [OK] data/{lp.name} -> {remote_data_dir}/{lp.name} ({sz:,} bytes)")

    ftp.quit()

    if public_url and not skip_verify:
        base = public_url if public_url.endswith("/") else f"{public_url}/"
        print("\nVerifying HTTP endpoints...")
        for url in (base, f"{base}data/manifest.json"):
            status, length = verify_http(url)
            print(f"  HTTP {status} — {url} ({length:,} bytes)")
            if status != 200 or length == 0:
                raise SystemExit(f"HTTP verification failed for {url}")

    print("Deployment completed successfully.")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for FTP deployment."""
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Atomic FTP uploader for the drone-pointcloud-web3d viewer.")
    parser.add_argument("--viewer-dir", type=Path, default=repo_root / "viewer", help="Path to viewer/ static files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "viewer" / "data",
        help="Path to exported .pcloud & manifest.json data directory.",
    )
    parser.add_argument("--env-file", type=Path, default=repo_root / ".env", help="Optional .env file path.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip post-upload HTTP GET check.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args(argv)
    deploy(args.viewer_dir, args.data_dir, args.env_file, skip_verify=args.skip_verify)


if __name__ == "__main__":
    main()
