#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live pipeline progress watcher & smart stall detector (progress.json + legacy log/process mode)."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.progress import make_openmvs_parser

DEFAULT_PROCESS_REGEX = r"colmap|DensifyPointCloud|ReconstructMesh|InterfaceCOLMAP"


def _parse_iso_age_s(iso_ts: str | None) -> float | None:
    if not iso_ts:
        return None
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except Exception:
        return None


def _fmt_duration(sec: float | None) -> str:
    if sec is None:
        return "--"
    s = int(max(0, round(sec)))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _progress_bar(pct: float | None, width: int = 24) -> str:
    if pct is None:
        return "[" + ("." * width) + "]   --%"
    clamped = max(0.0, min(100.0, float(pct)))
    fill = int(round(width * clamped / 100.0))
    return "[" + ("#" * fill) + ("-" * (width - fill)) + f"] {clamped:5.1f}%"


def _query_processes_windows(pid: int | None = None, name_regex: str | None = None) -> list[dict]:
    if pid is not None:
        ps = (
            f"$p = Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue; "
            "if ($p) { @($p) | Select-Object Id,ProcessName,@{N='CPU_s';E={[math]::Round($_.CPU,2)}},"
            "@{N='MemMB';E={[math]::Round($_.WorkingSet64/1MB,1)}},@{N='Threads';E={$_.Threads.Count}} "
            "| ConvertTo-Json -Compress }"
        )
    else:
        safe_pat = (name_regex or DEFAULT_PROCESS_REGEX).replace("'", "")
        ps = (
            f"$p = Get-Process -ErrorAction SilentlyContinue | Where-Object {{ $_.ProcessName -match '{safe_pat}' }}; "
            "if ($p) { @($p) | Select-Object Id,ProcessName,@{N='CPU_s';E={[math]::Round($_.CPU,2)}},"
            "@{N='MemMB';E={[math]::Round($_.WorkingSet64/1MB,1)}},@{N='Threads';E={$_.Threads.Count}} "
            "| ConvertTo-Json -Compress }"
        )
    res = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=False,
    )
    txt = (res.stdout or "").strip()
    if not txt:
        return []
    try:
        parsed = json.loads(txt)
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        return []


def _query_processes_posix(pid: int | None = None, name_regex: str | None = None) -> list[dict]:
    clk = os.sysconf(os.sysconf_names["SC_CLK_TCK"]) if hasattr(os, "sysconf") else 100
    pat = re.compile(name_regex or DEFAULT_PROCESS_REGEX, re.I)
    out: list[dict] = []
    proc_root = Path("/proc")
    if not proc_root.exists():
        return out

    pids = [str(pid)] if pid is not None else [p.name for p in proc_root.iterdir() if p.name.isdigit()]
    for pid_str in pids:
        pdir = proc_root / pid_str
        try:
            comm = (pdir / "comm").read_text(encoding="utf-8", errors="replace").strip()
            if pid is None and not pat.search(comm):
                continue
            stat_txt = (pdir / "stat").read_text(encoding="utf-8", errors="replace")
            rparen = stat_txt.rfind(")")
            fields = stat_txt[rparen + 2 :].split()
            utime, stime = int(fields[11]), int(fields[12])
            threads = int(fields[17])
            cpu_s = round((utime + stime) / max(1, clk), 2)
            status_txt = (pdir / "status").read_text(encoding="utf-8", errors="replace")
            m_rss = re.search(r"VmRSS:\s+(\d+)\s+kB", status_txt)
            mem_mb = round(int(m_rss.group(1)) / 1024.0, 1) if m_rss else 0.0
            out.append(
                {"Id": int(pid_str), "ProcessName": comm, "CPU_s": cpu_s, "MemMB": mem_mb, "Threads": threads}
            )
        except Exception:
            continue
    return out


def inspect_processes(pid: int | None = None, name_regex: str | None = None) -> list[dict]:
    """Inspect active OS process CPU seconds, RAM, and thread counts cross-platform."""
    if os.name == "nt":
        return _query_processes_windows(pid=pid, name_regex=name_regex)
    return _query_processes_posix(pid=pid, name_regex=name_regex)


def scan_legacy_directory(scan_dir: Path, process_regex: str) -> dict:
    """Inspect OpenMVS logs, COLMAP model dirs/snapshots, and OS processes without progress.json."""
    mvs_logs: list[Path] = []
    for base in (scan_dir, scan_dir / "undistorted", scan_dir / "sfm" / "undistorted", scan_dir / "mvs"):
        if base.exists():
            mvs_logs.extend(base.glob("DensifyPointCloud-*.log"))
            mvs_logs.extend(base.glob("ReconstructMesh-*.log"))
    mvs_logs.sort(key=lambda p: p.stat().st_mtime)

    mvs_status = None
    if mvs_logs:
        newest = mvs_logs[-1]
        parser = make_openmvs_parser(newest.stem)
        last_upd = None
        for ln in newest.read_text(encoding="utf-8", errors="replace").splitlines():
            upd = parser(ln)
            if upd is not None:
                last_upd = upd
        mvs_status = {
            "log_file": str(newest),
            "age_s": round(max(0.0, time.time() - newest.stat().st_mtime), 1),
            "current": last_upd.current if last_upd else None,
            "total": last_upd.total if last_upd else None,
            "pct": last_upd.pct if last_upd else None,
            "message": last_upd.message if last_upd else "Initialized",
        }

    models: list[str] = []
    for pat in ("sparse_global/*", "sparse_inc/*", "sparse_ba", "snapshots/*", "sfm/sparse_global/*", "sfm/sparse_inc/*", "sfm/snapshots/*"):
        for d in sorted(scan_dir.glob(pat)):
            if d.is_dir() and (d / "images.bin").exists():
                models.append(str(d.relative_to(scan_dir)))

    procs = inspect_processes(name_regex=process_regex)
    return {
        "scan_dir": str(scan_dir),
        "mvs_status": mvs_status,
        "colmap_models": models,
        "processes": procs,
    }


def compute_verdict(
    state: dict | None,
    age_s: float | None,
    proc: dict | None,
    cpu_delta_s: float | None,
    stall_seconds: float,
) -> str:
    """Distinguish healthy quiet Ceres BA/MVS waves from genuinely hung processes."""
    if state is None:
        return "WAITING_FOR_PROGRESS_FILE"
    st = str(state.get("status", "unknown")).lower()
    if st == "completed":
        return "COMPLETED"
    if st == "failed":
        return "FAILED"

    if proc is None:
        return "DEAD_PID"
    if cpu_delta_s is not None and cpu_delta_s >= 0.5:
        if age_s is not None and age_s > 20.0:
            return "ACTIVE_COMPUTING"
        return "HEALTHY"
    if age_s is not None and age_s > stall_seconds:
        return "STALL_WARNING"
    return "HEALTHY"


def _format_human_status(payload: dict) -> str:
    lines: list[str] = []
    state = payload.get("progress")
    verdict = payload["verdict"]
    proc = payload.get("process")
    cpu_delta = payload.get("cpu_delta_s")

    if state:
        step = state.get("step") or "--"
        stage = state.get("stage") or "--"
        cur = state.get("current")
        tot = state.get("total")
        pct = state.get("pct")
        unit = state.get("unit") or ""
        cnt_txt = f"{cur}/{tot} {unit}".strip() if (cur is not None and tot is not None) else unit
        snaps = state.get("snapshot_count", 0)
        lines.append(
            f"[{verdict}] {stage} :: {step} | {_progress_bar(pct)} ({cnt_txt}) | "
            f"Elapsed {_fmt_duration(state.get('elapsed_s'))} | ETA {_fmt_duration(state.get('eta_s'))} | Snaps {snaps}"
        )
        if state.get("last_message"):
            lines.append(f"  Last: {state['last_message']}")
    else:
        lines.append(f"[{verdict}] No progress.json found yet")

    if proc:
        d_txt = f" | dCPU +{cpu_delta:.1f}s" if cpu_delta is not None else ""
        lines.append(
            f"  Proc: PID {proc['Id']} ({proc['ProcessName']}) | CPU {proc['CPU_s']}s{d_txt} | "
            f"RAM {proc['MemMB']} MB | Threads {proc['Threads']}"
        )
    elif state and state.get("status") == "running":
        lines.append(f"  Proc: PID {state.get('pid')} not found in process table")

    legacy = payload.get("legacy")
    if legacy:
        if legacy.get("mvs_status"):
            ms = legacy["mvs_status"]
            lines.append(
                f"  MVS log: {_progress_bar(ms.get('pct'))} ({ms.get('current')}/{ms.get('total')}) — {ms.get('message')}"
            )
        if legacy.get("colmap_models"):
            lines.append(f"  COLMAP models on disk: {', '.join(legacy['colmap_models'])}")
        for p in legacy.get("processes", []):
            lines.append(
                f"  Active proc: PID {p['Id']} ({p['ProcessName']}) CPU={p['CPU_s']}s RAM={p['MemMB']}MB Threads={p['Threads']}"
            )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for watch_progress."""
    parser = argparse.ArgumentParser(
        description="Monitor live COLMAP + OpenMVS progress.json or inspect ad-hoc logs/processes without stalls."
    )
    parser.add_argument("--workdir", type=Path, default=None, help="Pipeline working directory containing progress.json.")
    parser.add_argument("--progress-file", type=Path, default=None, help="Explicit path to progress.json.")
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=None,
        help="Ad-hoc mode: scan a directory for OpenMVS logs, COLMAP models, and processes without progress.json.",
    )
    parser.add_argument(
        "--process-name",
        default=DEFAULT_PROCESS_REGEX,
        help=f"Regex matching reconstruction processes in --scan-dir mode (default: {DEFAULT_PROCESS_REGEX}).",
    )
    parser.add_argument("--interval", type=float, default=3.0, help="Polling interval in seconds (default: 3.0).")
    parser.add_argument(
        "--stall-seconds",
        type=float,
        default=None,
        help="Seconds of stale progress + zero CPU activity before raising STALL_WARNING (default: 180).",
    )
    parser.add_argument("--cpu-sample", type=float, default=0.0, help="Optional seconds to measure live CPU delta in --once mode.")
    parser.add_argument("--once", action="store_true", help="Print a single snapshot and exit immediately.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint; returns 0 cleanly or 2 on DEAD_PID / STALL_WARNING / FAILED in --once mode."""
    args = build_arg_parser().parse_args(argv)
    if args.workdir is None and args.progress_file is None and args.scan_dir is None:
        args.workdir = Path("./work")

    progress_file = args.progress_file or (args.workdir / "progress.json" if args.workdir else None)
    prev_cpu: float | None = None
    prev_pid: int | None = None

    while True:
        state = None
        age_s = None
        proc = None
        cpu_delta = None

        if progress_file is not None and progress_file.exists():
            try:
                state = json.loads(progress_file.read_text(encoding="utf-8"))
                age_s = _parse_iso_age_s(state.get("updated_at"))
            except Exception:
                state = None

        stall_s = float(args.stall_seconds or (state.get("stall_warn_s") if state else None) or 180.0)

        if state is not None and state.get("pid") and state.get("status") == "running":
            pid = int(state["pid"])
            plist = inspect_processes(pid=pid)
            proc = plist[0] if plist else None
            if args.once and args.cpu_sample > 0 and proc is not None:
                c0 = float(proc["CPU_s"])
                time.sleep(args.cpu_sample)
                plist2 = inspect_processes(pid=pid)
                proc = plist2[0] if plist2 else None
                if proc is not None:
                    cpu_delta = round(float(proc["CPU_s"]) - c0, 2)
            elif proc is not None and prev_pid == pid and prev_cpu is not None:
                cpu_delta = round(float(proc["CPU_s"]) - prev_cpu, 2)
            if proc is not None:
                prev_pid = pid
                prev_cpu = float(proc["CPU_s"])

        legacy = None
        if args.scan_dir is not None or (progress_file is not None and not progress_file.exists()):
            sdir = args.scan_dir or args.workdir
            if sdir is not None and sdir.exists():
                legacy = scan_legacy_directory(sdir, args.process_name)

        verdict = compute_verdict(state, age_s, proc, cpu_delta, stall_s)
        if state is None and legacy and legacy.get("processes"):
            verdict = "LEGACY_ACTIVE"

        payload = {
            "verdict": verdict,
            "age_s": round(age_s, 1) if age_s is not None else None,
            "cpu_delta_s": cpu_delta,
            "stall_seconds": stall_s,
            "progress": state,
            "process": proc,
            "legacy": legacy,
        }

        if args.json:
            print(json.dumps(payload, indent=2), flush=True)
        else:
            print(_format_human_status(payload), flush=True)

        if args.once:
            if verdict in ("FAILED", "DEAD_PID", "STALL_WARNING"):
                return 2
            return 0

        if verdict in ("COMPLETED", "FAILED"):
            return 0 if verdict == "COMPLETED" else 2

        time.sleep(max(0.5, args.interval))


if __name__ == "__main__":
    sys.exit(main())
