#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live subprocess log streaming, progress parsers, atomic progress.json, and log tailing."""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def count_snapshot_subdirs(snapshot_dir: Path | None) -> int:
    """Count non-empty COLMAP snapshot subdirectories."""
    if snapshot_dir is None or not snapshot_dir.exists():
        return 0
    return sum(1 for p in snapshot_dir.iterdir() if p.is_dir())


@dataclass
class StepProgressUpdate:
    """Parsed incremental update emitted by a log-line parser."""

    current: int | None = None
    total: int | None = None
    pct: float | None = None
    unit: str | None = None
    message: str | None = None


@dataclass
class CommandResult:
    """Result from a streamed subprocess invocation."""

    returncode: int
    elapsed_s: float
    timed_out: bool = False
    tail_lines: list[str] = field(default_factory=list)
    log_file: Path | None = None
    native_log_file: Path | None = None


class ProgressTracker:
    """Thread-safe atomic writer for pipeline progress.json files."""

    def __init__(
        self,
        progress_file: Path | None,
        stall_warn_s: float = 180.0,
        min_write_interval_s: float = 1.0,
    ) -> None:
        self.progress_file = progress_file
        self.stall_warn_s = float(stall_warn_s)
        self.min_write_interval_s = float(min_write_interval_s)
        self._last_write_ts = 0.0
        self._pipeline_t0 = time.monotonic()
        self._step_t0 = time.monotonic()
        self._snapshot_dir: Path | None = None
        self._lock = threading.Lock()
        self._state: dict = {
            "schema_version": 1,
            "status": "running",
            "pid": os.getpid(),
            "runner_pid": os.getpid(),
            "stage": "",
            "step": "",
            "current": None,
            "total": None,
            "pct": 0.0,
            "unit": "phase",
            "elapsed_s": 0.0,
            "step_elapsed_s": 0.0,
            "eta_s": None,
            "last_message": "Initializing",
            "log_file": None,
            "snapshot_dir": None,
            "snapshot_count": 0,
            "stall_warn_s": self.stall_warn_s,
            "started_at": utc_now_iso(),
            "step_started_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "steps": {},
        }

    def start_step(
        self,
        stage: str,
        step: str,
        *,
        unit: str = "phase",
        total: int | None = None,
        pid: int | None = None,
        log_file: Path | None = None,
        snapshot_dir: Path | None = None,
        message: str | None = None,
    ) -> None:
        """Mark a pipeline step as running and flush progress.json immediately."""
        with self._lock:
            now = time.monotonic()
            self._step_t0 = now
            self._snapshot_dir = snapshot_dir
            self._state["status"] = "running"
            self._state["stage"] = stage
            self._state["step"] = step
            self._state["current"] = 0 if total else None
            self._state["total"] = total
            self._state["pct"] = 0.0
            self._state["unit"] = unit
            self._state["eta_s"] = None
            self._state["pid"] = pid if pid is not None else os.getpid()
            self._state["log_file"] = str(log_file) if log_file else None
            self._state["snapshot_dir"] = str(snapshot_dir) if snapshot_dir else None
            self._state["snapshot_count"] = count_snapshot_subdirs(snapshot_dir)
            self._state["step_started_at"] = utc_now_iso()
            if message:
                self._state["last_message"] = message[:220]
            self._state["steps"][step] = {
                "stage": stage,
                "status": "running",
                "started_at": self._state["step_started_at"],
            }
            self._flush(force=True)

    def update(
        self,
        *,
        current: int | None = None,
        total: int | None = None,
        pct: float | None = None,
        unit: str | None = None,
        message: str | None = None,
        pid: int | None = None,
        force: bool = False,
    ) -> None:
        """Apply a progress delta and atomically persist when due."""
        with self._lock:
            now = time.monotonic()
            if pid is not None:
                self._state["pid"] = pid
            if unit is not None:
                self._state["unit"] = unit
            if total is not None and total > 0:
                self._state["total"] = int(total)
            if current is not None:
                self._state["current"] = int(current)

            eff_cur = self._state.get("current")
            eff_tot = self._state.get("total")
            if pct is not None:
                self._state["pct"] = round(max(0.0, min(100.0, float(pct))), 1)
            elif eff_cur is not None and eff_tot is not None and eff_tot > 0:
                self._state["pct"] = round(max(0.0, min(100.0, 100.0 * eff_cur / eff_tot)), 1)

            step_elapsed = max(0.0, now - self._step_t0)
            self._state["step_elapsed_s"] = round(step_elapsed, 1)
            self._state["elapsed_s"] = round(max(0.0, now - self._pipeline_t0), 1)

            pct_val = self._state.get("pct")
            if eff_cur is not None and eff_tot is not None and eff_cur > 0 and eff_tot > eff_cur and step_elapsed >= 1.0:
                self._state["eta_s"] = round(step_elapsed * (eff_tot - eff_cur) / eff_cur, 1)
            elif pct_val is not None and 2.0 <= pct_val < 99.5 and step_elapsed >= 2.0:
                self._state["eta_s"] = round(step_elapsed * (100.0 - pct_val) / pct_val, 1)
            elif pct_val is not None and pct_val >= 100.0:
                self._state["eta_s"] = 0.0

            if message:
                cleaned = " ".join(str(message).strip().split())
                if cleaned:
                    self._state["last_message"] = cleaned[:220]

            self._state["snapshot_count"] = count_snapshot_subdirs(self._snapshot_dir)
            self._flush(force=force)

    def finish_step(
        self,
        step: str,
        *,
        status: str = "completed",
        elapsed_s: float | None = None,
        message: str | None = None,
    ) -> None:
        """Record completion/skip/failure of a step and flush."""
        with self._lock:
            now = time.monotonic()
            dur = round(elapsed_s if elapsed_s is not None else max(0.0, now - self._step_t0), 1)
            entry = self._state["steps"].setdefault(step, {})
            entry.update(
                {
                    "status": status,
                    "elapsed_s": dur,
                    "finished_at": utc_now_iso(),
                }
            )
            if self._state.get("step") == step:
                if status == "completed":
                    self._state["pct"] = 100.0
                    self._state["eta_s"] = 0.0
                self._state["step_elapsed_s"] = dur
                self._state["pid"] = os.getpid()
            if message:
                self._state["last_message"] = " ".join(message.strip().split())[:220]
            self._state["snapshot_count"] = count_snapshot_subdirs(self._snapshot_dir)
            self._flush(force=True)

    def finish_pipeline(self, *, status: str = "completed", message: str | None = None) -> None:
        """Mark the overall pipeline run as completed or failed."""
        with self._lock:
            self._state["status"] = status
            self._state["pid"] = os.getpid()
            self._state["elapsed_s"] = round(max(0.0, time.monotonic() - self._pipeline_t0), 1)
            if status == "completed":
                self._state["pct"] = 100.0
                self._state["eta_s"] = 0.0
            if message:
                self._state["last_message"] = " ".join(message.strip().split())[:220]
            self._flush(force=True)

    def _flush(self, force: bool = False) -> None:
        if self.progress_file is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_write_ts) < self.min_write_interval_s:
            return
        self._state["updated_at"] = utc_now_iso()
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.progress_file.with_name(f"{self.progress_file.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        os.replace(tmp, self.progress_file)
        self._last_write_ts = now


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Terminate a subprocess and all child processes cleanly across Windows & POSIX."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _drain_stream_to_queue(stream, out_q: queue.Queue[str | None]) -> None:
    try:
        for line in iter(stream.readline, ""):
            out_q.put(line)
    finally:
        try:
            stream.close()
        except Exception:
            pass
        out_q.put(None)


def make_colmap_parser(
    step_name: str,
    total_images: int | None = None,
) -> Callable[[str], StepProgressUpdate | None]:
    """Create a stateful line parser for COLMAP 3.x / 4.x stderr logs."""
    re_feat = re.compile(r"Processed file \[(\d+)/(\d+)\]")
    re_match = re.compile(r"Matching block \[(\d+)/(\d+),\s*(\d+)/(\d+)\]")
    re_reg = re.compile(r"Registering image #\d+\s*\((\d+)\)")
    re_undist = re.compile(r"Undistorting image \[(\d+)/(\d+)\]")
    re_ba = re.compile(r"(Global bundle adjustment|Local bundle adjustment|Retriangulation)", re.IGNORECASE)
    re_gm_phases = [
        (re.compile(r"Reading|Loading|Initializing", re.I), 10.0),
        (re.compile(r"Rotation averaging|Global rotation", re.I), 35.0),
        (re.compile(r"Global positioning|Relative translation", re.I), 58.0),
        (re.compile(r"Bundle adjustment", re.I), 78.0),
        (re.compile(r"Triangulat|Retriangulat", re.I), 88.0),
        (re.compile(r"Writing|Saving|Reconstruction", re.I), 95.0),
    ]
    state = {"max_reg": 0, "gm_pct": 0.0}

    def parse(line: str) -> StepProgressUpdate | None:
        m = re_feat.search(line)
        if m:
            return StepProgressUpdate(
                current=int(m.group(1)),
                total=int(m.group(2)),
                unit="images",
                message=line,
            )

        m = re_match.search(line)
        if m:
            b1, n1, b2, n2 = map(int, m.groups())
            idx = max(0, (b1 - 1) * n2 + b2)
            tot = max(1, n1 * n2)
            return StepProgressUpdate(current=idx, total=tot, unit="blocks", message=line)

        m = re_reg.search(line)
        if m:
            k = int(m.group(1))
            if k > state["max_reg"]:
                state["max_reg"] = k
            return StepProgressUpdate(
                current=state["max_reg"],
                total=total_images,
                unit="images",
                message=f"Registered {state['max_reg']}/{total_images or '?'} images",
            )

        m = re_undist.search(line)
        if m:
            return StepProgressUpdate(
                current=int(m.group(1)),
                total=int(m.group(2)),
                unit="images",
                message=line,
            )

        m = re_ba.search(line)
        if m:
            return StepProgressUpdate(message=f"{m.group(1)} ({state['max_reg'] or '?'} imgs)")

        if "global_mapper" in step_name:
            for pat, pval in re_gm_phases:
                if pat.search(line) and pval > state["gm_pct"]:
                    state["gm_pct"] = pval
                    return StepProgressUpdate(pct=pval, unit="phase", message=line)

        if any(tok in line for tok in ("ERROR", "FATAL", "Checking", "Elapsed time", "=>")):
            return StepProgressUpdate(message=line)
        return None

    return parse


def make_openmvs_parser(base_name: str) -> Callable[[str], StepProgressUpdate | None]:
    """Create a stateful line parser for OpenMVS v2.x log files."""
    re_cal = re.compile(r"(\d+)\s+images\s+\((\d+)\s+calibrated\)")
    re_dmap = re.compile(r"Depth-map for image\s+\d+\s+estimated")
    re_fused = re.compile(r"Depth-maps dense fused and filtered")
    re_pc_saved = re.compile(r"Point-cloud\s+'.*'\s+saved")
    re_pc_loaded = re.compile(r"Point-cloud\s+'.*'\s+loaded")
    re_tet = re.compile(r"Delaunay tetrahedralization completed")
    re_wt = re.compile(r"Delaunay tetrahedras weighting completed")
    re_cut = re.compile(r"Delaunay tetrahedras graph-cut completed")
    re_clean = re.compile(r"Cleaned mesh:")
    re_mesh_saved = re.compile(r"Mesh\s+'.*'\s+saved")

    state = {"total_cal": None, "dmap_done": 0}

    def parse(line: str) -> StepProgressUpdate | None:
        m = re_cal.search(line)
        if m:
            state["total_cal"] = int(m.group(2))
            return StepProgressUpdate(
                current=0,
                total=state["total_cal"],
                pct=2.0,
                unit="depth_maps" if "Densify" in base_name else "phase",
                message=f"Loaded {state['total_cal']} calibrated views",
            )

        if re_dmap.search(line):
            state["dmap_done"] += 1
            tot = state["total_cal"]
            pct = round(min(92.0, 92.0 * state["dmap_done"] / max(1, tot)), 1) if tot else None
            return StepProgressUpdate(
                current=state["dmap_done"],
                total=tot,
                pct=pct,
                unit="depth_maps",
                message=f"Depth-map {state['dmap_done']}/{tot or '?'} estimated",
            )

        if re_fused.search(line):
            return StepProgressUpdate(pct=96.0, message=line)
        if re_pc_saved.search(line):
            return StepProgressUpdate(pct=100.0, message=line)
        if re_pc_loaded.search(line):
            return StepProgressUpdate(pct=10.0, message=line)
        if re_tet.search(line):
            return StepProgressUpdate(pct=35.0, message=line)
        if re_wt.search(line):
            return StepProgressUpdate(pct=60.0, message=line)
        if re_cut.search(line):
            return StepProgressUpdate(pct=85.0, message=line)
        if re_clean.search(line):
            return StepProgressUpdate(pct=95.0, message=line)
        if re_mesh_saved.search(line):
            return StepProgressUpdate(pct=100.0, message=line)
        return None

    return parse


def stream_command(
    cmd: list[str],
    *,
    stage: str,
    step_name: str,
    tracker: ProgressTracker | None = None,
    log_file: Path | None = None,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    line_parser: Callable[[str], StepProgressUpdate | None] | None = None,
    total_units: int | None = None,
    unit: str = "phase",
    snapshot_dir: Path | None = None,
    timeout_s: float | None = None,
    echo_milestones: bool = True,
    check: bool = True,
) -> CommandResult:
    """Run a command with real-time unbuffered stdout/stderr teeing and progress tracking."""
    t0 = time.monotonic()
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_file, "w", encoding="utf-8", errors="replace", buffering=1) if log_file else None

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if tracker is not None:
        tracker.start_step(
            stage,
            step_name,
            unit=unit,
            total=total_units,
            pid=proc.pid,
            log_file=log_file,
            snapshot_dir=snapshot_dir,
            message=f"Started {step_name}",
        )

    out_q: queue.Queue[str | None] = queue.Queue()
    reader = threading.Thread(target=_drain_stream_to_queue, args=(proc.stdout, out_q), daemon=True)
    reader.start()

    tail: list[str] = []
    eof = False
    timed_out = False
    last_heartbeat = t0
    last_pct_printed = -10.0

    try:
        while not (eof and proc.poll() is not None):
            if timeout_s is not None and timeout_s > 0 and (time.monotonic() - t0) > timeout_s:
                timed_out = True
                print(f"!!! [{step_name}] TIMED OUT after {time.monotonic() - t0:.1f}s — terminating tree", flush=True)
                _kill_process_tree(proc)
                break

            try:
                item = out_q.get(timeout=0.3)
            except queue.Empty:
                if tracker is not None and (time.monotonic() - last_heartbeat) >= 15.0:
                    tracker.update(pid=proc.pid, force=True)
                    last_heartbeat = time.monotonic()
                continue

            if item is None:
                eof = True
                continue

            line = item.rstrip("\r\n")
            if log_fh is not None:
                log_fh.write(line + "\n")
            tail.append(line)
            if len(tail) > 80:
                tail.pop(0)

            upd = line_parser(line) if line_parser is not None else None
            if upd is not None and tracker is not None:
                tracker.update(
                    current=upd.current,
                    total=upd.total,
                    pct=upd.pct,
                    unit=upd.unit,
                    message=upd.message,
                    pid=proc.pid,
                )
            if upd is not None and echo_milestones:
                cur_state_pct = tracker._state.get("pct") if tracker is not None else upd.pct
                is_pct_jump = cur_state_pct is not None and (cur_state_pct - last_pct_printed) >= 10.0
                if is_pct_jump or upd.current is None:
                    if cur_state_pct is not None:
                        last_pct_printed = float(cur_state_pct)
                    prefix = f"[{cur_state_pct:5.1f}%] " if cur_state_pct is not None else ""
                    print(f"  {step_name} {prefix}{upd.message or line}", flush=True)
    finally:
        reader.join(timeout=2.0)
        rc = proc.wait() if proc.poll() is None else proc.returncode
        if log_fh is not None:
            log_fh.close()

    dt = time.monotonic() - t0
    if tracker is not None:
        if timed_out or rc != 0:
            tracker.finish_step(step_name, status="failed", elapsed_s=dt, message=f"{step_name} exited rc={rc}")
        else:
            tracker.finish_step(step_name, status="completed", elapsed_s=dt, message=f"{step_name} done ({dt:.1f}s)")

    if (timed_out or rc != 0) and check:
        tail_txt = "\n".join(tail[-35:])
        raise RuntimeError(f"{step_name} failed (rc={rc}, timed_out={timed_out}, {dt:.1f}s):\n{tail_txt}")

    return CommandResult(returncode=rc, elapsed_s=dt, timed_out=timed_out, tail_lines=tail, log_file=log_file)


def tail_openmvs_process(
    cmd: list[str],
    *,
    base_name: str,
    stage: str,
    tracker: ProgressTracker | None = None,
    cwd: Path,
    search_dirs: list[Path],
    mirror_log_file: Path | None = None,
    env: dict[str, str] | None = None,
    echo_milestones: bool = True,
    check: bool = True,
) -> CommandResult:
    """Launch OpenMVS tool and tail both stdout and its native *.log file in real time."""
    t_spawn_wall = time.time() - 1.5
    t0 = time.monotonic()
    if mirror_log_file is not None:
        mirror_log_file.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(mirror_log_file, "w", encoding="utf-8", errors="replace", buffering=1) if mirror_log_file else None

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    parser = make_openmvs_parser(base_name)
    if tracker is not None:
        tracker.start_step(
            stage,
            base_name,
            unit="depth_maps" if "Densify" in base_name else "phase",
            pid=proc.pid,
            log_file=mirror_log_file,
            message=f"Started {base_name}",
        )

    out_q: queue.Queue[str | None] = queue.Queue()
    reader = threading.Thread(target=_drain_stream_to_queue, args=(proc.stdout, out_q), daemon=True)
    reader.start()

    active_native_log: Path | None = None
    native_offset = 0
    partial_line = ""
    tail: list[str] = []
    last_heartbeat = t0
    last_pct_printed = -10.0

    def handle_line(raw: str) -> None:
        nonlocal last_pct_printed
        line = raw.rstrip("\r\n")
        if not line:
            return
        if log_fh is not None:
            log_fh.write(line + "\n")
        tail.append(line)
        if len(tail) > 80:
            tail.pop(0)
        upd = parser(line)
        if upd is not None and tracker is not None:
            tracker.update(
                current=upd.current,
                total=upd.total,
                pct=upd.pct,
                unit=upd.unit,
                message=upd.message,
                pid=proc.pid,
            )
        if upd is not None and echo_milestones:
            cur_pct = tracker._state.get("pct") if tracker is not None else upd.pct
            if cur_pct is None or (cur_pct - last_pct_printed) >= 10.0 or cur_pct >= 96.0:
                if cur_pct is not None:
                    last_pct_printed = float(cur_pct)
                prefix = f"[{cur_pct:5.1f}%] " if cur_pct is not None else ""
                print(f"  {base_name} {prefix}{upd.message or line}", flush=True)

    def poll_native_log() -> None:
        nonlocal active_native_log, native_offset, partial_line
        if active_native_log is None:
            cands: list[Path] = []
            for d in search_dirs:
                if d.exists():
                    for p in d.glob(f"{base_name}*.log"):
                        if p.stat().st_mtime >= t_spawn_wall:
                            cands.append(p)
            if cands:
                cands.sort(key=lambda p: p.stat().st_mtime)
                active_native_log = cands[-1]
                if tracker is not None and mirror_log_file is None:
                    tracker.update(force=False)
        if active_native_log is not None and active_native_log.exists():
            size = active_native_log.stat().st_size
            if size > native_offset:
                with open(active_native_log, "rb") as nf:
                    nf.seek(native_offset)
                    chunk = nf.read(size - native_offset)
                native_offset = size
                text_chunk = partial_line + chunk.decode("utf-8", errors="replace")
                lines = text_chunk.splitlines(keepends=True)
                partial_line = ""
                for ln in lines:
                    if ln.endswith(("\n", "\r")):
                        handle_line(ln)
                    else:
                        partial_line = ln

    try:
        while proc.poll() is None:
            while not out_q.empty():
                item = out_q.get_nowait()
                if item is not None:
                    handle_line(item)
            poll_native_log()
            if tracker is not None and (time.monotonic() - last_heartbeat) >= 15.0:
                tracker.update(pid=proc.pid, force=True)
                last_heartbeat = time.monotonic()
            time.sleep(0.4)

        # Final flush after exit
        time.sleep(0.15)
        while not out_q.empty():
            item = out_q.get_nowait()
            if item is not None:
                handle_line(item)
        poll_native_log()
        if partial_line:
            handle_line(partial_line)
    finally:
        reader.join(timeout=2.0)
        rc = proc.wait()
        if log_fh is not None:
            log_fh.close()

    dt = time.monotonic() - t0
    if tracker is not None:
        status = "completed" if rc == 0 else "failed"
        tracker.finish_step(base_name, status=status, elapsed_s=dt, message=f"{base_name} rc={rc} ({dt:.1f}s)")

    if rc != 0 and check:
        native_tail = ""
        if active_native_log is not None and active_native_log.exists():
            native_tail = active_native_log.read_text(encoding="utf-8", errors="replace")[-2500:]
        tail_txt = "\n".join(tail[-25:])
        raise RuntimeError(f"OpenMVS {base_name} failed (rc={rc}, {dt:.1f}s):\n{tail_txt}\n{native_tail}")

    return CommandResult(
        returncode=rc,
        elapsed_s=dt,
        tail_lines=tail,
        log_file=mirror_log_file,
        native_log_file=active_native_log,
    )
