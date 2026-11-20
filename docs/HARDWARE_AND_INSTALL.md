# Hardware, Installation & Real-World Pitfalls

## 1. Recommended Hardware

- **GPU:** NVIDIA CUDA-capable GPU with >= 8 GB VRAM (tested on RTX 30/40/50 series) for COLMAP GPU SIFT & OpenMVS. CPU-only mode works with `--no-gpu` in Stage 2, though matching and densification are slower.
- **CPU & RAM:** 8+ cores, 16–32 GB RAM recommended for 200–400 image datasets at `2048x1152`.
- **Disk:** ~5–15 GB temporary workspace per project (`undistorted/` images, depth maps during `DensifyPointCloud`, and intermediate `.mvs`/`.ply` files).

---

## 2. Software Installation

### Python Environment

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
# Optional for Stage 4 .rrd export:
pip install rerun-sdk
```

### COLMAP

- **Windows:** Download the official COLMAP CUDA release archive (`COLMAP-*-windows-cuda.zip`) and point `--colmap-bin` (or `sfm.colmap_bin` in YAML) to `COLMAP.bat`.
- **Linux:** Install via `apt`, `conda`, or build from source with CUDA enabled (`colmap` on `PATH`).

### OpenMVS (External CLI — GNU AGPL-3.0)

- **Windows:** Download or build OpenMVS (`vc17/x64/Release` containing `InterfaceCOLMAP.exe`, `DensifyPointCloud.exe`, `ReconstructMesh.exe`, and companion DLLs) and pass `--openmvs-bin`.
- **Linux:** Build from https://github.com/cdcseacave/openMVS or run via Docker, ensuring the binaries are on `PATH` or passed via `--openmvs-bin`.

---

## 3. Real-World Pitfalls & Solutions

### Pitfall 1: COLMAP 4.x ONNX / ALIKED CUDA 12/13 DLL Clash (`WinError 1114`)
- **Symptom:** Running `feature_extractor --FeatureExtraction.type ALIKED_N16ROT` on Windows systems with newer CUDA 13 drivers or mismatched `onnxruntime_providers_cuda.dll` crashes immediately with `WinError 1114` (DLL initialization routine failed).
- **Fix:** Stage 2 (`pipeline/02_run_sfm.py`) intentionally uses **native COLMAP GPU SIFT** (`--FeatureExtraction.type SIFT --FeatureExtraction.use_gpu 1`) paired with `SIFT_BRUTEFORCE` + `guided_matching` over spatio-temporal `pairs.txt`. It avoids ONNX Runtime CUDA EP completely while achieving `~0.58 px` mean reprojection error and 100% registration on multi-flight drone orbits.

### Pitfall 2: OpenMVS `-w` Working Path & Native `*.log` Location
- **Symptom:** Invoking `InterfaceCOLMAP.exe` or `DensifyPointCloud.exe` without `-w <UNDIST_DIR>` causes OpenMVS to resolve relative `images/` paths against the output directory and fail silently. Additionally, on Windows OpenMVS v2.4.0 writes its verbose `DensifyPointCloud-*.log` / `ReconstructMesh-*.log` progress files into `-w <UNDIST_DIR>` rather than stdout or `cwd`.
- **Fix:** `pipeline/03_run_openmvs.py` **always** passes `-w <undistorted_dir>`, prepends `openmvs_bin` to `PATH` so MSVC/OpenMVS DLLs resolve, tails the newest matching `*.log` across **both** `<undistorted_dir>` and `cwd` every `0.4 s` into `logs/<step>.log` + `progress.json`, and prints the full native log tail on any error.

### Pitfall 3: Apache `.gz` Double-Decompression Trap on Shared Hosting
- **Symptom:** Naming compressed WebGL payloads `*.bin.gz` or `*.pcloud.gz` causes Apache (`mod_mime` / `mod_negotiation` / `mod_deflate`) to either attach `Content-Encoding: gzip` (causing `fetch()` to transparently gunzip the stream so client-side `DecompressionStream('gzip')` fails with corrupt magic bytes!) or double-compress the already-compressed binary.
- **Fix:**
  1. Compressed files use the opaque **`.pcloud`** extension (never `.gz`).
  2. `viewer/.htaccess` explicitly sets `AddType application/octet-stream .pcloud` and `SetEnv no-gzip 1`, ensuring exact byte-for-byte delivery to `gunzipBuffer()` in `viewer/app.js`.

### Pitfall 4: COLMAP 3.x vs 4.x Flag Drift & Subprocess Output Buffering
- **Symptom:** Using `subprocess.run(..., capture_output=True)` swallows all COLMAP logs until exit, making 20-minute SfM runs look frozen. Moreover, COLMAP 4.x renamed `--Mapper.snapshot_images_freq` to `--Mapper.snapshot_frames_freq` and defaults to `--log_target stderr_and_file`.
- **Fix:** `pipeline/02_run_sfm.py` probes `colmap mapper -h` once via `detect_colmap_capabilities()` to select the exact COLMAP 3.x/4.x logging and snapshot flags dynamically, streams output line-by-line via `pipeline/progress.py`, and uses `global_mapper` early-accept gating to skip redundant incremental passes when global SfM already achieved `>= 95%` registration at `<= 1.2 px`.
