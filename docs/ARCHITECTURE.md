# Architecture & Engineering Design

## 1. Why Classical Photogrammetry + Global `Sim(3)` Beats Uncalibrated Neural SLAM on Drone Loops

Modern feed-forward 3D foundation models (e.g., **MASt3R-SLAM**, **Depth Anything 3**) excel at rapid monocular previews, but exhibit two systematic failure modes on multi-flight outdoor drone acquisitions:
1. **Scale & Elevation Drift:** Without a global bundle adjustment over calibrated radial/tangential lens distortion (`OPENCV` model), trajectory drift accumulates across turns and elevation transitions, producing doubled roofs and split building facades when flights overlap.
2. **Per-Keyframe Warping Artifacts:** Attempting to force GPS consistency via per-frame pose pinning introduces high-frequency surface shear (`21–32 cm` planar RMSE on flat roofs).

This pipeline resolves both issues by keeping geometry **purely rigid** during bundle adjustment and applying **exactly one 7-DoF global `Sim(3)` transform** (`scale`, `rotation`, `translation`) at the end of Stage 2.

---

## 2. Stage-by-Stage Technical Breakdown

### Stage 1: Telemetry Parsing, Conservative Sky Masking & Match Graph (`pipeline/01_prepare_dataset.py`)
- **WGS84 → Local ENU Projection:**
  $$x_E = \Delta\lambda \cdot R_{\oplus} \cos(\phi_0), \quad y_N = \Delta\phi \cdot R_{\oplus}, \quad z_U = h - h_0$$
  where $R_{\oplus} = 6\,378\,137\text{ m}$ and $(\phi_0, \lambda_0, h_0)$ is derived automatically from the first frame or fixed manually for multi-scan alignment.
- **Conservative Top-Connected Sky Mask:**
  Operates on a `640x360` proxy image. Candidate sky pixels require low Sobel gradient magnitude (`< 32.0`) combined with either bright desaturated HSV or blue-dominant chroma in the upper 45% of the frame. Only connected components touching row `y = 0` are kept and eroded by 1 iteration (`3x3`), preventing false masking of horizon treetops or sunlit walls.
- **Bounded Spatio-Temporal `pairs.txt`:**
  Instead of $O(N^2)$ exhaustive matching (which explodes on 300+ frame multi-flight datasets), pairs combine:
  - Sequential neighbors (`±temporal_window`) sorted by original frame index within each video.
  - Spatial `k`-NN neighbors (`gps_neighbors`) in ENU coordinates across all videos and stills.

### Stage 2: Native GPU SIFT SfM & Rigid `Sim(3)` Alignment (`pipeline/02_run_sfm.py`)
- **Feature Extraction:** COLMAP `OPENCV` camera model (`single_camera_per_folder = 1` separates video streams from high-res stills) using native GPU SIFT (`max_num_features = 8192`).
- **Dual Mapper Auto-Selection:** Runs both `global_mapper` (fast global rotation/translation averaging) and incremental `mapper`, selecting the candidate maximizing registered source coverage, total registered images, and minimal mean reprojection error.
- **Filtering & Bundle Adjustment:** `point_filtering` (`min_track_len = 3`, `max_reproj_error = 2.0 px`) followed by strict `bundle_adjuster` refining focal length and radial/tangential distortion.
- **RANSAC + Huber IRLS `Sim(3)`:**
  - In **standalone mode**, camera projection centers are aligned to ENU GPS priors (`gps_priors_enu.txt`).
  - In **locked multi-scan mode** (`--reference-sparse`), the new model aligns via shared camera names directly onto an existing core `sparse_enu/0` reconstruction, achieving `< 10 cm` co-registration across disparate resolutions (e.g., 5 cm architectural core + 15 cm surrounding terrain).

### Stage 3: OpenMVS Visibility-Carved Reconstruction (`pipeline/03_run_openmvs.py`)
1. `InterfaceCOLMAP` ingests the undistorted `PINHOLE` workspace (`-w <undistorted>`).
2. `DensifyPointCloud` performs multi-view PatchMatch stereo with strict multi-view consistency (`--number-views-fuse 4 --fusion-filter 2`).
3. `ReconstructMesh` runs **Delaunay tetrahedralization + s-t graph-cut visibility carving**, reconstructing a watertight surface manifold that naturally rejects floating sky ghosts and behind-wall depth noise.
4. **Open3D `RaycastingScene` Proximity Filter:** Every raw dense point is queried against the carved mesh; points farther than `mesh_proximity_m` (default `0.20 m`) from the visibility-carved surface are pruned before voxel downsampling, Statistical Outlier Removal (SOR), and Radius Outlier Removal (ROR).

### Stage 4: Quantitative Geometric Audit (`pipeline/04_audit_cloud.py`)
- Evaluates 2D grid cell Z-spread (`p75 - p25` IQR and `p95 - p5` span) to detect residual double-layering.
- Computes optional RANSAC planar RMSE (cm) on user-specified building roofs or flat pads.
- Renders a 4-panel orthophoto / DSM / cross-section / summary figure and optional `.rrd` Rerun scene.

### Stage 5: Progressive `.pcloud` Streaming & WebGL2 Renderer (`web_exporter/`, `viewer/`)
- Quantizes ENU coordinates into 16-bit unsigned integers per axis (~0.1–2 mm precision over a 100 m–300 m site) and shuffles point order once at export time.
- The browser decompresses `.pcloud` via native `DecompressionStream('gzip')` (falling back to `fflate`), renders circular antialiased splats with a custom `THREE.ShaderMaterial`, and adapts point density (`25%` / `65%` / `100%`) in O(1) via `setDrawRange`.
