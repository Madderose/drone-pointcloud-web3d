# Third-Party Notices & Attributions

This project (`drone-pointcloud-web3d`) is licensed under the **MIT License** (see [`LICENSE`](LICENSE)).
It orchestrates external photogrammetry executables via command-line subprocesses and relies on open-source Python and JavaScript libraries. The licenses, copyrights, and academic citations of those third-party components are documented below.

---

## 1. Core External Photogrammetry Executables (Invoked via Subprocess)

### COLMAP & PyCOLMAP
- **Authors / Copyright:** Johannes L. Schönberger and contributors
- **License:** BSD 3-Clause License
- **Repository:** https://github.com/colmap/colmap
- **Academic Citations:**
  - Schönberger, J. L., & Frahm, J.-M. (2016). *Structure-from-Motion Revisited*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
  - Schönberger, J. L., Zheng, E., Frahm, J.-M., & Pollefeys, M. (2016). *Pixelwise View Selection for Unstructured Multi-View Stereo*. European Conference on Computer Vision (ECCV).

### OpenMVS (Multi-View Stereo Reconstruction Library)
- **Authors / Copyright:** Dan Cernea (cDc) and OpenMVS contributors
- **License:** **GNU Affero General Public License v3.0 (GNU AGPL-3.0)**
- **Repository:** https://github.com/cdcseacave/openMVS
- **Academic Citation:**
  - Cernea, D. (2020). *OpenMVS: Multi-View Stereo Reconstruction Library*.

> [!IMPORTANT]
> **OpenMVS AGPL-3.0 Legal Compliance Notice:**
> `drone-pointcloud-web3d` does **not** bundle, statically link, dynamically link, or embed any OpenMVS source code or precompiled binaries. Stage 3 (`pipeline/03_run_openmvs.py`) invokes user-installed external OpenMVS CLI executables (`InterfaceCOLMAP`, `DensifyPointCloud`, `ReconstructMesh`) strictly as separate OS subprocesses over standard intermediate files on disk.
>
> - Users must obtain and install OpenMVS independently and comply with the **GNU AGPL-3.0** license terms.
> - **Network / SaaS Copyleft Warning:** Running OpenMVS (or any modified version of it) as part of a network-accessible reconstruction service / SaaS backend triggers **AGPL-3.0 Section 13** remote network interaction obligations, requiring you to offer the corresponding source code of your OpenMVS service to network users.
> - **Static Web Viewer Exemption:** Serving static pre-exported `.pcloud` point-cloud files through our MIT-licensed Three.js `viewer/` contains **zero OpenMVS code** and does not link against or execute OpenMVS on the web server.

---

## 2. Python Libraries (Pipeline & Exporter)

| Component | License | Author / Project | URL / Citation |
|---|---|---|---|
| **Open3D** | MIT | Intel ISL / Open3D contributors | https://www.open3d.org/ — Zhou, Q.-Y., Park, J., & Koltun, V. (2018). *Open3D: A Modern Library for 3D Data Processing*. arXiv:1801.09847. |
| **OpenCV (`opencv-python`)** | Apache 2.0 | OpenCV team | https://opencv.org/ |
| **NumPy** | BSD 3-Clause | NumPy Developers | https://numpy.org/ |
| **SciPy** | BSD 3-Clause | SciPy Developers | https://scipy.org/ |
| **Matplotlib** | PSF / BSD-style | Matplotlib Development Team | https://matplotlib.org/ |
| **Pillow (`PIL`)** | HPND (Historical Permission Notice and Disclaimer) | Jeffrey A. Clark & contributors | https://python-pillow.org/ |
| **PyYAML** | MIT | Kirill Simonov & YAML contributors | https://pyyaml.org/ |
| **Rerun SDK (`rerun-sdk`, optional)** | Dual MIT / Apache 2.0 | Rerun Technologies AB | https://rerun.io/ |

---

## 3. WebGL2 Viewer Libraries (Client-Side Runtime)

| Component | License | Author / Project | URL |
|---|---|---|---|
| **Three.js** | MIT | Ricardo Cabello (`mrdoob`) & contributors | https://threejs.org/ |
| **fflate** | MIT | Arjun Barrett (`101arrowz`) | https://github.com/101arrowz/fflate |

*Note:* The viewer loads Three.js (`r128`) and `fflate` (`0.8.2`) from public CDNs (`cdnjs.cloudflare.com` / `cdn.jsdelivr.net`), or you may self-host those MIT assets alongside `viewer/`.

---

## 4. Benchmarked & Reference Neural Works (Not Redistributed)

Our `README.md` and `docs/ARCHITECTURE.md` compare this classical photogrammetry pipeline against neural 3D reconstruction baselines. Neither model weights nor code from these projects are bundled or required:

- **MASt3R & MASt3R-SLAM:** Naver Labs Europe — **CC BY-NC-SA 4.0** (Non-Commercial ShareAlike).
  - Leroy, V., Cabon, Y., & Revaud, J. (2024). *Grounding Image Matching in 3D with MASt3R*. ECCV 2024.
  - Murai, R., Dexheimer, E., & Davison, A. J. (2025). *MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors*. CVPR 2025.
- **Depth Anything 3 (DA3):** ByteDance / HKU — Apache 2.0 / model-specific license terms.
  - Yang, L. et al. *Depth Anything* series.
