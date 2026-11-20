# `.pcloud` (`CHPC` v1) & Manifest Specification

## 1. Binary Container (`.pcloud`)

Each `.pcloud` file is a **gzip level-9** compressed stream containing a 48-byte Little-Endian header followed by randomly shuffled quantized positions and 8-bit sRGB vertex colors.

### Uncompressed Byte Layout

| Offset (bytes) | Size | Type (LE) | Field | Description |
|---:|---:|---|---|---|
| `0` | 4 | `char[4]` | `magic` | ASCII `"CHPC"` (`0x43 0x48 0x50 0x43`) |
| `4` | 4 | `uint32` | `version` | Format version (`1`) |
| `8` | 4 | `uint32` | `count` | Number of 3D points $N$ |
| `12` | 4 | `uint32` | `flags` | Reserved bitmask (`0`) |
| `16` | 4 | `float32` | `minX` | ENU East minimum (m) |
| `20` | 4 | `float32` | `minY` | ENU North minimum (m) |
| `24` | 4 | `float32` | `minZ` | ENU Up minimum (m) |
| `28` | 4 | `float32` | `maxX` | ENU East maximum (m) |
| `32` | 4 | `float32` | `maxY` | ENU North maximum (m) |
| `36` | 4 | `float32` | `maxZ` | ENU Up maximum (m) |
| `40` | 4 | `float32` | `zP02` | 2nd percentile ENU Z (for DSM colormap range) |
| `44` | 4 | `float32` | `zP98` | 98th percentile ENU Z (for DSM colormap range) |
| `48` | $6N$ | `uint16[3N]` | `xyz_u16` | Quantized ENU `[qx, qy, qz]` in `[0, 65535]` |
| `48 + 6N` | $3N$ | `uint8[3N]` | `rgb_u8` | sRGB `[r, g, b]` in `[0, 255]` |

### Dequantization & Coordinate Mapping

For point $i \in [0, N-1]$:

$$X_{\text{ENU}} = \text{minX} + q_x \cdot \frac{\max(\text{maxX}-\text{minX}, 10^{-5})}{65535}$$
$$Y_{\text{ENU}} = \text{minY} + q_y \cdot \frac{\max(\text{maxY}-\text{minY}, 10^{-5})}{65535}$$
$$Z_{\text{ENU}} = \text{minZ} + q_z \cdot \frac{\max(\text{maxZ}-\text{minZ}, 10^{-5})}{65535}$$

Mapping to **Three.js** ($+Y$ Up, $-Z$ Forward = North):

$$(X_{\text{Three}}, Y_{\text{Three}}, Z_{\text{Three}}) = (X_{\text{ENU}}, Z_{\text{ENU}}, -Y_{\text{ENU}})$$

### Progressive LOD via Random Shuffling

Points are randomly permuted prior to serialization (`np.random.default_rng(42)`). Because any prefix `0..K` is a uniform spatial sample of the entire cloud, the WebGL2 viewer switches instantaneously between **Eco Mobile (25%)**, **Standard (65%)**, and **Ultra HD (100%)** using `geometry.setDrawRange(0, K)` with zero GPU buffer reallocation.

---

## 2. `manifest.json` Schema

The viewer reads `data/manifest.json` at startup:

```json
{
  "version": 1,
  "version_ts": 1700000000,
  "site": "Site Title",
  "subtitle": "Georeferenced ENU Photogrammetry",
  "coordinate_frame": "ENU (X=East, Y=North, Z=Up) -> Three.js (X, Z, -Y)",
  "enu_origin": { "lat0": 0.0, "lon0": 0.0, "abs_alt_ref": 0.0 },
  "layers": [
    {
      "id": "site_preview",
      "label": "Quick Preview",
      "file": "site_preview.pcloud",
      "role": "preview",
      "replaces": "site_hd",
      "voxel_cm": 25.0,
      "version_ts": 1700000000,
      "points": 220000,
      "bytes": 1950000,
      "bbox_enu": { "min": [-20, -20, -5], "max": [20, 20, 15], "z_p02": -3.0, "z_p98": 12.0 }
    },
    {
      "id": "site_hd",
      "label": "Main Reconstruction (HD)",
      "file": "site_hd.pcloud",
      "role": "primary",
      "voxel_cm": 4.0,
      "version_ts": 1700000000,
      "points": 1400000,
      "bytes": 12450000,
      "bbox_enu": { "min": [-20, -20, -5], "max": [20, 20, 15], "z_p02": -3.0, "z_p98": 12.0 }
    }
  ],
  "pending_layers": [],
  "trajectories_file": "trajectories.json",
  "presets": [
    { "id": "oblique_34", "label": "3/4 Oblique", "cam_enu": [-20, -25, 22], "target_enu": [0, 0, 2] }
  ],
  "summary": { "active_points": 1400000, "total_download_mb": 13.73, "flights_count": 1 }
}
```

---

## 3. `trajectories.json` Schema (Optional)

```json
{
  "enu_origin": { "lat0": 0.0, "lon0": 0.0, "abs_alt_ref": 0.0 },
  "flights": [
    {
      "id": "flight_01",
      "label": "Orbit Flight",
      "color": "#F59E0B",
      "step_frames": 10,
      "points": [{ "f": 1, "enu": [0.0, 0.0, 10.0], "rel_alt": 10.0 }]
    }
  ],
  "pois": [
    { "id": "poi_1", "label": "Main Structure", "enu": [12.0, 4.0, 5.0], "preset": "oblique_34" }
  ]
}
```
