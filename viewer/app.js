/**
 * ============================================================================
 * GENERIC WEBGL2 PROGRESSIVE DRONE POINT CLOUD VIEWER
 * ============================================================================
 * - Reads site title, layers, presets, ENU bounds & POIs from `data/manifest.json`
 * - Progressive `.pcloud` (`CHPC` v1) gzip streaming (DecompressionStream + fflate)
 * - ENU (X=East, Y=North, Z=Up) -> Three.js (X, Z, -Y)
 * - Circular antialiased ShaderMaterial (Real RGB / DSM Turbo / Hybrid Relief)
 * - Dynamic GPU point budget via `setDrawRange` on pre-shuffled arrays
 * - Dynamic ENU X/Y/Z clipping, GPS flight paths, POI pins & 3D point measurement
 */

(function () {
  'use strict';

  const container = document.getElementById('webgl-canvas');
  const markersContainer = document.getElementById('markersContainer');
  const streamPill = document.getElementById('streamPill');
  const streamText = document.getElementById('streamText');
  const headerTitleEl = document.getElementById('headerTitle');
  const headerSubtitleEl = document.getElementById('headerSubtitle');
  const metricPointsEl = document.getElementById('metricPoints');
  const metricSizeEl = document.getElementById('metricSize');
  const compassDial = document.getElementById('compassDial');
  const toastBanner = document.getElementById('toastBanner');
  const layerTogglesEl = document.getElementById('layerToggles');

  let isLightTheme = false;
  const isMobile = window.innerWidth < 768;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x090d16);

  const camera = new THREE.PerspectiveCamera(
    48,
    container.clientWidth / Math.max(1, container.clientHeight),
    0.2,
    2500
  );
  camera.position.set(0, 30, 45);

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: false,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 1.5;
  controls.maxDistance = 800.0;
  controls.maxPolarAngle = Math.PI * 0.495;
  controls.target.set(0, 0, 0);

  const gridHelper = new THREE.GridHelper(260, 52, 0x334155, 0x1e293b);
  gridHelper.position.set(0, -5, 0);
  gridHelper.material.transparent = true;
  gridHelper.material.opacity = 0.25;
  scene.add(gridHelper);

  const sharedUniforms = {
    uPointSize: { value: 2.6 },
    uPixelRatio: { value: Math.min(window.devicePixelRatio || 1, 2) },
    uColorMode: { value: 0 },
    uZRange: { value: new THREE.Vector2(-6.0, 14.0) },
    uClipMin: { value: new THREE.Vector3(-500, -500, -200) },
    uClipMax: { value: new THREE.Vector3(500, 500, 200) },
  };

  const VERTEX_SHADER = `
    attribute vec3 color;
    uniform float uPointSize;
    uniform float uPixelRatio;
    uniform int uColorMode;
    uniform vec2 uZRange;
    uniform vec3 uClipMin;
    uniform vec3 uClipMax;

    varying vec3 vColor;
    varying float vVisible;

    vec3 turboColormap(float t) {
      vec3 c0 = vec3(0.114, 0.067, 0.278);
      vec3 c1 = vec3(0.133, 0.490, 0.878);
      vec3 c2 = vec3(0.180, 0.835, 0.569);
      vec3 c3 = vec3(0.929, 0.808, 0.192);
      vec3 c4 = vec3(0.847, 0.196, 0.078);
      float x = clamp(t, 0.0, 1.0) * 4.0;
      if (x < 1.0) return mix(c0, c1, x);
      if (x < 2.0) return mix(c1, c2, x - 1.0);
      if (x < 3.0) return mix(c2, c3, x - 2.0);
      return mix(c3, c4, x - 3.0);
    }

    void main() {
      vec3 enu = vec3(position.x, -position.z, position.y);
      if (
        enu.x < uClipMin.x || enu.y < uClipMin.y || enu.z < uClipMin.z ||
        enu.x > uClipMax.x || enu.y > uClipMax.y || enu.z > uClipMax.z
      ) {
        vVisible = 0.0;
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        return;
      }

      vVisible = 1.0;
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mvPosition;

      float dist = max(1.5, -mvPosition.z);
      gl_PointSize = clamp(uPointSize * (28.0 / dist) * uPixelRatio, 1.5 * uPixelRatio, 18.0 * uPixelRatio);

      float t = clamp((enu.z - uZRange.x) / max(0.05, uZRange.y - uZRange.x), 0.0, 1.0);
      if (uColorMode == 1) {
        vColor = turboColormap(t);
      } else if (uColorMode == 2) {
        float contour = 0.90 + 0.10 * smoothstep(0.85, 0.98, fract(enu.z * 0.5));
        vColor = mix(color, turboColormap(t), 0.44) * contour;
      } else {
        vColor = color;
      }
    }
  `;

  const FRAGMENT_SHADER = `
    varying vec3 vColor;
    varying float vVisible;

    void main() {
      if (vVisible < 0.5) discard;
      vec2 c = gl_PointCoord - vec2(0.5);
      float r2 = dot(c, c);
      if (r2 > 0.25) discard;
      float alpha = 1.0 - smoothstep(0.16, 0.25, r2);
      if (alpha < 0.15) discard;
      float shade = 1.0 - r2 * 0.35;
      gl_FragColor = vec4(vColor * shade, alpha);
    }
  `;

  const cloudMaterial = new THREE.ShaderMaterial({
    uniforms: sharedUniforms,
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    transparent: true,
    depthTest: true,
    depthWrite: true,
  });

  const state = {
    quality: isMobile ? 'eco' : 'standard',
    qualityRatios: { eco: 0.25, standard: 0.65, ultra: 1.0 },
    slots: {},       // slotId -> { pointsObj, count, pos, bbox, visible }
    slotOrder: [],
    trajGroup: new THREE.Group(),
    poiItems: [],
    showPois: true,
    showTraj: true,
    measureActive: false,
    measurePoints: [],
    measureGroup: new THREE.Group(),
    globalBbox: { min: [-100, -100, -20], max: [100, 100, 30], zP02: -6, zP98: 14 },
    clipBoundsInitialized: false,
    cameraInitialized: false,
  };

  scene.add(state.trajGroup);
  scene.add(state.measureGroup);

  let toastTimer = null;
  function showToast(msg) {
    toastBanner.textContent = msg;
    toastBanner.classList.add('visible');
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastBanner.classList.remove('visible'), 2600);
  }

  function enuToThree(enu) {
    return new THREE.Vector3(enu[0], enu[2], -enu[1]);
  }

  function threeToEnu(v) {
    return new THREE.Vector3(v.x, -v.z, v.y);
  }

  let camAnim = null;
  function animateCameraTo(camEnu, targetEnu, durationMs = 700) {
    camAnim = {
      startPos: camera.position.clone(),
      startTarget: controls.target.clone(),
      endPos: enuToThree(camEnu),
      endTarget: enuToThree(targetEnu),
      startTime: performance.now(),
      durationMs,
    };
  }

  controls.addEventListener('start', () => {
    camAnim = null;
  });

  async function gunzipBuffer(arrayBuffer) {
    if ('DecompressionStream' in window) {
      try {
        const ds = new DecompressionStream('gzip');
        const stream = new Blob([arrayBuffer]).stream().pipeThrough(ds);
        return await new Response(stream).arrayBuffer();
      } catch (e) {
        // Fallback to fflate
      }
    }
    if (window.fflate && typeof window.fflate.gunzipSync === 'function') {
      const out = window.fflate.gunzipSync(new Uint8Array(arrayBuffer));
      return out.buffer;
    }
    throw new Error('Gzip decompression unavailable in this browser.');
  }

  function parsePcloud(buffer) {
    const dv = new DataView(buffer);
    const magic =
      String.fromCharCode(dv.getUint8(0)) +
      String.fromCharCode(dv.getUint8(1)) +
      String.fromCharCode(dv.getUint8(2)) +
      String.fromCharCode(dv.getUint8(3));
    if (magic !== 'CHPC') {
      throw new Error(`Invalid .pcloud header (${magic})`);
    }
    const version = dv.getUint32(4, true);
    const count = dv.getUint32(8, true);
    const minX = dv.getFloat32(16, true);
    const minY = dv.getFloat32(20, true);
    const minZ = dv.getFloat32(24, true);
    const maxX = dv.getFloat32(28, true);
    const maxY = dv.getFloat32(32, true);
    const maxZ = dv.getFloat32(36, true);
    const zP02 = dv.getFloat32(40, true);
    const zP98 = dv.getFloat32(44, true);

    const qXyz = new Uint16Array(buffer, 48, count * 3);
    const rgbU8 = new Uint8Array(buffer, 48 + count * 6, count * 3);

    const pos = new Float32Array(count * 3);
    const sx = Math.max(1e-5, maxX - minX) / 65535.0;
    const sy = Math.max(1e-5, maxY - minY) / 65535.0;
    const sz = Math.max(1e-5, maxZ - minZ) / 65535.0;

    for (let i = 0; i < count; i++) {
      const j = i * 3;
      const ex = minX + qXyz[j] * sx;
      const ey = minY + qXyz[j + 1] * sy;
      const ez = minZ + qXyz[j + 2] * sz;
      pos[j] = ex;
      pos[j + 1] = ez;
      pos[j + 2] = -ey;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(rgbU8, 3, true));

    const tMin = new THREE.Vector3(minX, minZ, -maxY);
    const tMax = new THREE.Vector3(maxX, maxZ, -minY);
    geometry.boundingBox = new THREE.Box3(tMin, tMax);
    geometry.boundingSphere = new THREE.Sphere();
    geometry.boundingBox.getBoundingSphere(geometry.boundingSphere);

    return {
      version,
      count,
      geometry,
      pos,
      bbox: { min: [minX, minY, minZ], max: [maxX, maxY, maxZ], zP02, zP98 },
    };
  }

  function applyQualityBudget() {
    const ratio = state.qualityRatios[state.quality] || 0.65;
    state.slotOrder.forEach((slotId) => {
      const entry = state.slots[slotId];
      if (!entry) return;
      const drawCount = Math.min(entry.count, Math.max(1000, Math.floor(entry.count * ratio)));
      entry.pointsObj.geometry.setDrawRange(0, drawCount);
    });
    updateMetricsDisplay();
  }

  function updateMetricsDisplay() {
    const ratio = state.qualityRatios[state.quality] || 0.65;
    let totalLoaded = 0;
    let totalRendered = 0;
    state.slotOrder.forEach((slotId) => {
      const entry = state.slots[slotId];
      if (!entry) return;
      totalLoaded += entry.count;
      if (entry.pointsObj.visible) {
        totalRendered += Math.floor(entry.count * ratio);
      }
    });
    if (totalLoaded > 0) {
      metricPointsEl.textContent = `${(totalRendered / 1e6).toFixed(2)} M`;
    }
  }

  function attachCloud(slotId, parsed, visible = true) {
    const prev = state.slots[slotId];
    if (prev) {
      scene.remove(prev.pointsObj);
      prev.pointsObj.geometry.dispose();
    } else {
      state.slotOrder.push(slotId);
    }
    const ptsObj = new THREE.Points(parsed.geometry, cloudMaterial);
    ptsObj.visible = visible;
    ptsObj.frustumCulled = false;
    scene.add(ptsObj);

    state.slots[slotId] = {
      pointsObj: ptsObj,
      count: parsed.count,
      pos: parsed.pos,
      bbox: parsed.bbox,
      visible,
    };

    expandGlobalBbox(parsed.bbox);
    applyQualityBudget();
  }

  function expandGlobalBbox(bbox) {
    if (!state.clipBoundsInitialized) {
      state.globalBbox = {
        min: [...bbox.min],
        max: [...bbox.max],
        zP02: bbox.zP02,
        zP98: bbox.zP98,
      };
      state.clipBoundsInitialized = true;
    } else {
      for (let k = 0; k < 3; k++) {
        state.globalBbox.min[k] = Math.min(state.globalBbox.min[k], bbox.min[k]);
        state.globalBbox.max[k] = Math.max(state.globalBbox.max[k], bbox.max[k]);
      }
      state.globalBbox.zP02 = Math.min(state.globalBbox.zP02, bbox.zP02);
      state.globalBbox.zP98 = Math.max(state.globalBbox.zP98, bbox.zP98);
    }
    sharedUniforms.uZRange.value.set(state.globalBbox.zP02, state.globalBbox.zP98);

    const cx = 0.5 * (state.globalBbox.min[0] + state.globalBbox.max[0]);
    const cy = 0.5 * (state.globalBbox.min[1] + state.globalBbox.max[1]);
    gridHelper.position.set(cx, state.globalBbox.min[2] - 1.0, -cy);

    syncClipSlidersToBbox();
  }

  // Clipping controls
  const clipZMax = document.getElementById('clipZMax');
  const clipXMin = document.getElementById('clipXMin');
  const clipXMax = document.getElementById('clipXMax');
  const clipYMin = document.getElementById('clipYMin');
  const clipYMax = document.getElementById('clipYMax');

  function syncClipSlidersToBbox() {
    const b = state.globalBbox;
    const x0 = Math.floor(b.min[0]) - 1;
    const x1 = Math.ceil(b.max[0]) + 1;
    const y0 = Math.floor(b.min[1]) - 1;
    const y1 = Math.ceil(b.max[1]) + 1;
    const z0 = Math.floor(b.min[2]) - 1;
    const z1 = Math.ceil(b.max[2]) + 1;

    clipZMax.min = z0; clipZMax.max = z1; clipZMax.value = z1;
    clipXMin.min = x0; clipXMin.max = x1; clipXMin.value = x0;
    clipXMax.min = x0; clipXMax.max = x1; clipXMax.value = x1;
    clipYMin.min = y0; clipYMin.max = y1; clipYMin.value = y0;
    clipYMax.min = y0; clipYMax.max = y1; clipYMax.value = y1;

    updateClipUniforms();
  }

  function updateClipUniforms() {
    let xmin = parseFloat(clipXMin.value);
    let xmax = parseFloat(clipXMax.value);
    if (xmin > xmax) { const t = xmin; xmin = xmax; xmax = t; }
    let ymin = parseFloat(clipYMin.value);
    let ymax = parseFloat(clipYMax.value);
    if (ymin > ymax) { const t = ymin; ymin = ymax; ymax = t; }
    const zmax = parseFloat(clipZMax.value);
    const zmin = Math.floor(state.globalBbox.min[2]) - 5;

    sharedUniforms.uClipMin.value.set(xmin, ymin, zmin);
    sharedUniforms.uClipMax.value.set(xmax, ymax, zmax);

    document.getElementById('clipZMaxVal').textContent =
      zmax >= parseFloat(clipZMax.max) - 0.1 ? 'Auto' : `${zmax.toFixed(1)} m`;
    document.getElementById('clipXVal').textContent = `${xmin.toFixed(0)}..${xmax.toFixed(0)} m`;
    document.getElementById('clipYVal').textContent = `${ymin.toFixed(0)}..${ymax.toFixed(0)} m`;
  }

  [clipZMax, clipXMin, clipXMax, clipYMin, clipYMax].forEach((el) => {
    el.addEventListener('input', updateClipUniforms);
  });
  document.getElementById('btnResetClip').addEventListener('click', () => {
    syncClipSlidersToBbox();
    showToast('ENU clipping bounds reset');
  });

  function buildTrajectoriesAndPois(trajData, presets) {
    while (state.trajGroup.children.length) {
      state.trajGroup.remove(state.trajGroup.children[0]);
    }
    markersContainer.innerHTML = '';
    state.poiItems = [];

    (trajData.flights || []).forEach((fl) => {
      if (!fl.points || fl.points.length < 2) return;
      const pts3 = fl.points.map((p) => enuToThree(p.enu));
      const geom = new THREE.BufferGeometry().setFromPoints(pts3);
      const mat = new THREE.LineBasicMaterial({
        color: new THREE.Color(fl.color || '#14B8A6'),
        transparent: true,
        opacity: 0.88,
      });
      state.trajGroup.add(new THREE.Line(geom, mat));
    });

    (trajData.pois || []).forEach((poi) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'poi-pin';
      btn.innerHTML = `<span class="poi-dot"></span><span></span>`;
      btn.querySelector('span:last-child').textContent = poi.label;
      btn.addEventListener('click', () => {
        const preset = presets.find((pr) => pr.id === poi.preset);
        if (preset) {
          selectPresetButton(preset.id);
          animateCameraTo(preset.cam_enu, preset.target_enu, 750);
        } else {
          const c = poi.enu;
          animateCameraTo([c[0] - 18, c[1] - 18, c[2] + 14], c, 750);
        }
      });
      markersContainer.appendChild(btn);
      state.poiItems.push({
        data: poi,
        threePos: enuToThree(poi.enu),
        dom: btn,
      });
    });
  }

  function updatePoiProjection() {
    if (!state.showPois) {
      markersContainer.style.display = 'none';
      return;
    }
    markersContainer.style.display = 'block';
    const w = container.clientWidth;
    const h = container.clientHeight;
    const tmp = new THREE.Vector3();
    state.poiItems.forEach((item) => {
      tmp.copy(item.threePos).project(camera);
      if (tmp.z > -1 && tmp.z < 1) {
        const sx = (tmp.x * 0.5 + 0.5) * w;
        const sy = (-tmp.y * 0.5 + 0.5) * h;
        item.dom.style.left = `${sx}px`;
        item.dom.style.top = `${sy}px`;
        item.dom.style.display = 'inline-flex';
      } else {
        item.dom.style.display = 'none';
      }
    });
  }

  const presetsCarousel = document.getElementById('presetsCarousel');
  let activePresetId = '';

  function selectPresetButton(id) {
    activePresetId = id;
    presetsCarousel.querySelectorAll('.preset-pill').forEach((b) => {
      b.classList.toggle('active', b.dataset.preset === id);
    });
  }

  function buildPresetsCarousel(presets) {
    presetsCarousel.innerHTML = '';
    if (!presets.length) return;
    if (!activePresetId) activePresetId = presets[0].id;
    presets.forEach((pr) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'preset-pill' + (pr.id === activePresetId ? ' active' : '');
      btn.dataset.preset = pr.id;
      btn.textContent = pr.label;
      btn.addEventListener('click', () => {
        selectPresetButton(pr.id);
        animateCameraTo(pr.cam_enu, pr.target_enu, 750);
      });
      presetsCarousel.appendChild(btn);
    });
  }

  function computeDefaultPresets(bbox) {
    const cx = 0.5 * (bbox.min[0] + bbox.max[0]);
    const cy = 0.5 * (bbox.min[1] + bbox.max[1]);
    const cz = 0.5 * (bbox.min[2] + bbox.max[2]);
    const span = Math.max(15, Math.hypot(bbox.max[0] - bbox.min[0], bbox.max[1] - bbox.min[1]));
    return [
      {
        id: 'oblique',
        label: '3/4 Oblique',
        cam_enu: [cx - 0.55 * span, cy - 0.65 * span, cz + 0.45 * span],
        target_enu: [cx, cy, cz],
      },
      {
        id: 'zenith',
        label: 'Zenith',
        cam_enu: [cx, cy - 0.5, cz + 1.05 * span],
        target_enu: [cx, cy, cz],
      },
    ];
  }

  // Fast screen-space nearest-point picker
  const pickMat = new THREE.Matrix4();
  function pickNearestCloudPoint(clientX, clientY) {
    const rect = renderer.domElement.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    const px = clientX - rect.left;
    const py = clientY - rect.top;

    camera.updateMatrixWorld();
    pickMat.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const m = pickMat.elements;

    const cMin = sharedUniforms.uClipMin.value;
    const cMax = sharedUniforms.uClipMax.value;
    const ratio = state.qualityRatios[state.quality] || 0.65;

    let bestDistSq = 18 * 18;
    let bestZ = Infinity;
    let bestPt = null;

    const candidateClouds = state.slotOrder
      .map((k) => state.slots[k])
      .filter((c) => c && c.pointsObj.visible);

    for (let ci = 0; ci < candidateClouds.length; ci++) {
      const cloud = candidateClouds[ci];
      const pos = cloud.pos;
      const maxPts = Math.min(cloud.count, Math.floor(cloud.count * ratio));

      for (let i = 0; i < maxPts; i++) {
        const j = i * 3;
        const x = pos[j];
        const y = pos[j + 1];
        const z = pos[j + 2];

        const ey = -z;
        if (x < cMin.x || ey < cMin.y || y < cMin.z || x > cMax.x || ey > cMax.y || y > cMax.z) {
          continue;
        }

        const clipW = m[3] * x + m[7] * y + m[11] * z + m[15];
        if (clipW <= 0.1) continue;
        const invW = 1.0 / clipW;
        const ndcX = (m[0] * x + m[4] * y + m[8] * z + m[12]) * invW;
        const ndcY = (m[1] * x + m[5] * y + m[9] * z + m[13]) * invW;
        if (ndcX < -1 || ndcX > 1 || ndcY < -1 || ndcY > 1) continue;

        const sx = (ndcX * 0.5 + 0.5) * w;
        const sy = (-ndcY * 0.5 + 0.5) * h;
        const dx = sx - px;
        const dy = sy - py;
        const d2 = dx * dx + dy * dy;

        if (d2 < 144) {
          const ndcZ = (m[2] * x + m[6] * y + m[10] * z + m[14]) * invW;
          if (ndcZ < bestZ) {
            bestZ = ndcZ;
            bestDistSq = d2;
            bestPt = new THREE.Vector3(x, y, z);
          }
        } else if (bestZ === Infinity && d2 < bestDistSq) {
          bestDistSq = d2;
          bestPt = new THREE.Vector3(x, y, z);
        }
      }
    }
    return bestPt;
  }

  function renderMeasurementGraphics() {
    while (state.measureGroup.children.length) {
      state.measureGroup.remove(state.measureGroup.children[0]);
    }
    const pts = state.measurePoints;
    pts.forEach((p, idx) => {
      const sph = new THREE.Mesh(
        new THREE.SphereGeometry(0.28, 16, 16),
        new THREE.MeshBasicMaterial({ color: idx === 0 ? 0x2dd4bf : 0xf59e0b })
      );
      sph.position.copy(p.three);
      state.measureGroup.add(sph);
    });

    if (pts.length === 2) {
      const lineGeom = new THREE.BufferGeometry().setFromPoints([pts[0].three, pts[1].three]);
      const lineMat = new THREE.LineBasicMaterial({ color: 0xfef08a, linewidth: 2 });
      state.measureGroup.add(new THREE.Line(lineGeom, lineMat));

      const a = pts[0].enu;
      const b = pts[1].enu;
      const d3d = a.distanceTo(b);
      const dxy = Math.hypot(b.x - a.x, b.y - a.y);
      const dz = b.z - a.z;

      document.getElementById('valDist3d').textContent = `${d3d.toFixed(2)} m`;
      document.getElementById('valDistXY').textContent = `${dxy.toFixed(2)} m`;
      document.getElementById('valDistZ').textContent = `${dz >= 0 ? '+' : ''}${dz.toFixed(2)} m`;
      document.getElementById('measureHint').textContent = 'Measured (click any point to restart)';
    } else if (pts.length === 1) {
      document.getElementById('valDist3d').textContent = '—';
      document.getElementById('valDistXY').textContent = '—';
      document.getElementById('valDistZ').textContent = '—';
      document.getElementById('measureHint').textContent = 'Point A set — click point B…';
    } else {
      document.getElementById('valDist3d').textContent = '—';
      document.getElementById('valDistXY').textContent = '—';
      document.getElementById('valDistZ').textContent = '—';
      document.getElementById('measureHint').textContent = 'Click 2 points on the cloud…';
    }
  }

  let pointerDownXY = { x: 0, y: 0 };
  renderer.domElement.addEventListener('pointerdown', (e) => {
    pointerDownXY = { x: e.clientX, y: e.clientY };
  });

  renderer.domElement.addEventListener('pointerup', (e) => {
    if (!state.measureActive) return;
    const moved = Math.hypot(e.clientX - pointerDownXY.x, e.clientY - pointerDownXY.y);
    if (moved > 8) return;

    const hitThree = pickNearestCloudPoint(e.clientX, e.clientY);
    if (!hitThree) {
      showToast('No point hit here — zoom in or click directly on points');
      return;
    }
    if (state.measurePoints.length >= 2) {
      state.measurePoints = [];
    }
    state.measurePoints.push({
      three: hitThree,
      enu: threeToEnu(hitThree),
    });
    renderMeasurementGraphics();
  });

  // UI Wiring
  const btnMeasure = document.getElementById('btnMeasure');
  const measureHud = document.getElementById('measureHud');
  const controlsSheet = document.getElementById('controlsSheet');
  const btnOpenControls = document.getElementById('btnOpenControls');

  btnMeasure.addEventListener('click', () => {
    state.measureActive = !state.measureActive;
    btnMeasure.setAttribute('aria-pressed', String(state.measureActive));
    measureHud.classList.toggle('visible', state.measureActive);
    document.documentElement.classList.toggle('measure-active', state.measureActive);
    if (state.measureActive) {
      showToast('3D Measure active: click 2 points on the cloud');
    }
  });

  document.getElementById('btnMeasureClear').addEventListener('click', () => {
    state.measurePoints = [];
    renderMeasurementGraphics();
  });

  btnOpenControls.addEventListener('click', () => {
    const open = !controlsSheet.classList.contains('open');
    controlsSheet.classList.toggle('open', open);
    btnOpenControls.setAttribute('aria-expanded', String(open));
  });
  document.getElementById('closeControlsBtn').addEventListener('click', () => {
    controlsSheet.classList.remove('open');
    btnOpenControls.setAttribute('aria-expanded', 'false');
  });

  // Quality selector
  const qualityButtons = document.querySelectorAll('#qualityGroup .seg-btn');
  function setQualityUI(q) {
    state.quality = q;
    qualityButtons.forEach((b) => b.classList.toggle('active', b.dataset.quality === q));
    const labels = { eco: 'Eco Mobile (25%)', standard: 'Standard (65%)', ultra: 'Ultra HD (100%)' };
    document.getElementById('budgetLabel').textContent = labels[q] || q;
    applyQualityBudget();
  }
  qualityButtons.forEach((b) => {
    b.addEventListener('click', () => setQualityUI(b.dataset.quality));
  });
  setQualityUI(state.quality);

  // Color mode selector
  const colorButtons = document.querySelectorAll('#colorModeGroup .seg-btn');
  colorButtons.forEach((b) => {
    b.addEventListener('click', () => {
      sharedUniforms.uColorMode.value = parseInt(b.dataset.colormode, 10);
      colorButtons.forEach((x) => x.classList.toggle('active', x === b));
    });
  });

  // Point size slider
  const pointSizeSlider = document.getElementById('pointSizeSlider');
  pointSizeSlider.addEventListener('input', () => {
    const v = parseFloat(pointSizeSlider.value);
    sharedUniforms.uPointSize.value = v;
    document.getElementById('pointSizeVal').textContent = `${v.toFixed(1)} px`;
  });

  // Dynamic Layer Toggle Builder
  const slotVisibility = {};
  function createDynamicLayerToggles(layers, pendingLayers) {
    const rowTraj = document.getElementById('rowTrajToggle');
    const seenSlots = new Set();

    layers.forEach((layer) => {
      if (layer.role === 'preview' && layer.replaces) return;
      const slotId = layer.replaces || layer.id;
      if (seenSlots.has(slotId)) return;
      seenSlots.add(slotId);
      slotVisibility[slotId] = true;

      const labelEl = document.createElement('label');
      labelEl.className = 'toggle-row';
      labelEl.innerHTML = `
        <span></span>
        <input type="checkbox" class="sr-only-input" checked>
        <span class="toggle-switch" aria-hidden="true"></span>
      `;
      labelEl.querySelector('span:first-child').textContent = `☁️ ${layer.label || layer.id}`;
      const chk = labelEl.querySelector('input');
      chk.addEventListener('change', () => {
        slotVisibility[slotId] = chk.checked;
        if (state.slots[slotId]) {
          state.slots[slotId].pointsObj.visible = chk.checked;
        }
        updateMetricsDisplay();
      });
      layerTogglesEl.insertBefore(labelEl, rowTraj);
    });

    (pendingLayers || []).forEach((pl) => {
      const labelEl = document.createElement('label');
      labelEl.className = 'toggle-row disabled';
      labelEl.innerHTML = `
        <span></span>
        <input type="checkbox" class="sr-only-input" disabled>
        <span class="toggle-switch" aria-hidden="true"></span>
      `;
      labelEl.querySelector('span:first-child').textContent = `⏳ ${pl.label || pl.id}`;
      layerTogglesEl.insertBefore(labelEl, rowTraj);
    });
  }

  // Trajectories & POI toggles
  const toggleTraj = document.getElementById('toggleTraj');
  const togglePois = document.getElementById('togglePois');
  const btnToggleTrajQuick = document.getElementById('btnToggleTrajQuick');
  const btnTogglePoisQuick = document.getElementById('btnTogglePoisQuick');

  function setTrajVisible(val) {
    state.showTraj = val;
    state.trajGroup.visible = val;
    toggleTraj.checked = val;
    btnToggleTrajQuick.setAttribute('aria-pressed', String(val));
  }
  toggleTraj.addEventListener('change', () => setTrajVisible(toggleTraj.checked));
  btnToggleTrajQuick.addEventListener('click', () => setTrajVisible(!state.showTraj));

  function setPoisVisible(val) {
    state.showPois = val;
    togglePois.checked = val;
    btnTogglePoisQuick.setAttribute('aria-pressed', String(val));
  }
  togglePois.addEventListener('change', () => setPoisVisible(togglePois.checked));
  btnTogglePoisQuick.addEventListener('click', () => setPoisVisible(!state.showPois));

  // Theme toggle
  const btnThemeToggle = document.getElementById('btnThemeToggle');
  btnThemeToggle.addEventListener('click', () => {
    isLightTheme = !isLightTheme;
    document.documentElement.setAttribute('data-theme', isLightTheme ? 'light' : 'dark');
    btnThemeToggle.setAttribute('aria-pressed', String(isLightTheme));
    scene.background = new THREE.Color(isLightTheme ? 0xe2e8f0 : 0x090d16);
    showToast(isLightTheme ? 'Sunlight theme enabled' : 'Dark Studio theme enabled');
  });

  // Compass North reset
  document.getElementById('btnCompassReset').addEventListener('click', () => {
    const offset = camera.position.clone().sub(controls.target);
    const horiz = Math.max(1, Math.hypot(offset.x, offset.z));
    const t = controls.target;
    const camPos = new THREE.Vector3(t.x, t.y + Math.max(4, offset.y), t.z + horiz);
    const camEnu = threeToEnu(camPos);
    const tgtEnu = threeToEnu(t);
    camEnu.z = Math.max(tgtEnu.z + 2, camEnu.z);
    animateCameraTo([camEnu.x, camEnu.y, camEnu.z], [tgtEnu.x, tgtEnu.y, tgtEnu.z], 650);
    showToast('🧭 North aligned up');
  });

  window.addEventListener('resize', () => {
    const w = container.clientWidth;
    const h = Math.max(1, container.clientHeight);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });

  async function loadProgressive() {
    try {
      const mResp = await fetch(`data/manifest.json?v=${Date.now()}`);
      if (!mResp.ok) throw new Error(`HTTP ${mResp.status} loading manifest.json`);
      const manifest = await mResp.json();

      if (manifest.site) {
        document.title = `${manifest.site} — 3D Point Cloud Viewer`;
        headerTitleEl.textContent = manifest.site;
      }
      if (manifest.subtitle || manifest.coordinate_frame) {
        headerSubtitleEl.textContent = manifest.subtitle || manifest.coordinate_frame;
      }
      if (manifest.summary && typeof manifest.summary.total_download_mb === 'number') {
        metricSizeEl.textContent = `${manifest.summary.total_download_mb.toFixed(2)} MB`;
      }

      const layers = manifest.layers || [];
      createDynamicLayerToggles(layers, manifest.pending_layers);

      let presets = manifest.presets || [];
      if (presets.length > 0) {
        buildPresetsCarousel(presets);
        const initial = presets[0];
        camera.position.copy(enuToThree(initial.cam_enu));
        controls.target.copy(enuToThree(initial.target_enu));
        controls.update();
        state.cameraInitialized = true;
      }

      if (manifest.trajectories_file) {
        fetch(`data/${manifest.trajectories_file}?v=${manifest.version_ts || 1}`)
          .then((r) => (r.ok ? r.json() : null))
          .then((traj) => {
            if (traj) buildTrajectoriesAndPois(traj, presets);
          })
          .catch(() => {});
      }

      // Stream preview layers first, then primary/secondary layers
      const orderedLayers = [
        ...layers.filter((l) => l.role === 'preview'),
        ...layers.filter((l) => l.role !== 'preview'),
      ];

      for (const layer of orderedLayers) {
        const ptsLabel =
          layer.points >= 1e6
            ? `${(layer.points / 1e6).toFixed(2)}M pts`
            : `${Math.round(layer.points / 1000)}k pts`;
        streamText.textContent = `Streaming ${layer.label || layer.id} (${ptsLabel})…`;
        const resp = await fetch(`data/${layer.file}?v=${layer.version_ts || manifest.version_ts || 1}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status} loading ${layer.file}`);
        const gzBuf = await resp.arrayBuffer();
        const rawBuf = await gunzipBuffer(gzBuf);
        const parsed = parsePcloud(rawBuf);
        const slotId = layer.replaces || layer.id;
        const vis = slotVisibility[slotId] !== false;
        attachCloud(slotId, parsed, vis);

        if (!state.cameraInitialized && presets.length === 0) {
          presets = computeDefaultPresets(state.globalBbox);
          buildPresetsCarousel(presets);
          camera.position.copy(enuToThree(presets[0].cam_enu));
          controls.target.copy(enuToThree(presets[0].target_enu));
          controls.update();
          state.cameraInitialized = true;
        }
      }

      streamPill.classList.add('ready');
      const totalPts = state.slotOrder.reduce((acc, k) => acc + (state.slots[k]?.count || 0), 0);
      streamText.textContent = `Ready (${(totalPts / 1e6).toFixed(2)}M pts)`;
    } catch (err) {
      console.error(err);
      streamText.textContent = `Error: ${err.message}`;
    }
  }

  function animate() {
    requestAnimationFrame(animate);

    if (camAnim) {
      const elapsed = performance.now() - camAnim.startTime;
      const t = Math.min(1.0, elapsed / camAnim.durationMs);
      const ease = 1.0 - Math.pow(1.0 - t, 3);
      camera.position.lerpVectors(camAnim.startPos, camAnim.endPos, ease);
      controls.target.lerpVectors(camAnim.startTarget, camAnim.endTarget, ease);
      if (t >= 1.0) camAnim = null;
    }

    controls.update();

    if (compassDial) {
      const azimuth = controls.getAzimuthalAngle();
      compassDial.style.transform = `rotate(${azimuth}rad)`;
    }

    updatePoiProjection();
    renderer.render(scene, camera);
  }

  animate();
  loadProgressive();
})();
