(function bootstrapReactUI() {
  const { useEffect, useMemo, useRef, useState } = React;
  const html = htm.bind(React.createElement);

  const MULTIMODAL_FILE_LABELS = {
    flair: "FLAIR",
    t1: "T1",
    t1ce: "T1ce",
    t2: "T2",
  };

  const ENGINE_OPTIONS = [
    { value: "all", label: "All engines (ensemble > deep > baseline)" },
    { value: "auto", label: "Auto (ensemble > deep > baseline)" },
    { value: "deep", label: "Deep model only" },
    { value: "ensemble", label: "Ensemble (k-fold checkpoints)" },
    { value: "baseline", label: "Baseline only" },
  ];

  const CLASS_ORDER = ["1", "2", "4"];

  function listOrNA(values) {
    return Array.isArray(values) && values.length ? values.join(", ") : "N/A";
  }

  function toNumber(value) {
    const num = Number(value);
    return Number.isFinite(num) ? num : 0;
  }

  function fixed(value, digits) {
    const num = Number(value);
    return Number.isFinite(num) ? num.toFixed(digits) : "N/A";
  }

  function safeList(value) {
    return Array.isArray(value) ? value : [];
  }

  function scaledVertices(vertices, scale) {
    if (!Array.isArray(vertices) || vertices.length === 0 || scale === 1) {
      return vertices;
    }

    const count = vertices.length;
    const centroid = vertices.reduce(
      (acc, point) => [acc[0] + point[0], acc[1] + point[1], acc[2] + point[2]],
      [0, 0, 0],
    );

    const cx = centroid[0] / count;
    const cy = centroid[1] / count;
    const cz = centroid[2] / count;

    return vertices.map((point) => [
      cx + (point[0] - cx) * scale,
      cy + (point[1] - cy) * scale,
      cz + (point[2] - cz) * scale,
    ]);
  }

  function buildMeshTrace(mesh, options) {
    const color = options.color;
    const targetOpacity = Number.isFinite(options.opacity) ? options.opacity : 0.8;
    const initialOpacity = Number.isFinite(options.initialOpacity)
      ? options.initialOpacity
      : targetOpacity;
    const name = options.name;
    const scale = Number.isFinite(options.scale) ? options.scale : 1;
    const showLegend = options.showLegend !== false;

    const vertices = scaledVertices(mesh.vertices, scale);
    return {
      type: "mesh3d",
      x: vertices.map((v) => v[0]),
      y: vertices.map((v) => v[1]),
      z: vertices.map((v) => v[2]),
      i: mesh.faces.map((f) => f[0]),
      j: mesh.faces.map((f) => f[1]),
      k: mesh.faces.map((f) => f[2]),
      opacity: initialOpacity,
      color,
      name,
      showlegend: showLegend,
      flatshading: false,
      showscale: false,
      hoverinfo: "skip",
      lighting: {
        ambient: 0.62,
        diffuse: 0.9,
        roughness: 0.28,
        fresnel: 0.16,
        specular: 0.46,
      },
      lightposition: {
        x: 120,
        y: 90,
        z: 180,
      },
    };
  }

  function buildScanlineTrace(brainMesh, initialOpacity) {
    const vertices = brainMesh && Array.isArray(brainMesh.vertices) ? brainMesh.vertices : [];
    if (!vertices.length) {
      return null;
    }

    let xMin = Number.POSITIVE_INFINITY;
    let xMax = Number.NEGATIVE_INFINITY;
    let yMin = Number.POSITIVE_INFINITY;
    let yMax = Number.NEGATIVE_INFINITY;
    let zMin = Number.POSITIVE_INFINITY;
    let zMax = Number.NEGATIVE_INFINITY;

    vertices.forEach((vertex) => {
      const x = Number(vertex[0]);
      const y = Number(vertex[1]);
      const z = Number(vertex[2]);

      if (x < xMin) xMin = x;
      if (x > xMax) xMax = x;
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
      if (z < zMin) zMin = z;
      if (z > zMax) zMax = z;
    });

    const trace = {
      type: "surface",
      x: [
        [xMin, xMax],
        [xMin, xMax],
      ],
      y: [
        [yMin, yMin],
        [yMax, yMax],
      ],
      z: [
        [zMin, zMin],
        [zMin, zMin],
      ],
      surfacecolor: [
        [0, 0],
        [0, 0],
      ],
      colorscale: [
        [0, "rgba(120, 220, 255, 0.0)"],
        [1, "rgba(120, 220, 255, 0.95)"],
      ],
      cmin: 0,
      cmax: 1,
      opacity: Number.isFinite(initialOpacity) ? initialOpacity : 0,
      showscale: false,
      hoverinfo: "skip",
      showlegend: false,
      lighting: {
        ambient: 1,
        diffuse: 0,
        roughness: 0.1,
        specular: 0,
      },
      contours: {
        x: { show: false },
        y: { show: false },
        z: { show: false },
      },
    };

    return {
      trace,
      zMin,
      zMax,
    };
  }

  function restyleMeshTrace(element, traceIndex, meshVertices, scale, opacity) {
    if (!element || !element.data || !element.data[traceIndex]) {
      return;
    }

    const vertices = scaledVertices(meshVertices, scale);
    Plotly.restyle(
      element,
      {
        x: [vertices.map((v) => v[0])],
        y: [vertices.map((v) => v[1])],
        z: [vertices.map((v) => v[2])],
        opacity: [opacity],
      },
      [traceIndex],
    );
  }

  function animateMeshBuildSequence(element, traceSpecs) {
    if (!window.anime || !element || !element.data || !Array.isArray(traceSpecs) || traceSpecs.length === 0) {
      return null;
    }

    const totalDurationMs = 3000;
    const leadInMs = 100;
    const availableMs = Math.max(200, totalDurationMs - leadInMs);

    traceSpecs.forEach((spec, index) => {
      spec.traceIndex = index;
    });

    const primarySpecs = traceSpecs.filter((spec) => spec.kind === "primary");
    const glowSpecs = traceSpecs.filter((spec) => spec.kind === "glow");
    const scanlineSpecs = traceSpecs.filter((spec) => spec.kind === "scanline");
    if (!primarySpecs.length && !scanlineSpecs.length) {
      return null;
    }

    const cameraStart = { x: 2.35, y: 1.85, z: 1.72 };
    const cameraEnd = { x: 1.55, y: 1.55, z: 1.25 };
    const cameraState = { t: 0 };

    const timeline = anime.timeline({ autoplay: true });

    timeline.add(
      {
        targets: cameraState,
        t: 1,
        duration: totalDurationMs,
        easing: "easeInOutSine",
        update: () => {
          if (!element || !element.data) {
            return;
          }

          const t = Number(cameraState.t);
          const mix = (a, b) => (a * (1 - t)) + (b * t);
          const baseX = mix(cameraStart.x, cameraEnd.x);
          const baseY = mix(cameraStart.y, cameraEnd.y);
          const baseZ = mix(cameraStart.z, cameraEnd.z);

          // Slight orbit while moving in, so the build-up feels spatially dynamic.
          const theta = (1 - t) * 0.55;
          const cosT = Math.cos(theta);
          const sinT = Math.sin(theta);
          const eyeX = (baseX * cosT) - (baseY * sinT);
          const eyeY = (baseX * sinT) + (baseY * cosT);

          Plotly.relayout(element, {
            "scene.camera": {
              eye: { x: eyeX, y: eyeY, z: baseZ },
              up: { x: 0, y: 0, z: 1 },
            },
          });
        },
      },
      0,
    );

    scanlineSpecs.forEach((spec) => {
      const scanline = {
        z: Number(spec.zMin),
        opacity: 0,
      };

      timeline.add(
        {
          targets: scanline,
          z: Number(spec.zMax),
          opacity: [0, 0.36, 0.22, 0],
          duration: totalDurationMs,
          easing: "easeInOutSine",
          update: () => {
            if (!element || !element.data || !element.data[spec.traceIndex]) {
              return;
            }

            Plotly.restyle(
              element,
              {
                z: [
                  [
                    [scanline.z, scanline.z],
                    [scanline.z, scanline.z],
                  ],
                ],
                opacity: [scanline.opacity],
              },
              [spec.traceIndex],
            );
          },
          complete: () => {
            if (!element || !element.data || !element.data[spec.traceIndex]) {
              return;
            }
            Plotly.restyle(element, { opacity: [0] }, [spec.traceIndex]);
          },
        },
        0,
      );
    });

    const orderedPrimary = primarySpecs
      .slice()
      .sort((a, b) => Number(a.revealOrder) - Number(b.revealOrder));

    const weights = orderedPrimary.map((_, index) => (index === 0 ? 1.25 : 1));
    const totalWeight = weights.reduce((sum, value) => sum + value, 0);
    const revealWindows = new Map();
    let cursor = leadInMs;

    orderedPrimary.forEach((spec, index) => {
      const slotDuration = Math.max(220, Math.round((availableMs * weights[index]) / totalWeight));
      revealWindows.set(Number(spec.revealOrder), {
        start: cursor,
        duration: slotDuration,
      });

      const targetOpacity = Number.isFinite(spec.targetOpacity) ? Number(spec.targetOpacity) : 0.8;
      const targetScale = Number.isFinite(spec.targetScale) ? Number(spec.targetScale) : 1;
      const startOpacity = Number.isFinite(spec.initialOpacity) ? Number(spec.initialOpacity) : 0;
      const startScale = Number.isFinite(spec.initialScale) ? Number(spec.initialScale) : targetScale;

      const controller = {
        opacity: startOpacity,
        scale: startScale,
        lastFrameMs: 0,
      };

      const overshootScale = targetScale * (targetScale >= 1 ? 1.045 : 1.07);
      timeline.add(
        {
          targets: controller,
          opacity: [
            startOpacity,
            targetOpacity * 0.76,
            targetOpacity,
          ],
          scale: [startScale, overshootScale, targetScale],
          duration: slotDuration,
          easing: index === 0 ? "easeOutQuint" : "easeOutCubic",
          update: () => {
            if (!element || !element.data || !element.data[spec.traceIndex]) {
              return;
            }

            const now = Date.now();
            if (now - controller.lastFrameMs < 40) {
              return;
            }
            controller.lastFrameMs = now;
            restyleMeshTrace(
              element,
              spec.traceIndex,
              spec.meshVertices,
              controller.scale,
              controller.opacity,
            );
          },
          complete: () => {
            restyleMeshTrace(element, spec.traceIndex, spec.meshVertices, targetScale, targetOpacity);
          },
        },
        cursor,
      );

      cursor += slotDuration;
    });

    glowSpecs
      .slice()
      .sort((a, b) => Number(a.revealOrder) - Number(b.revealOrder))
      .forEach((spec) => {
        const window = revealWindows.get(Number(spec.revealOrder));
        if (!window) {
          return;
        }

        const startOpacity = Number.isFinite(spec.initialOpacity) ? Number(spec.initialOpacity) : 0;
        const peakOpacity = Number.isFinite(spec.peakOpacity) ? Number(spec.peakOpacity) : 0.42;
        const initialScale = Number.isFinite(spec.initialScale) ? Number(spec.initialScale) : 1;
        const peakScale = Number.isFinite(spec.peakScale) ? Number(spec.peakScale) : initialScale;
        const finalScale = Number.isFinite(spec.targetScale) ? Number(spec.targetScale) : initialScale;

        const controller = {
          opacity: startOpacity,
          scale: initialScale,
          lastFrameMs: 0,
        };

        const duration = Math.max(360, Math.round(window.duration * 0.72));
        const startAt = window.start + Math.round(window.duration * 0.14);

        timeline.add(
          {
            targets: controller,
            opacity: [startOpacity, peakOpacity, 0],
            scale: [initialScale, peakScale, finalScale],
            duration,
            easing: "easeOutCubic",
            update: () => {
              if (!element || !element.data || !element.data[spec.traceIndex]) {
                return;
              }

              const now = Date.now();
              if (now - controller.lastFrameMs < 40) {
                return;
              }
              controller.lastFrameMs = now;

              restyleMeshTrace(
                element,
                spec.traceIndex,
                spec.meshVertices,
                controller.scale,
                controller.opacity,
              );
            },
            complete: () => {
              restyleMeshTrace(element, spec.traceIndex, spec.meshVertices, finalScale, 0);
            },
          },
          startAt,
        );
      });

    return timeline;
  }

  function buildMetricsRows(metrics, inference, classMetrics) {
    if (!metrics || !inference) {
      return [];
    }

    const rows = [
      {
        key: "Detected",
        value: metrics.detected ? "Yes" : "No",
        tone: metrics.detected ? "good" : "bad",
      },
      { key: "Tumor voxels", value: String(metrics.voxel_count ?? "N/A") },
      { key: "Occupancy", value: `${fixed(metrics.occupancy_percent, 4)} %` },
      { key: "Volume", value: `${fixed(metrics.volume_mm3, 2)} mm^3` },
      { key: "Volume", value: `${fixed(metrics.volume_ml, 3)} mL` },
      {
        key: "Equivalent diameter",
        value: `${fixed(metrics.equivalent_diameter_mm, 2)} mm`,
      },
      {
        key: "BBox min",
        value: `[${safeList(metrics.bbox_min).join(", ")}]`,
      },
      {
        key: "BBox max",
        value: `[${safeList(metrics.bbox_max).join(", ")}]`,
      },
      {
        key: "Extent",
        value: `[${safeList(metrics.extent_mm)
          .map((x) => fixed(x, 2))
          .join(", ")}] mm`,
      },
      {
        key: "Centroid voxel",
        value: `[${safeList(metrics.centroid_voxel)
          .map((x) => fixed(x, 2))
          .join(", ")}]`,
      },
      {
        key: "Centroid mm",
        value: `[${safeList(metrics.centroid_mm)
          .map((x) => fixed(x, 2))
          .join(", ")}]`,
      },
      {
        key: "Inference engine",
        value: String(inference.engine || "unknown"),
      },
      {
        key: "Ensemble size",
        value: Number.isFinite(inference.ensemble_size) ? String(inference.ensemble_size) : "N/A",
      },
      {
        key: "Probability mean",
        value: Number.isFinite(inference.probability_mean) ? fixed(inference.probability_mean, 4) : "N/A",
      },
      {
        key: "Probability max",
        value: Number.isFinite(inference.probability_max) ? fixed(inference.probability_max, 4) : "N/A",
      },
      { key: "Fold indices used", value: listOrNA(inference.fold_indices || []) },
      {
        key: "Checkpoints used",
        value: Array.isArray(inference.checkpoints)
          ? `${inference.checkpoints.length} selected`
          : inference.checkpoint || "N/A",
      },
    ];

    CLASS_ORDER.forEach((label) => {
      const entry = classMetrics ? classMetrics[label] : null;
      if (!entry) {
        return;
      }

      rows.push({
        key: `${entry.name} voxels`,
        value: String(entry.voxel_count ?? "N/A"),
        tone: entry.detected ? "good" : "",
      });
      rows.push({
        key: `${entry.name} volume`,
        value: `${fixed(entry.volume_ml, 3)} mL`,
      });
    });

    return rows;
  }

  function renderViewer(element, scene, options = {}) {
    const animateBuild = Boolean(options.animateBuild);

    if (!element) {
      return;
    }

    if (!scene || !scene.inputInfo) {
      Plotly.purge(element);
      return;
    }

    const inputInfo = scene.inputInfo;
    const tumorMesh = scene.tumorMesh;
    const brainMesh = scene.brainMesh;
    const classMeshes = safeList(scene.classMeshes).filter((entry) =>
      Boolean(entry && entry.mesh && safeList(entry.mesh.vertices).length),
    );

    const hasTumor = Boolean(tumorMesh && safeList(tumorMesh.vertices).length);
    const hasBrain = Boolean(brainMesh && safeList(brainMesh.vertices).length);
    const hasClassDataset = Boolean(scene.hasClassDataset);
    const hasClassMeshes = classMeshes.length > 0;

    if (!hasTumor && !hasBrain && !hasClassMeshes) {
      Plotly.purge(element);
      return;
    }

    const traceSpecs = [];
    let revealOrder = 0;

    if (hasBrain) {
      const brainOpacity = 0.34;
      const brainScale = 1;
      const brainInitialScale = animateBuild ? 0.82 : brainScale;

      traceSpecs.push({
        kind: "primary",
        revealOrder,
        meshVertices: brainMesh.vertices,
        targetScale: brainScale,
        initialScale: brainInitialScale,
        targetOpacity: brainOpacity,
        initialOpacity: animateBuild ? 0.02 : brainOpacity,
        trace: buildMeshTrace(brainMesh, {
          color: "#55a5ff",
          opacity: brainOpacity,
          initialOpacity: animateBuild ? 0.02 : brainOpacity,
          name: "Brain",
          scale: brainInitialScale,
        }),
      });
      revealOrder += 1;

      if (animateBuild) {
        const scanline = buildScanlineTrace(brainMesh, 0);
        if (scanline) {
          traceSpecs.push({
            kind: "scanline",
            trace: scanline.trace,
            zMin: scanline.zMin,
            zMax: scanline.zMax,
          });
        }
      }
    }

    if (hasClassDataset) {
      classMeshes.forEach((entry) => {
        const classOpacity = 0.8;
        const classScale = 0.92;
        const classInitialScale = animateBuild ? classScale * 0.82 : classScale;
        const layerOrder = revealOrder;

        traceSpecs.push({
          kind: "primary",
          revealOrder: layerOrder,
          meshVertices: entry.mesh.vertices,
          targetScale: classScale,
          initialScale: classInitialScale,
          targetOpacity: classOpacity,
          initialOpacity: animateBuild ? 0.02 : classOpacity,
          trace: buildMeshTrace(entry.mesh, {
            color: entry.color || "#ffb347",
            opacity: classOpacity,
            initialOpacity: animateBuild ? 0.02 : classOpacity,
            name: entry.name || `Class ${entry.label}`,
            scale: classInitialScale,
          }),
        });

        if (animateBuild) {
          const glowScale = classScale * 1.03;
          const glowInitialScale = classScale * 0.94;
          traceSpecs.push({
            kind: "glow",
            revealOrder: layerOrder,
            meshVertices: entry.mesh.vertices,
            initialScale: glowInitialScale,
            peakScale: classScale * 1.09,
            targetScale: glowScale,
            initialOpacity: 0,
            peakOpacity: 0.4,
            trace: buildMeshTrace(entry.mesh, {
              color: entry.color || "#ffb347",
              opacity: 0,
              initialOpacity: 0,
              name: `${entry.name || `Class ${entry.label}`} Glow`,
              scale: glowInitialScale,
              showLegend: false,
            }),
          });
        }

        revealOrder += 1;
      });
    } else if (hasTumor) {
      const tumorOpacity = 0.82;
      const tumorScale = 0.92;
      const tumorInitialScale = animateBuild ? tumorScale * 0.84 : tumorScale;

      traceSpecs.push({
        kind: "primary",
        revealOrder,
        meshVertices: tumorMesh.vertices,
        targetScale: tumorScale,
        initialScale: tumorInitialScale,
        targetOpacity: tumorOpacity,
        initialOpacity: animateBuild ? 0.02 : tumorOpacity,
        trace: buildMeshTrace(tumorMesh, {
          color: "#ffb347",
          opacity: tumorOpacity,
          initialOpacity: animateBuild ? 0.02 : tumorOpacity,
          name: "Tumor",
          scale: tumorInitialScale,
        }),
      });
    }

    const traces = traceSpecs.map((entry) => entry.trace);

    const shape = safeList(inputInfo.volume_shape);
    const spacing = safeList(inputInfo.voxel_spacing_mm)
      .map((x) => fixed(x, 2))
      .join(", ");

    const layout = {
      paper_bgcolor: "#07131d",
      plot_bgcolor: "#07131d",
      margin: { l: 0, r: 0, t: 0, b: 0 },
      uirevision: "mesh-viewer",
      scene: {
        bgcolor: "#07131d",
        dragmode: "orbit",
        camera: {
          eye: { x: 1.55, y: 1.55, z: 1.25 },
          up: { x: 0, y: 0, z: 1 },
        },
        xaxis: {
          title: "",
          showgrid: false,
          zeroline: false,
          showticklabels: false,
          showbackground: false,
        },
        yaxis: {
          title: "",
          showgrid: false,
          zeroline: false,
          showticklabels: false,
          showbackground: false,
        },
        zaxis: {
          title: "",
          showgrid: false,
          zeroline: false,
          showticklabels: false,
          showbackground: false,
        },
        aspectmode: "data",
      },
      legend: {
        orientation: "h",
        yanchor: "top",
        y: 1,
        xanchor: "right",
        x: 1,
        font: { color: "#c6dfef", size: 12 },
        bgcolor: "rgba(6, 19, 29, 0.56)",
        bordercolor: "rgba(139, 184, 212, 0.24)",
        borderwidth: 1,
      },
      annotations: [
        {
          xref: "paper",
          yref: "paper",
          x: 0.01,
          y: 0.99,
          showarrow: false,
          font: { color: "#9ebed5", size: 11 },
          text: `Shape: ${shape.join(" x ")} | Spacing: ${spacing} mm`,
        },
      ],
    };

    const renderPromise = Plotly.react(element, traces, layout, {
      responsive: true,
      displaylogo: false,
      scrollZoom: true,
    });

    if (animateBuild && traceSpecs.length > 0 && window.anime) {
      let cancelled = false;
      let timeline = null;

      Promise.resolve(renderPromise).then(() => {
        if (cancelled) {
          return;
        }
        timeline = animateMeshBuildSequence(element, traceSpecs);
      });

      return {
        pause: () => {
          cancelled = true;
          if (timeline && typeof timeline.pause === "function") {
            timeline.pause();
          }
        },
      };
    }

    return null;
  }

  function App() {
    const viewerRef = useRef(null);
    const viewerAnimationRef = useRef(null);
    const consumedBuildTokenRef = useRef(0);
    const [multimodalFiles, setMultimodalFiles] = useState({
      flair: null,
      t1: null,
      t1ce: null,
      t2: null,
    });
    const [engineMode, setEngineMode] = useState("all");
    const [threshold, setThreshold] = useState("0.50");

    const [inventory, setInventory] = useState(null);
    const [inventoryLoading, setInventoryLoading] = useState(false);
    const [inventoryError, setInventoryError] = useState("");
    const [selectedFolds, setSelectedFolds] = useState(new Set());

    const [running, setRunning] = useState(false);
    const [result, setResult] = useState(null);
    const [meshBuildToken, setMeshBuildToken] = useState(0);
    const [visibleClassLabels, setVisibleClassLabels] = useState(new Set());
    const [status, setStatus] = useState({
      text: "Waiting for input.",
      type: "",
    });

    const foldEntries = useMemo(() => {
      const folds = inventory && inventory.ensemble ? inventory.ensemble.folds : [];
      if (!Array.isArray(folds)) {
        return [];
      }

      return folds.filter((entry) => Number.isInteger(entry.fold_index));
    }, [inventory]);

    const selectedFoldIndices = useMemo(
      () =>
        Array.from(selectedFolds)
          .map((v) => Number(v))
          .filter((v) => Number.isInteger(v) && v >= 0)
          .sort((a, b) => a - b),
      [selectedFolds],
    );

    const deepExists = Boolean(inventory && inventory.deep && inventory.deep.exists);

    const allClassMeshes = useMemo(() => {
      const meshes = result && Array.isArray(result.class_meshes) ? result.class_meshes : [];
      return meshes.filter((entry) => Boolean(entry && entry.mesh && safeList(entry.mesh.vertices).length));
    }, [result]);

    const activeClassMeshes = useMemo(() => {
      if (!allClassMeshes.length) {
        return [];
      }

      return allClassMeshes.filter((entry) => visibleClassLabels.has(String(entry.label)));
    }, [allClassMeshes, visibleClassLabels]);

    const scene = useMemo(() => {
      if (!result) {
        return null;
      }

      return {
        tumorMesh: result.mesh,
        brainMesh: result.brain_mesh,
        classMeshes: activeClassMeshes,
        hasClassDataset: allClassMeshes.length > 0,
        inputInfo: result.input,
      };
    }, [result, activeClassMeshes, allClassMeshes]);

    const metricRows = useMemo(
      () => buildMetricsRows(result ? result.metrics : null, result ? result.inference : null, result ? result.class_metrics : null),
      [result],
    );

    const foldInfoText = useMemo(() => {
      if (inventoryLoading) {
        return "Loading checkpoint inventory...";
      }

      if (inventoryError) {
        return inventoryError;
      }

      if (!foldEntries.length) {
        return `No fold checkpoints found. Deep checkpoint available: ${deepExists ? "yes" : "no"}`;
      }

      return `${selectedFoldIndices.length}/${foldEntries.length} folds selected. Deep checkpoint available: ${deepExists ? "yes" : "no"}`;
    }, [inventoryLoading, inventoryError, foldEntries, selectedFoldIndices, deepExists]);

    const heroMetrics = useMemo(
      () => [
        {
          key: "Engine",
          value: result && result.inference && result.inference.engine ? result.inference.engine : "not run",
        },
        {
          key: "Task",
          value: result && result.inference && result.inference.task ? result.inference.task : "n/a",
        },
        {
          key: "Input",
          value: result && result.inference && result.inference.input_mode
            ? result.inference.input_mode
            : "multimodal",
        },
        {
          key: "Class Meshes",
          value: result && Array.isArray(result.class_meshes) ? String(result.class_meshes.length) : "0",
        },
      ],
      [result],
    );

    useEffect(() => {
      if (!window.anime) {
        document.querySelectorAll(".reveal").forEach((node) => {
          node.style.opacity = "1";
          node.style.transform = "none";
        });
        return undefined;
      }

      const intro = anime({
        targets: ".reveal",
        translateY: [22, 0],
        opacity: [0, 1],
        delay: anime.stagger(85),
        duration: 900,
        easing: "easeOutExpo",
      });

      const orbA = anime({
        targets: ".orb-a",
        translateX: [-20, 18],
        translateY: [-14, 10],
        duration: 6800,
        direction: "alternate",
        loop: true,
        easing: "easeInOutSine",
      });

      const orbB = anime({
        targets: ".orb-b",
        translateX: [24, -18],
        translateY: [16, -12],
        duration: 7400,
        direction: "alternate",
        loop: true,
        easing: "easeInOutSine",
      });

      const orbC = anime({
        targets: ".orb-c",
        translateX: [-14, 20],
        translateY: [18, -10],
        duration: 6200,
        direction: "alternate",
        loop: true,
        easing: "easeInOutSine",
      });

      return () => {
        intro.pause();
        orbA.pause();
        orbB.pause();
        orbC.pause();
      };
    }, []);

    useEffect(() => {
      const hasScene = Boolean(scene && scene.inputInfo);
      const animateBuild = hasScene && meshBuildToken > consumedBuildTokenRef.current;
      if (animateBuild) {
        consumedBuildTokenRef.current = meshBuildToken;
      }

      if (viewerAnimationRef.current && typeof viewerAnimationRef.current.pause === "function") {
        viewerAnimationRef.current.pause();
      }

      const animationHandle = renderViewer(viewerRef.current, scene, {
        animateBuild,
      });
      viewerAnimationRef.current = animationHandle;

      return () => {
        if (animationHandle && typeof animationHandle.pause === "function") {
          animationHandle.pause();
        }
      };
    }, [scene, meshBuildToken]);

    useEffect(() => {
      const onResize = () => {
        if (viewerRef.current && viewerRef.current.data) {
          Plotly.Plots.resize(viewerRef.current);
        }
      };

      window.addEventListener("resize", onResize);
      return () => window.removeEventListener("resize", onResize);
    }, []);

    useEffect(() => {
      const labels = allClassMeshes.map((entry) => String(entry.label));
      setVisibleClassLabels(new Set(labels));
    }, [allClassMeshes]);

    useEffect(() => {
      if (!window.anime) {
        return;
      }

      anime({
        targets: ".metric-card",
        opacity: [0, 1],
        translateY: [10, 0],
        delay: anime.stagger(16),
        duration: 460,
        easing: "easeOutQuad",
      });
    }, [metricRows.length]);

    async function loadCheckpointInventory() {
      setInventoryLoading(true);
      setInventoryError("");

      try {
        const response = await fetch("/api/checkpoints");
        const payload = await response.json();

        if (!response.ok) {
          throw new Error(payload.detail || "Checkpoint inventory request failed.");
        }

        setInventory(payload);
        const foldCandidate = payload && payload.ensemble ? payload.ensemble.folds : [];
        const folds = Array.isArray(foldCandidate) ? foldCandidate : [];
        const available = new Set(
          folds
            .filter((entry) => Number.isInteger(entry.fold_index))
            .map((entry) => String(entry.fold_index)),
        );

        setSelectedFolds((prev) => {
          const kept = Array.from(prev).filter((value) => available.has(value));
          if (kept.length > 0) {
            return new Set(kept);
          }
          return new Set(available);
        });
      } catch (error) {
        setInventory(null);
        setInventoryError(String(error && error.message ? error.message : error));
      } finally {
        setInventoryLoading(false);
      }
    }

    useEffect(() => {
      loadCheckpointInventory();
    }, []);

    function setAllFolds(checked) {
      if (!checked) {
        setSelectedFolds(new Set());
        return;
      }

      const all = new Set(foldEntries.map((entry) => String(entry.fold_index)));
      setSelectedFolds(all);
    }

    function toggleFold(foldKey) {
      setSelectedFolds((prev) => {
        const next = new Set(prev);
        if (next.has(foldKey)) {
          next.delete(foldKey);
        } else {
          next.add(foldKey);
        }
        return next;
      });
    }

    function toggleClassLabel(label) {
      setVisibleClassLabels((prev) => {
        const next = new Set(prev);
        if (next.has(label)) {
          next.delete(label);
        } else {
          next.add(label);
        }
        return next;
      });
    }

    async function runSegmentation() {
      const missingModalities = Object.entries(multimodalFiles)
        .filter(([, selectedFile]) => !selectedFile)
        .map(([name]) => MULTIMODAL_FILE_LABELS[name] || name.toUpperCase());

      if (missingModalities.length > 0) {
        setStatus({
          text: `Please provide all four modality files. Missing: ${missingModalities.join(", ")}.`,
          type: "bad",
        });
        return;
      }

      if (engineMode === "ensemble" && selectedFoldIndices.length === 0) {
        setStatus({
          text: "Select at least one fold checkpoint for ensemble mode.",
          type: "bad",
        });
        return;
      }

      setRunning(true);
      setStatus({ text: "Uploading volume and running segmentation...", type: "" });

      try {
        const formData = new FormData();

        formData.append("flair_file", multimodalFiles.flair);
        formData.append("t1_file", multimodalFiles.t1);
        formData.append("t1ce_file", multimodalFiles.t1ce);
        formData.append("t2_file", multimodalFiles.t2);

        formData.append("engine", engineMode);
        formData.append("threshold", threshold);

        if (selectedFoldIndices.length) {
          formData.append("ensemble_folds", selectedFoldIndices.join(","));
        }

        const response = await fetch("/api/segment", {
          method: "POST",
          body: formData,
        });

        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Segmentation request failed.");
        }

        setResult(payload);
        setMeshBuildToken((value) => value + 1);

        const ensembleSuffix = Number.isFinite(payload.inference && payload.inference.ensemble_size)
          ? ` | Ensemble size: ${payload.inference.ensemble_size}`
          : "";
        const taskSuffix = payload.inference && payload.inference.task
          ? ` | Task: ${payload.inference.task}`
          : "";
        const inputSuffix = payload.inference && payload.inference.input_mode
          ? ` | Input: ${payload.inference.input_mode}`
          : "";
        const classSuffix = Array.isArray(payload.class_meshes) && payload.class_meshes.length
          ? ` | Class meshes: ${payload.class_meshes.length}`
          : "";

        setStatus({
          text: `Done. Engine: ${payload.inference.engine}${ensembleSuffix}${taskSuffix}${inputSuffix}${classSuffix} | Vertices: ${payload.mesh.vertex_count} | Faces: ${payload.mesh.face_count}`,
          type: "good",
        });

        if (window.anime) {
          anime({
            targets: ".status-chip",
            scale: [0.97, 1],
            opacity: [0.7, 1],
            duration: 420,
            easing: "easeOutQuad",
          });
        }
      } catch (error) {
        setStatus({
          text: `Error: ${String(error && error.message ? error.message : error)}`,
          type: "bad",
        });
      } finally {
        setRunning(false);
      }
    }

    return html`
      <main className="app-shell">
        <header className="panel hero reveal">
          <div>
            <p className="eyebrow">BraTS 3D Pipeline</p>
            <h1>NeuroScope React Tumor Studio</h1>
            <p>
              Upload a .nii or .nii.gz MRI volume, run segmentation, inspect a rotatable brain-plus-tumor mesh,
              and compare class-wise volumetrics in one place.
            </p>
          </div>
          <div className="hero-grid">
            ${heroMetrics.map(
              (item) => html`
                <div className="hero-metric" key=${item.key}>
                  <span>${item.key}</span>
                  <strong>${item.value}</strong>
                </div>
              `,
            )}
          </div>
        </header>

        <section className="workspace">
          <aside className="panel controls-panel reveal">
            <h2>Inference Console</h2>

            <div className="field">
              <span className="field-label">BraTS modality files (required)</span>

              ${Object.keys(MULTIMODAL_FILE_LABELS).map((key) =>
                html`
                  <label className="field" key=${key}>
                    <span className="field-label">${MULTIMODAL_FILE_LABELS[key]} file</span>
                    <input
                      className="input"
                      type="file"
                      accept=".nii,.gz,.nii.gz"
                      onChange=${(event) => {
                        const picked = event.target.files && event.target.files[0] ? event.target.files[0] : null;
                        setMultimodalFiles((prev) => ({
                          ...prev,
                          [key]: picked,
                        }));
                      }}
                    />
                  </label>
                `,
              )}

              <div className="file-pill">
                ${Object.entries(MULTIMODAL_FILE_LABELS)
                  .map(([key, label]) => `${label}: ${multimodalFiles[key] ? multimodalFiles[key].name : "missing"}`)
                  .join(" | ")}
              </div>
            </div>

            <label className="field">
              <span className="field-label">Inference engine</span>
              <select
                className="select"
                value=${engineMode}
                onChange=${(event) => setEngineMode(event.target.value)}
              >
                ${ENGINE_OPTIONS.map(
                  (option) => html`
                    <option key=${option.value} value=${option.value}>${option.label}</option>
                  `,
                )}
              </select>
            </label>

            <label className="field">
              <span className="field-label">Mask threshold (deep mode)</span>
              <input
                className="number"
                type="number"
                min="0.05"
                max="0.95"
                step="0.05"
                value=${threshold}
                onChange=${(event) => setThreshold(event.target.value)}
              />
            </label>

            <div className="field">
              <span className="field-label">Fold checkpoint selector (ensemble)</span>
              <div className="fold-toolbar">
                <button
                  type="button"
                  className="ghost-btn"
                  disabled=${inventoryLoading || foldEntries.length === 0}
                  onClick=${() => setAllFolds(true)}
                >
                  Select all
                </button>
                <button
                  type="button"
                  className="ghost-btn"
                  disabled=${inventoryLoading || foldEntries.length === 0}
                  onClick=${() => setAllFolds(false)}
                >
                  Clear all
                </button>
                <button
                  type="button"
                  className="ghost-btn"
                  disabled=${inventoryLoading}
                  onClick=${loadCheckpointInventory}
                >
                  Refresh inventory
                </button>
              </div>

              <div className="fold-cloud">
                ${
                  foldEntries.length === 0
                    ? html`<div className="fold-empty">No fold checkpoints found in models/kfold.</div>`
                    : foldEntries.map((entry) => {
                        const foldKey = String(entry.fold_index);
                        return html`
                          <label className="fold-chip" key=${foldKey} title=${entry.path || ""}>
                            <input
                              type="checkbox"
                              checked=${selectedFolds.has(foldKey)}
                              onChange=${() => toggleFold(foldKey)}
                            />
                            <span>Fold ${entry.fold_index}</span>
                          </label>
                        `;
                      })
                }
              </div>
              <p className="meta-note">${foldInfoText}</p>
            </div>

            <button type="button" className="action-btn" disabled=${running} onClick=${runSegmentation}>
              ${running ? "Segmenting..." : "Run 3D Segmentation"}
            </button>

            <div className=${`status-chip ${status.type || ""}`}>${status.text}</div>
          </aside>

          <div className="stack">
            <section className="panel viewer-panel reveal">
              <div className="panel-head">
                <h2>3D Brain + Tumor/Class Mesh</h2>
                <div className="panel-pill">
                  ${result && result.inference && result.inference.task
                    ? `Task: ${result.inference.task}`
                    : "Task: n/a"}
                </div>
              </div>

              <div className="viewer-frame" ref=${viewerRef}></div>

              <div className="mesh-control-wrap">
                <h3>Class Mesh Visibility</h3>
                <div className="class-toggle-row">
                  ${
                    allClassMeshes.length === 0
                      ? html`<div className="mesh-empty">Run multiclass segmentation to enable class mesh toggles.</div>`
                      : allClassMeshes.map((entry) => {
                          const label = String(entry.label);
                          return html`
                            <label className="class-chip" key=${label}>
                              <input
                                type="checkbox"
                                checked=${visibleClassLabels.has(label)}
                                onChange=${() => toggleClassLabel(label)}
                              />
                              <span
                                className="class-swatch"
                                style=${{ "--swatch-color": entry.color || "#f59e0b" }}
                              ></span>
                              <span>${entry.name || `Class ${label}`}</span>
                            </label>
                          `;
                        })
                  }
                </div>
              </div>
            </section>

            <section className="panel metrics-panel reveal">
              <div className="panel-head">
                <h2>Quantitative Metrics</h2>
                <div className="panel-pill">
                  ${result && result.inference && result.inference.engine
                    ? `Engine: ${result.inference.engine}`
                    : "Engine: n/a"}
                </div>
              </div>

              <div className="metrics-grid">
                ${
                  metricRows.length === 0
                    ? html`<div className="metric-empty">Run segmentation to view volumetric and class-wise metrics.</div>`
                    : metricRows.map((row, index) =>
                        html`
                          <article className="metric-card" key=${`${row.key}-${index}`}>
                            <span className="metric-key">${row.key}</span>
                            <strong className=${`metric-value ${row.tone || ""}`}>${row.value}</strong>
                          </article>
                        `,
                      )
                }
              </div>
            </section>
          </div>
        </section>
      </main>
    `;
  }

  ReactDOM.createRoot(document.getElementById("root")).render(html`<${App} />`);
})();