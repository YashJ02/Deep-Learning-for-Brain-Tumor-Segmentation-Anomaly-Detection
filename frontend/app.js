// -----yash jain------
(function bootstrapNeuroScopePrime() {
  const { useEffect, useMemo, useRef, useState } = React;
  const html = htm.bind(React.createElement);

  const R3F = window.ReactThreeFiber || null;
  const THREE = window.THREE || null;

  const MODALITY_LABELS = {
    flair: "FLAIR",
    t1: "T1",
    t1ce: "T1ce",
    t2: "T2",
  };

  const ENGINE_OPTIONS = [
    { value: "all", label: "All engines (ensemble > deep > baseline)" },
    { value: "auto", label: "Auto (ensemble > deep > baseline)" },
    { value: "deep", label: "Deep model only" },
    { value: "ensemble", label: "Ensemble only" },
    { value: "baseline", label: "Baseline only" },
  ];

  const REPORT_TONES = [
    { value: "executive", label: "Executive" },
    { value: "clinical", label: "Clinical" },
    { value: "technical", label: "Technical" },
  ];

  const CLASS_ORDER = ["1", "2", "4"];
  const CLASS_COLORS = {
    "1": "#f97316",
    "2": "#22c55e",
    "4": "#ef4444",
  };

  function safeArray(value) {
    return Array.isArray(value) ? value : [];
  }

  function toNumber(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function fixed(value, digits = 2) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      return "N/A";
    }
    return parsed.toFixed(digits);
  }

  function clamp(value, minValue, maxValue) {
    return Math.min(maxValue, Math.max(minValue, value));
  }

  function hasMesh(mesh) {
    return Boolean(
      mesh
      && Array.isArray(mesh.vertices)
      && Array.isArray(mesh.faces)
      && mesh.vertices.length > 0
      && mesh.faces.length > 0,
    );
  }

  function basename(pathLike) {
    if (!pathLike) {
      return "";
    }
    const split = String(pathLike).split(/[/\\]+/g);
    return split[split.length - 1] || "";
  }

  function caseIdFromInput(result, selectedFiles) {
    const fromPayload = result && result.input && result.input.source_files
      ? basename(result.input.source_files.flair)
      : "";
    const fromSelection = selectedFiles && selectedFiles.flair ? selectedFiles.flair.name : "";

    const raw = fromPayload || fromSelection;
    if (!raw) {
      return "unknown_case";
    }

    return raw
      .replace(/_(flair|t1ce|t1|t2)\.nii(\.gz)?$/i, "")
      .replace(/\.nii(\.gz)?$/i, "")
      .trim() || "unknown_case";
  }

  function timestampLabel() {
    if (window.dayjs) {
      return window.dayjs().format("YYYY-MM-DD_HH-mm-ss");
    }

    const now = new Date();
    return now.toISOString().replace(/[:.]/g, "-");
  }

  function toneSummaryPrefix(tone) {
    if (tone === "clinical") {
      return "Clinical interpretation";
    }
    if (tone === "technical") {
      return "Technical interpretation";
    }
    return "Executive interpretation";
  }

  function burdenBand(volumeMl) {
    const v = toNumber(volumeMl, -1);
    if (v < 0) {
      return "Unknown";
    }
    if (v < 5) {
      return "Low";
    }
    if (v < 20) {
      return "Moderate";
    }
    return "High";
  }

  function classRowsFromResult(result) {
    const classMetrics = result && result.class_metrics ? result.class_metrics : {};

    return CLASS_ORDER
      .map((label) => {
        const entry = classMetrics[label];
        if (!entry) {
          return null;
        }

        return {
          label,
          name: String(entry.name || `Class ${label}`),
          detected: Boolean(entry.detected),
          voxelCount: toNumber(entry.voxel_count, 0),
          volumeMl: Number.isFinite(Number(entry.volume_ml)) ? Number(entry.volume_ml) : null,
          color: CLASS_COLORS[label] || "#f59e0b",
        };
      })
      .filter(Boolean);
  }

  function metricCardsFromResult(result) {
    if (!result) {
      return [];
    }

    const metrics = result.metrics || {};
    const inference = result.inference || {};
    const mesh = result.mesh || {};

    return [
      {
        key: "Detection",
        value: metrics.detected ? "Positive" : "Not detected",
        tone: metrics.detected ? "good" : "bad",
      },
      {
        key: "Tumor volume",
        value: `${fixed(metrics.volume_ml, 3)} mL`,
      },
      {
        key: "Equivalent diameter",
        value: `${fixed(metrics.equivalent_diameter_mm, 2)} mm`,
      },
      {
        key: "Occupancy",
        value: `${fixed(metrics.occupancy_percent, 4)} %`,
      },
      {
        key: "Engine",
        value: String(inference.engine || "N/A"),
      },
      {
        key: "Ensemble size",
        value: Number.isFinite(inference.ensemble_size) ? String(inference.ensemble_size) : "N/A",
      },
      {
        key: "Confidence mean",
        value: Number.isFinite(inference.probability_mean) ? fixed(inference.probability_mean, 4) : "N/A",
      },
      {
        key: "Tumor mesh",
        value: `${toNumber(mesh.vertex_count, 0)} vertices`,
      },
    ];
  }

  function composeReport(result, context) {
    if (!result) {
      return null;
    }

    const metrics = result.metrics || {};
    const inference = result.inference || {};
    const input = result.input || {};
    const mesh = result.mesh || {};
    const brainMesh = result.brain_mesh || {};

    const classRows = classRowsFromResult(result);
    const dominantClass = classRows
      .filter((row) => row.volumeMl !== null)
      .sort((a, b) => Number(b.volumeMl) - Number(a.volumeMl))[0] || null;

    const caseId = caseIdFromInput(result, context.selectedFiles);
    const generatedAtIso = window.dayjs ? window.dayjs().toISOString() : new Date().toISOString();
    const generatedAtDisplay = window.dayjs
      ? window.dayjs().format("YYYY-MM-DD HH:mm:ss")
      : new Date().toLocaleString();

    const tone = String(context.reportTone || "executive");
    const summary = `${toneSummaryPrefix(tone)} for ${caseId}: `
      + `${metrics.detected ? "Tumor regions detected." : "No tumor region detected."} `
      + `Estimated volume ${fixed(metrics.volume_ml, 3)} mL, equivalent diameter ${fixed(metrics.equivalent_diameter_mm, 2)} mm, occupancy ${fixed(metrics.occupancy_percent, 4)}%. `
      + `${dominantClass ? `Dominant class: ${dominantClass.name} (${fixed(dominantClass.volumeMl, 3)} mL).` : "Dominant class not available."}`;

    const findings = [
      `Burden profile: ${burdenBand(metrics.volume_ml)} by volume estimate.`,
      `Inference engine: ${String(inference.engine || "unknown")}${Number.isFinite(inference.ensemble_size) ? ` (ensemble size ${inference.ensemble_size})` : ""}.`,
      `Voxel count: ${toNumber(metrics.voxel_count, 0)} with extent [${safeArray(metrics.extent_mm).map((item) => fixed(item, 2)).join(", ")}] mm.`,
      `Tumor mesh complexity: ${toNumber(mesh.vertex_count, 0)} vertices / ${toNumber(mesh.face_count, 0)} faces.`,
      `Brain mesh complexity: ${toNumber(brainMesh.vertex_count, 0)} vertices / ${toNumber(brainMesh.face_count, 0)} faces.`,
    ];

    if (context.reportNotes && context.reportNotes.trim()) {
      findings.push(`Analyst note: ${context.reportNotes.trim()}`);
    }

    return {
      report_title: "NeuroScope Prime Segmentation Report",
      generated_at_iso: generatedAtIso,
      generated_at_local: generatedAtDisplay,
      tone,
      case_id: caseId,
      executive_summary: summary,
      findings,
      source_files: {
        flair: input.source_files ? input.source_files.flair : null,
        t1: input.source_files ? input.source_files.t1 : null,
        t1ce: input.source_files ? input.source_files.t1ce : null,
        t2: input.source_files ? input.source_files.t2 : null,
      },
      request: {
        engine_requested: String(input.engine_requested || "unknown"),
        threshold: toNumber(context.thresholdValue, toNumber(input.threshold, 0.5)),
        ensemble_folds_requested: safeArray(input.ensemble_folds_requested),
      },
      inference: {
        engine: String(inference.engine || "unknown"),
        task: String(inference.task || "unknown"),
        input_mode: String(inference.input_mode || "multimodal"),
        ensemble_size: Number.isFinite(inference.ensemble_size) ? inference.ensemble_size : null,
        fold_indices: safeArray(inference.fold_indices),
        checkpoint: inference.checkpoint || null,
        checkpoints: Array.isArray(inference.checkpoints) ? inference.checkpoints : null,
        probability_mean: Number.isFinite(inference.probability_mean) ? Number(inference.probability_mean) : null,
        probability_max: Number.isFinite(inference.probability_max) ? Number(inference.probability_max) : null,
      },
      metrics: {
        detected: Boolean(metrics.detected),
        voxel_count: toNumber(metrics.voxel_count, 0),
        occupancy_percent: Number.isFinite(Number(metrics.occupancy_percent)) ? Number(metrics.occupancy_percent) : null,
        volume_mm3: Number.isFinite(Number(metrics.volume_mm3)) ? Number(metrics.volume_mm3) : null,
        volume_ml: Number.isFinite(Number(metrics.volume_ml)) ? Number(metrics.volume_ml) : null,
        equivalent_diameter_mm: Number.isFinite(Number(metrics.equivalent_diameter_mm)) ? Number(metrics.equivalent_diameter_mm) : null,
        centroid_mm: safeArray(metrics.centroid_mm),
        extent_mm: safeArray(metrics.extent_mm),
      },
      class_breakdown: classRows,
      mesh: {
        tumor_vertex_count: toNumber(mesh.vertex_count, 0),
        tumor_face_count: toNumber(mesh.face_count, 0),
        brain_vertex_count: toNumber(brainMesh.vertex_count, 0),
        brain_face_count: toNumber(brainMesh.face_count, 0),
      },
      notes: context.reportNotes || "",
    };
  }

  function reportToMarkdown(report) {
    if (!report) {
      return "Run segmentation to generate report preview.";
    }

    const classLines = safeArray(report.class_breakdown)
      .map((row) => `| ${row.label} | ${row.name} | ${row.detected ? "yes" : "no"} | ${row.voxelCount} | ${row.volumeMl === null ? "N/A" : fixed(row.volumeMl, 3)} |`)
      .join("\n");

    return [
      "# NeuroScope Prime Segmentation Report",
      "",
      `- Generated: ${report.generated_at_local}`,
      `- Case: ${report.case_id}`,
      `- Tone: ${report.tone}`,
      "",
      "## Executive Summary",
      "",
      report.executive_summary,
      "",
      "## Findings",
      "",
      ...safeArray(report.findings).map((line) => `- ${line}`),
      "",
      "## Inference",
      "",
      `- Engine: ${report.inference.engine}`,
      `- Task: ${report.inference.task}`,
      `- Input mode: ${report.inference.input_mode}`,
      `- Ensemble size: ${report.inference.ensemble_size === null ? "N/A" : report.inference.ensemble_size}`,
      `- Folds used: ${safeArray(report.inference.fold_indices).length ? safeArray(report.inference.fold_indices).join(", ") : "N/A"}`,
      "",
      "## Core Metrics",
      "",
      `- Detected: ${report.metrics.detected ? "yes" : "no"}`,
      `- Tumor volume (mL): ${report.metrics.volume_ml === null ? "N/A" : fixed(report.metrics.volume_ml, 3)}`,
      `- Equivalent diameter (mm): ${report.metrics.equivalent_diameter_mm === null ? "N/A" : fixed(report.metrics.equivalent_diameter_mm, 2)}`,
      `- Occupancy (%): ${report.metrics.occupancy_percent === null ? "N/A" : fixed(report.metrics.occupancy_percent, 4)}`,
      "",
      "## Class Breakdown",
      "",
      "| Label | Name | Detected | Voxels | Volume (mL) |",
      "| --- | --- | --- | ---: | ---: |",
      classLines || "| N/A | N/A | N/A | 0 | N/A |",
      "",
      "## Notes",
      "",
      report.notes || "No notes provided.",
      "",
    ].join("\n");
  }

  function downloadText(filename, text, mimeType) {
    const blob = new Blob([text], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  function buildMeshTrace(mesh, options) {
    return {
      type: "mesh3d",
      x: mesh.vertices.map((point) => point[0]),
      y: mesh.vertices.map((point) => point[1]),
      z: mesh.vertices.map((point) => point[2]),
      i: mesh.faces.map((face) => face[0]),
      j: mesh.faces.map((face) => face[1]),
      k: mesh.faces.map((face) => face[2]),
      opacity: options.opacity,
      color: options.color,
      name: options.name,
      showlegend: options.showLegend !== false,
      hoverinfo: "skip",
      showscale: false,
      flatshading: false,
      lighting: {
        ambient: 0.62,
        diffuse: 0.86,
        roughness: 0.32,
        specular: 0.42,
      },
      lightposition: {
        x: 100,
        y: 110,
        z: 170,
      },
    };
  }

  function renderPlotlyScene(element, sceneModel) {
    if (!element) {
      return;
    }

    if (!sceneModel || !sceneModel.inputInfo) {
      Plotly.purge(element);
      return;
    }

    const traces = [];
    const targets = [];

    if (sceneModel.showBrain && hasMesh(sceneModel.brainMesh)) {
      traces.push(
        buildMeshTrace(sceneModel.brainMesh, {
          name: "Brain",
          color: "#78b6ff",
          opacity: sceneModel.animate ? 0.02 : sceneModel.brainOpacity,
        }),
      );
      targets.push(sceneModel.brainOpacity);
    }

    const classMeshes = safeArray(sceneModel.classMeshes).filter((entry) => hasMesh(entry.mesh));
    if (sceneModel.showTumor && classMeshes.length > 0) {
      classMeshes.forEach((entry) => {
        traces.push(
          buildMeshTrace(entry.mesh, {
            name: entry.name || `Class ${entry.label}`,
            color: entry.color || "#f59e0b",
            opacity: sceneModel.animate ? 0.02 : sceneModel.tumorOpacity,
          }),
        );
        targets.push(sceneModel.tumorOpacity);
      });
    } else if (sceneModel.showTumor && hasMesh(sceneModel.tumorMesh)) {
      traces.push(
        buildMeshTrace(sceneModel.tumorMesh, {
          name: "Tumor",
          color: "#ffb347",
          opacity: sceneModel.animate ? 0.02 : sceneModel.tumorOpacity,
        }),
      );
      targets.push(sceneModel.tumorOpacity);
    }

    if (!traces.length) {
      Plotly.purge(element);
      return;
    }

    const shapeText = safeArray(sceneModel.inputInfo.volume_shape).join(" x ");
    const spacingText = safeArray(sceneModel.inputInfo.voxel_spacing_mm)
      .map((item) => fixed(item, 2))
      .join(", ");

    const layout = {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      margin: { l: 0, r: 0, t: 0, b: 0 },
      uirevision: "neuroscope-prime-scene",
      legend: {
        orientation: "h",
        x: 1,
        xanchor: "right",
        y: 1,
        yanchor: "top",
        bgcolor: "rgba(10, 18, 28, 0.56)",
        bordercolor: "rgba(179, 203, 231, 0.24)",
        borderwidth: 1,
        font: { color: "#c6d7ef", size: 12 },
      },
      scene: {
        dragmode: "orbit",
        aspectmode: "data",
        bgcolor: "rgba(0,0,0,0)",
        camera: {
          eye: { x: 1.58, y: 1.5, z: 1.2 },
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
      },
      annotations: [
        {
          xref: "paper",
          yref: "paper",
          x: 0.01,
          y: 0.99,
          showarrow: false,
          font: { size: 11, color: "#a6bbd8" },
          text: `Shape: ${shapeText} | Spacing: ${spacingText} mm`,
        },
      ],
    };

    Plotly.react(element, traces, layout, {
      responsive: true,
      displaylogo: false,
      scrollZoom: true,
    }).then(() => {
      if (!sceneModel.animate || !window.anime) {
        return;
      }

      const motion = { t: 0 };
      anime({
        targets: motion,
        t: 1,
        duration: 1180,
        easing: "easeOutQuart",
        update: () => {
          targets.forEach((targetOpacity, index) => {
            const reveal = clamp((motion.t * 1.26) - (index * 0.16), 0, 1);
            Plotly.restyle(element, { opacity: [targetOpacity * reveal] }, [index]);
          });
        },
      });
    });
  }

  function HeroR3FExperience() {
    const Canvas = R3F && R3F.Canvas ? R3F.Canvas : null;
    const useFrame = R3F && R3F.useFrame ? R3F.useFrame : null;

    if (!Canvas || !useFrame || !THREE) {
      return html`
        <div className="empty-chip">
          R3F module is unavailable in the current runtime. Plotly clinical viewer remains fully active.
        </div>
      `;
    }

    function PulseOrb() {
      const orbRef = useRef(null);
      const ringARef = useRef(null);
      const ringBRef = useRef(null);

      useFrame((state, delta) => {
        if (!orbRef.current || !ringARef.current || !ringBRef.current) {
          return;
        }

        const t = state.clock.getElapsedTime();

        orbRef.current.rotation.y += delta * 0.38;
        orbRef.current.rotation.x += delta * 0.13;

        ringARef.current.rotation.z += delta * 0.54;
        ringARef.current.rotation.y = Math.sin(t * 0.8) * 0.65;

        ringBRef.current.rotation.x += delta * 0.42;
        ringBRef.current.rotation.z = Math.cos(t * 0.66) * 0.58;
      });

      return html`
        <group>
          <ambientLight intensity=${0.62} />
          <directionalLight position=${[2.8, 2.4, 3.7]} intensity=${1.18} color=${"#8dd9ff"} />
          <pointLight position=${[-2.8, -1.5, 2.8]} intensity=${0.9} color=${"#ffc375"} />

          <mesh ref=${orbRef}>
            <icosahedronGeometry args=${[1.16, 6]} />
            <meshPhysicalMaterial
              color=${"#5ec5ff"}
              roughness=${0.23}
              metalness=${0.18}
              clearcoat=${1}
              clearcoatRoughness=${0.28}
              transmission=${0.22}
              thickness=${1.8}
              emissive=${"#133046"}
              emissiveIntensity=${0.48}
            />
          </mesh>

          <mesh ref=${ringARef} rotation=${[Math.PI * 0.5, 0, 0]}>
            <torusGeometry args=${[1.9, 0.05, 28, 280]} />
            <meshStandardMaterial color=${"#ffb66a"} emissive=${"#7d4f1d"} emissiveIntensity=${0.62} />
          </mesh>

          <mesh ref=${ringBRef} rotation=${[Math.PI * 0.25, Math.PI * 0.16, Math.PI * 0.2]}>
            <torusGeometry args=${[2.18, 0.028, 22, 280]} />
            <meshStandardMaterial color=${"#66ffc4"} emissive=${"#1f4f3f"} emissiveIntensity=${0.56} />
          </mesh>
        </group>
      `;
    }

    return html`
      <div className="hero-canvas-shell">
        <${Canvas}
          camera=${{ position: [0, 0, 4.45], fov: 43 }}
          gl=${{ antialias: true, alpha: true }}
          dpr=${[1, 1.8]}
        >
          <${PulseOrb} />
        </${Canvas}>
      </div>
    `;
  }

  function App() {
    const viewerRef = useRef(null);
    const reportCardRef = useRef(null);
    const consumedSceneTokenRef = useRef(0);

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
    const [sceneToken, setSceneToken] = useState(0);

    const [showBrain, setShowBrain] = useState(true);
    const [showTumor, setShowTumor] = useState(true);
    const [brainOpacity, setBrainOpacity] = useState(0.34);
    const [tumorOpacity, setTumorOpacity] = useState(0.83);

    const [visibleClassLabels, setVisibleClassLabels] = useState(new Set());

    const [reportTone, setReportTone] = useState("executive");
    const [reportNotes, setReportNotes] = useState("");
    const [exportingPdf, setExportingPdf] = useState(false);

    const [status, setStatus] = useState({
      text: "Ready. Upload all four MRI modalities to begin.",
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
      () => Array.from(selectedFolds)
        .map((value) => Number(value))
        .filter((value) => Number.isInteger(value) && value >= 0)
        .sort((a, b) => a - b),
      [selectedFolds],
    );

    const deepAvailable = Boolean(inventory && inventory.deep && inventory.deep.exists);

    const classMeshesAll = useMemo(() => {
      const meshes = result && Array.isArray(result.class_meshes) ? result.class_meshes : [];
      return meshes.filter((entry) => Boolean(entry && entry.mesh && hasMesh(entry.mesh)));
    }, [result]);

    const activeClassMeshes = useMemo(() => {
      if (!classMeshesAll.length) {
        return [];
      }

      return classMeshesAll.filter((entry) => visibleClassLabels.has(String(entry.label)));
    }, [classMeshesAll, visibleClassLabels]);

    const sceneModel = useMemo(() => {
      if (!result) {
        return null;
      }

      return {
        inputInfo: result.input,
        brainMesh: result.brain_mesh,
        tumorMesh: result.mesh,
        classMeshes: activeClassMeshes,
        showBrain,
        showTumor,
        brainOpacity,
        tumorOpacity,
        animate: false,
      };
    }, [result, activeClassMeshes, showBrain, showTumor, brainOpacity, tumorOpacity]);

    const metricCards = useMemo(() => metricCardsFromResult(result), [result]);
    const classRows = useMemo(() => classRowsFromResult(result), [result]);

    const reportObject = useMemo(
      () => composeReport(result, {
        reportTone,
        reportNotes,
        thresholdValue: threshold,
        selectedFiles: multimodalFiles,
      }),
      [result, reportTone, reportNotes, threshold, multimodalFiles],
    );

    const markdownPreview = useMemo(() => reportToMarkdown(reportObject), [reportObject]);

    const heroKpis = useMemo(() => {
      const inference = result && result.inference ? result.inference : {};
      const metrics = result && result.metrics ? result.metrics : {};

      return [
        {
          key: "Engine",
          value: inference.engine || "not run",
        },
        {
          key: "Task",
          value: inference.task || "n/a",
        },
        {
          key: "Tumor Volume",
          value: result ? `${fixed(metrics.volume_ml, 3)} mL` : "n/a",
        },
        {
          key: "Risk Band",
          value: result ? burdenBand(metrics.volume_ml) : "n/a",
        },
      ];
    }, [result]);

    useEffect(() => {
      if (!window.anime) {
        document.querySelectorAll(".reveal").forEach((node) => {
          node.style.opacity = "1";
          node.style.transform = "none";
        });
        return;
      }

      const intro = anime({
        targets: ".reveal",
        translateY: [20, 0],
        opacity: [0, 1],
        delay: anime.stagger(60),
        duration: 840,
        easing: "easeOutExpo",
      });

      const driftA = anime({
        targets: ".prime-radial-a",
        translateX: [-20, 16],
        translateY: [-16, 14],
        direction: "alternate",
        loop: true,
        duration: 7200,
        easing: "easeInOutSine",
      });

      const driftB = anime({
        targets: ".prime-radial-b",
        translateX: [18, -14],
        translateY: [16, -12],
        direction: "alternate",
        loop: true,
        duration: 7600,
        easing: "easeInOutSine",
      });

      return () => {
        intro.pause();
        driftA.pause();
        driftB.pause();
      };
    }, []);

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
      const labels = classMeshesAll.map((entry) => String(entry.label));
      setVisibleClassLabels(new Set(labels));
    }, [classMeshesAll]);

    useEffect(() => {
      const shouldAnimate = sceneToken > consumedSceneTokenRef.current;
      if (shouldAnimate) {
        consumedSceneTokenRef.current = sceneToken;
      }

      const effectiveModel = sceneModel
        ? {
            ...sceneModel,
            animate: shouldAnimate,
          }
        : null;

      renderPlotlyScene(viewerRef.current, effectiveModel);
    }, [sceneModel, sceneToken]);

    useEffect(() => {
      if (!window.lucide || typeof window.lucide.createIcons !== "function") {
        return;
      }
      window.lucide.createIcons();
    });

    useEffect(() => {
      refreshInventory();
    }, []);

    async function refreshInventory() {
      setInventoryLoading(true);
      setInventoryError("");

      try {
        const response = await fetch("/api/checkpoints");
        const payload = await response.json();

        if (!response.ok) {
          throw new Error(payload.detail || "Checkpoint inventory request failed.");
        }

        setInventory(payload);

        const availableFoldKeys = new Set(
          safeArray(payload.ensemble && payload.ensemble.folds)
            .filter((entry) => Number.isInteger(entry.fold_index))
            .map((entry) => String(entry.fold_index)),
        );

        setSelectedFolds((prev) => {
          const retained = Array.from(prev).filter((key) => availableFoldKeys.has(key));
          if (retained.length > 0) {
            return new Set(retained);
          }
          return new Set(availableFoldKeys);
        });
      } catch (error) {
        setInventory(null);
        setInventoryError(String(error && error.message ? error.message : error));
      } finally {
        setInventoryLoading(false);
      }
    }

    function setAllFolds(checked) {
      if (!checked) {
        setSelectedFolds(new Set());
        return;
      }

      setSelectedFolds(new Set(foldEntries.map((entry) => String(entry.fold_index))));
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

    function toggleClass(label) {
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
      const missing = Object.entries(multimodalFiles)
        .filter(([, file]) => !file)
        .map(([key]) => MODALITY_LABELS[key] || key.toUpperCase());

      if (missing.length > 0) {
        setStatus({
          text: `Missing required modalities: ${missing.join(", ")}.`,
          type: "bad",
        });
        return;
      }

      const thresholdNumeric = Number(threshold);
      if (!Number.isFinite(thresholdNumeric) || thresholdNumeric <= 0 || thresholdNumeric >= 1) {
        setStatus({ text: "Threshold must be between 0 and 1 (exclusive).", type: "bad" });
        return;
      }

      if (engineMode === "ensemble" && selectedFoldIndices.length === 0) {
        setStatus({ text: "Select at least one fold for ensemble mode.", type: "bad" });
        return;
      }

      setRunning(true);
      setStatus({ text: "Uploading modalities and running inference...", type: "" });

      try {
        const formData = new FormData();
        formData.append("flair_file", multimodalFiles.flair);
        formData.append("t1_file", multimodalFiles.t1);
        formData.append("t1ce_file", multimodalFiles.t1ce);
        formData.append("t2_file", multimodalFiles.t2);
        formData.append("engine", engineMode);
        formData.append("threshold", String(thresholdNumeric));

        if (selectedFoldIndices.length > 0) {
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
        setSceneToken((token) => token + 1);

        const inference = payload.inference || {};
        const suffix = Number.isFinite(inference.ensemble_size)
          ? ` | ensemble ${inference.ensemble_size}`
          : "";

        setStatus({
          text: `Inference complete with ${inference.engine || "unknown"}${suffix}. Tumor mesh: ${toNumber(payload.mesh && payload.mesh.vertex_count, 0)} vertices.`,
          type: "good",
        });

        if (window.anime) {
          anime({
            targets: ".status",
            scale: [0.97, 1],
            opacity: [0.75, 1],
            duration: 340,
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

    async function exportScenePng() {
      if (!viewerRef.current || !viewerRef.current.data) {
        setStatus({ text: "Render a scene before exporting image.", type: "bad" });
        return;
      }

      try {
        const imageData = await Plotly.toImage(viewerRef.current, {
          format: "png",
          width: 1800,
          height: 1100,
          scale: 1,
        });

        const anchor = document.createElement("a");
        anchor.href = imageData;
        anchor.download = `scene_${timestampLabel()}.png`;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
      } catch (error) {
        setStatus({
          text: `Scene export failed: ${String(error && error.message ? error.message : error)}`,
          type: "bad",
        });
      }
    }

    async function exportReportPdf() {
      if (!reportObject) {
        setStatus({ text: "Run segmentation before exporting report PDF.", type: "bad" });
        return;
      }

      if (!reportCardRef.current) {
        setStatus({ text: "Report panel is not available for PDF export.", type: "bad" });
        return;
      }

      if (!window.html2canvas || !window.jspdf || !window.jspdf.jsPDF) {
        setStatus({
          text: "PDF libraries are unavailable in this runtime.",
          type: "bad",
        });
        return;
      }

      setExportingPdf(true);
      try {
        const canvas = await window.html2canvas(reportCardRef.current, {
          backgroundColor: "#0b111a",
          scale: 2,
          useCORS: true,
        });

        const pdf = new window.jspdf.jsPDF({
          orientation: "p",
          unit: "pt",
          format: "a4",
        });

        const pageWidth = pdf.internal.pageSize.getWidth();
        const pageHeight = pdf.internal.pageSize.getHeight();

        const imageWidth = canvas.width;
        const imageHeight = canvas.height;
        const fit = Math.min(pageWidth / imageWidth, (pageHeight - 40) / imageHeight);

        const outWidth = imageWidth * fit;
        const outHeight = imageHeight * fit;

        const x = (pageWidth - outWidth) / 2;
        const y = 20;

        pdf.addImage(canvas.toDataURL("image/png"), "PNG", x, y, outWidth, outHeight, undefined, "FAST");
        pdf.save(`${reportObject.case_id}_${timestampLabel()}.pdf`);
      } catch (error) {
        setStatus({
          text: `PDF export failed: ${String(error && error.message ? error.message : error)}`,
          type: "bad",
        });
      } finally {
        setExportingPdf(false);
      }
    }

    async function copyExecutiveSummary() {
      if (!reportObject) {
        setStatus({ text: "Run segmentation before copying summary.", type: "bad" });
        return;
      }

      const text = reportObject.executive_summary || "";
      try {
        await navigator.clipboard.writeText(text);
        setStatus({ text: "Executive summary copied to clipboard.", type: "good" });
      } catch (error) {
        setStatus({ text: "Clipboard access failed in this browser context.", type: "bad" });
      }
    }

    const foldInfoText = useMemo(() => {
      if (inventoryLoading) {
        return "Loading checkpoint inventory...";
      }
      if (inventoryError) {
        return inventoryError;
      }
      if (!foldEntries.length) {
        return `No fold checkpoints found. Deep checkpoint available: ${deepAvailable ? "yes" : "no"}`;
      }
      return `${selectedFoldIndices.length}/${foldEntries.length} folds selected. Deep checkpoint available: ${deepAvailable ? "yes" : "no"}`;
    }, [inventoryLoading, inventoryError, foldEntries, selectedFoldIndices, deepAvailable]);

    return html`
      <main className="prime-shell">
        <header className="prime-card prime-hero reveal">
          <div>
            <p className="kicker">NeuroScope Prime</p>
            <h1>Clinical 3D Intelligence Studio</h1>
            <p>
              A premium, React-first workspace for multimodal BraTS segmentation, cinematic 3D inspection,
              and decision-grade report generation in one workflow.
            </p>
          </div>

          <div>
            <div className="hero-kpis">
              ${heroKpis.map((item) => html`
                <article className="kpi" key=${item.key}>
                  <span>${item.key}</span>
                  <strong>${item.value}</strong>
                </article>
              `)}
            </div>

            <div className="control-block" style=${{ marginTop: "10px" }}>
              <span className="control-label">R3F Experience Module</span>
              <${HeroR3FExperience} />
            </div>
          </div>
        </header>

        <section className="prime-card viewer-card reveal">
          <div className="card-head">
            <div>
              <h2>3D Brain Stage</h2>
              <p>Interactive scene focused on brain context, tumor anatomy, and class-wise regions.</p>
            </div>
            <div className="badge-row">
              <div className="badge">Engine: ${result && result.inference ? result.inference.engine : "n/a"}</div>
              <div className="badge">Task: ${result && result.inference ? result.inference.task : "n/a"}</div>
              <div className="badge">Input: ${result && result.inference ? result.inference.input_mode : "multimodal"}</div>
            </div>
          </div>

          <div className="viewer-layout">
            <div className="viewer-stage">
              <div ref=${viewerRef}></div>
              ${!result ? html`
                <div className="viewer-overlay">
                  <div>
                    <strong>3D scene ready</strong>
                    Upload four modalities and run segmentation to visualize brain and tumor meshes.
                  </div>
                </div>
              ` : null}
            </div>

            <aside className="scene-controls">
              <h3>Scene Controls</h3>
              <p className="mini-note">
                Tune transparency, toggle anatomical layers, and export high-resolution stage snapshots.
              </p>

              <div className="control-block">
                <label className="toggle-chip">
                  <input
                    type="checkbox"
                    checked=${showBrain}
                    onChange=${(event) => setShowBrain(event.target.checked)}
                  />
                  <span>Show brain context</span>
                </label>
              </div>

              <div className="control-block">
                <label className="toggle-chip">
                  <input
                    type="checkbox"
                    checked=${showTumor}
                    onChange=${(event) => setShowTumor(event.target.checked)}
                  />
                  <span>Show tumor/class layers</span>
                </label>
              </div>

              <div className="control-block">
                <span className="control-label">Brain opacity (${fixed(brainOpacity, 2)})</span>
                <input
                  className="range"
                  type="range"
                  min="0.05"
                  max="0.95"
                  step="0.01"
                  value=${brainOpacity}
                  onChange=${(event) => setBrainOpacity(toNumber(event.target.value, 0.34))}
                />
              </div>

              <div className="control-block">
                <span className="control-label">Tumor opacity (${fixed(tumorOpacity, 2)})</span>
                <input
                  className="range"
                  type="range"
                  min="0.1"
                  max="1"
                  step="0.01"
                  value=${tumorOpacity}
                  onChange=${(event) => setTumorOpacity(toNumber(event.target.value, 0.83))}
                />
              </div>

              <div className="control-block">
                <span className="control-label">Class visibility</span>
                <div className="class-row">
                  ${classMeshesAll.length === 0
                    ? html`<div className="empty-chip">Run multiclass inference to enable class toggles.</div>`
                    : classMeshesAll.map((entry) => {
                        const label = String(entry.label);
                        return html`
                          <label className="class-chip" key=${label}>
                            <input
                              type="checkbox"
                              checked=${visibleClassLabels.has(label)}
                              onChange=${() => toggleClass(label)}
                            />
                            <span className="swatch" style=${{ background: entry.color || "#f59e0b" }}></span>
                            <span>${entry.name || `Class ${label}`}</span>
                          </label>
                        `;
                      })}
                </div>
              </div>

              <button type="button" className="btn" onClick=${exportScenePng}>Export Scene PNG</button>
            </aside>
          </div>
        </section>

        <section className="prime-workbench">
          <aside className="prime-card intake-card reveal">
            <div className="card-head">
              <div>
                <h2>Inference Intake</h2>
                <p>Multimodal upload, checkpoint selection, and segmentation execution.</p>
              </div>
              <div className="badge">${running ? "running" : "idle"}</div>
            </div>

            <div className="field">
              <label>BraTS modalities (.nii / .nii.gz)</label>
              ${Object.keys(MODALITY_LABELS).map((key) => html`
                <div className="file-row" key=${key}>
                  <label className="control-label">${MODALITY_LABELS[key]}</label>
                  <input
                    className="input"
                    type="file"
                    accept=".nii,.nii.gz,.gz"
                    onChange=${(event) => {
                      const selected = event.target.files && event.target.files[0] ? event.target.files[0] : null;
                      setMultimodalFiles((prev) => ({
                        ...prev,
                        [key]: selected,
                      }));
                    }}
                  />
                  <div className="file-pill">
                    ${multimodalFiles[key] ? multimodalFiles[key].name : "missing"}
                  </div>
                </div>
              `)}
            </div>

            <div className="field">
              <label>Inference engine</label>
              <select
                className="select"
                value=${engineMode}
                onChange=${(event) => setEngineMode(event.target.value)}
              >
                ${ENGINE_OPTIONS.map((option) => html`
                  <option key=${option.value} value=${option.value}>${option.label}</option>
                `)}
              </select>
            </div>

            <div className="field">
              <label>Threshold</label>
              <input
                className="number"
                type="number"
                min="0.05"
                max="0.95"
                step="0.01"
                value=${threshold}
                onChange=${(event) => setThreshold(event.target.value)}
              />
            </div>

            <div className="field">
              <label>Ensemble fold selector</label>
              <div className="stack-actions">
                <button type="button" className="btn btn-soft" disabled=${inventoryLoading || foldEntries.length === 0} onClick=${() => setAllFolds(true)}>
                  Select all
                </button>
                <button type="button" className="btn btn-soft" disabled=${inventoryLoading || foldEntries.length === 0} onClick=${() => setAllFolds(false)}>
                  Clear
                </button>
                <button type="button" className="btn btn-soft" disabled=${inventoryLoading} onClick=${refreshInventory}>
                  Refresh
                </button>
              </div>

              <div className="fold-row">
                ${foldEntries.length === 0
                  ? html`<div className="empty-chip">No fold checkpoints discovered.</div>`
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
                    })}
              </div>

              <p className="mini-note">${foldInfoText}</p>
            </div>

            <button type="button" className="btn btn-primary" disabled=${running} onClick=${runSegmentation}>
              ${running ? "Running Segmentation..." : "Run Segmentation"}
            </button>

            <div className=${`status ${status.type || ""}`}>
              ${status.text}
            </div>
          </aside>

          <div>
            <section className="prime-card report-card reveal" ref=${reportCardRef}>
              <div className="card-head">
                <div>
                  <h2>Report Forge</h2>
                  <p>Generate executive, clinical, or technical report packs directly from current inference.</p>
                </div>
                <div className="badge">
                  ${reportObject ? `Case: ${reportObject.case_id}` : "No report yet"}
                </div>
              </div>

              <div className="report-grid">
                <article className="report-stat">
                  <span>Risk band</span>
                  <strong>${result ? burdenBand(result.metrics && result.metrics.volume_ml) : "n/a"}</strong>
                </article>
                <article className="report-stat">
                  <span>Detected</span>
                  <strong>${result && result.metrics && result.metrics.detected ? "Yes" : result ? "No" : "n/a"}</strong>
                </article>
                <article className="report-stat">
                  <span>Volume</span>
                  <strong>${result ? `${fixed(result.metrics && result.metrics.volume_ml, 3)} mL` : "n/a"}</strong>
                </article>
                <article className="report-stat">
                  <span>Scene mesh</span>
                  <strong>${result ? `${toNumber(result.mesh && result.mesh.vertex_count, 0)} vertices` : "n/a"}</strong>
                </article>
              </div>

              <div className="field">
                <label>Report tone</label>
                <select
                  className="select"
                  value=${reportTone}
                  onChange=${(event) => setReportTone(event.target.value)}
                >
                  ${REPORT_TONES.map((option) => html`
                    <option key=${option.value} value=${option.value}>${option.label}</option>
                  `)}
                </select>
              </div>

              <div className="field">
                <label>Analyst notes</label>
                <textarea
                  className="textarea"
                  value=${reportNotes}
                  onInput=${(event) => setReportNotes(event.target.value)}
                  placeholder="Add interpretation notes, recommendations, follow-up items..."
                ></textarea>
              </div>

              <div className="stack-actions">
                <button
                  type="button"
                  className="btn"
                  disabled=${!reportObject}
                  onClick=${() => {
                    if (!reportObject) {
                      return;
                    }
                    downloadText(
                      `${reportObject.case_id}_${timestampLabel()}.json`,
                      `${JSON.stringify(reportObject, null, 2)}\n`,
                      "application/json;charset=utf-8",
                    );
                  }}
                >
                  Export JSON
                </button>

                <button
                  type="button"
                  className="btn"
                  disabled=${!reportObject}
                  onClick=${() => {
                    if (!reportObject) {
                      return;
                    }
                    downloadText(
                      `${reportObject.case_id}_${timestampLabel()}.md`,
                      `${markdownPreview}\n`,
                      "text/markdown;charset=utf-8",
                    );
                  }}
                >
                  Export Markdown
                </button>

                <button
                  type="button"
                  className="btn"
                  disabled=${!reportObject || exportingPdf}
                  onClick=${exportReportPdf}
                >
                  ${exportingPdf ? "Exporting PDF..." : "Export PDF"}
                </button>

                <button
                  type="button"
                  className="btn"
                  disabled=${!reportObject}
                  onClick=${copyExecutiveSummary}
                >
                  Copy Summary
                </button>
              </div>

              <div className="preview-box">
                <pre>${markdownPreview}</pre>
              </div>
            </section>

            <section className="prime-card metrics-card reveal" style=${{ marginTop: "14px" }}>
              <div className="card-head">
                <div>
                  <h2>Quantitative Board</h2>
                  <p>Immediate volumetric, confidence, and class-region metrics.</p>
                </div>
                <div className="badge">
                  ${result && result.inference ? `Engine: ${result.inference.engine}` : "Engine: n/a"}
                </div>
              </div>

              <div className="metrics-grid">
                ${metricCards.length === 0
                  ? html`<div className="empty-chip">Run segmentation to populate the quantitative board.</div>`
                  : metricCards.map((item, index) => html`
                      <article className="metric" key=${`${item.key}-${index}`}>
                        <span>${item.key}</span>
                        <strong className=${item.tone || ""}>${item.value}</strong>
                      </article>
                    `)}
              </div>

              <div className="class-table">
                <table>
                  <thead>
                    <tr>
                      <th>Class</th>
                      <th>Detected</th>
                      <th>Voxels</th>
                      <th>Volume (mL)</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${classRows.length === 0
                      ? html`
                          <tr>
                            <td colSpan="4" style=${{ color: "#95a3bc" }}>
                              No class-wise metrics yet.
                            </td>
                          </tr>
                        `
                      : classRows.map((row) => html`
                          <tr key=${row.label}>
                            <td>
                              <span className="class-name">
                                <span className="dot" style=${{ background: row.color }}></span>
                                <span>${row.name}</span>
                              </span>
                            </td>
                            <td>${row.detected ? "yes" : "no"}</td>
                            <td>${row.voxelCount}</td>
                            <td>${row.volumeMl === null ? "N/A" : fixed(row.volumeMl, 3)}</td>
                          </tr>
                        `)}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        </section>
      </main>
    `;
  }

  ReactDOM.createRoot(document.getElementById("root")).render(html`<${App} />`);
})();
