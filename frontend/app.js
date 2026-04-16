// -----yash jain------
(function bootstrapNeuroScopePrime() {
  const { useEffect, useLayoutEffect, useMemo, useRef, useState } = React;
  const html = htm.bind(React.createElement);

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

  const REPORT_TONE_TEMPLATES = {
    executive: {
      tone: "executive",
      label: "Executive",
      summary_title: "Executive Summary",
      findings_title: "Findings",
      analyst_note_prefix: "Analyst note",
      summary_template: "Executive interpretation for {{case_id}}: {{detection_sentence}} Estimated tumor burden is {{volume_ml}} mL ({{burden_band}}), with equivalent diameter {{equivalent_diameter_mm}} mm and occupancy {{occupancy_percent}}%. {{dominant_class_sentence}}",
      findings_templates: [
        "Burden profile: {{burden_band}} by volume estimate.",
        "Inference engine: {{engine_with_ensemble}}.",
        "Voxel count: {{voxel_count}} with extent [{{extent_mm}}] mm.",
        "Tumor mesh complexity: {{tumor_vertices}} vertices / {{tumor_faces}} faces.",
        "Brain mesh complexity: {{brain_vertices}} vertices / {{brain_faces}} faces.",
      ],
    },
    clinical: {
      tone: "clinical",
      label: "Clinical",
      summary_title: "Clinical Impression",
      findings_title: "Clinical Findings",
      analyst_note_prefix: "Clinical note",
      summary_template: "Clinical interpretation for {{case_id}}: {{detection_sentence_clinical}} Quantified tumor volume is {{volume_ml}} mL with equivalent diameter {{equivalent_diameter_mm}} mm and occupancy {{occupancy_percent}}%. {{dominant_class_sentence}}",
      findings_templates: [
        "Clinical burden category: {{burden_band}}.",
        "Estimated lesion extent (mm): [{{extent_mm}}].",
        "Tumor voxel count: {{voxel_count}}.",
        "Reconstruction proxy: tumor {{tumor_vertices}} vertices / {{tumor_faces}} faces; brain {{brain_vertices}} vertices / {{brain_faces}} faces.",
        "Inference route: {{engine_with_ensemble}}.",
      ],
    },
    technical: {
      tone: "technical",
      label: "Technical",
      summary_title: "Technical Interpretation",
      findings_title: "Technical Findings",
      analyst_note_prefix: "Operator note",
      summary_template: "Technical interpretation for {{case_id}}: {{detection_sentence}} Pipeline used {{engine_with_ensemble}} at threshold {{threshold}}. Volume={{volume_ml}} mL, diameter={{equivalent_diameter_mm}} mm, occupancy={{occupancy_percent}}%.",
      findings_templates: [
        "Confidence statistics: mean={{confidence_mean}}, max={{confidence_max}}.",
        "Voxel occupancy: {{voxel_count}} voxels with extent [{{extent_mm}}] mm.",
        "Mesh topology: tumor {{tumor_vertices}}v/{{tumor_faces}}f, brain {{brain_vertices}}v/{{brain_faces}}f.",
        "Input mode: {{input_mode}}.",
        "Dominant class channel: {{dominant_class_short}}.",
      ],
    },
  };

  const WORKFLOW_STEPS = [
    { value: "login", label: "1. Login" },
    { value: "intake", label: "2. Intake" },
    { value: "generate", label: "3. Generate + Report" },
    { value: "export", label: "4. Export" },
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

  function toneTemplateFor(tone) {
    const key = String(tone || "executive").toLowerCase();
    return REPORT_TONE_TEMPLATES[key] || REPORT_TONE_TEMPLATES.executive;
  }

  function applyToneTemplate(line, values) {
    return String(line || "").replace(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g, (_, token) => {
      const value = values[token];
      return value === null || value === undefined ? "N/A" : String(value);
    });
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

    const template = toneTemplateFor(context.reportTone);
    const tone = template.tone;
    const extentText = safeArray(metrics.extent_mm).map((item) => fixed(item, 2)).join(", ") || "N/A";
    const ensembleSuffix = Number.isFinite(inference.ensemble_size) ? ` (ensemble size ${inference.ensemble_size})` : "";
    const engineWithEnsemble = `${String(inference.engine || "unknown")}${ensembleSuffix}`;
    const dominantClassSentence = dominantClass
      ? `Dominant class: ${dominantClass.name} (${fixed(dominantClass.volumeMl, 3)} mL).`
      : "Dominant class not available.";

    const templateValues = {
      case_id: caseId,
      detection_sentence: metrics.detected ? "Tumor regions detected." : "No tumor region detected.",
      detection_sentence_clinical: metrics.detected
        ? "Tumor regions are detected in the current segmentation run."
        : "No tumor region is detected in the current segmentation run.",
      detection_flag: metrics.detected ? "positive" : "negative",
      volume_ml: fixed(metrics.volume_ml, 3),
      equivalent_diameter_mm: fixed(metrics.equivalent_diameter_mm, 2),
      occupancy_percent: fixed(metrics.occupancy_percent, 4),
      burden_band: burdenBand(metrics.volume_ml),
      dominant_class_sentence: dominantClassSentence,
      dominant_class_short: dominantClass ? `${dominantClass.name} (${fixed(dominantClass.volumeMl, 3)} mL)` : "not available",
      engine_with_ensemble: engineWithEnsemble,
      voxel_count: toNumber(metrics.voxel_count, 0),
      extent_mm: extentText,
      tumor_vertices: toNumber(mesh.vertex_count, 0),
      tumor_faces: toNumber(mesh.face_count, 0),
      brain_vertices: toNumber(brainMesh.vertex_count, 0),
      brain_faces: toNumber(brainMesh.face_count, 0),
      confidence_mean: Number.isFinite(inference.probability_mean) ? fixed(inference.probability_mean, 4) : "N/A",
      confidence_max: Number.isFinite(inference.probability_max) ? fixed(inference.probability_max, 4) : "N/A",
      threshold: fixed(toNumber(context.thresholdValue, toNumber(input.threshold, 0.5)), 2),
      input_mode: String(inference.input_mode || "multimodal"),
    };

    const summary = applyToneTemplate(template.summary_template, templateValues);
    const findings = safeArray(template.findings_templates).map((line) => applyToneTemplate(line, templateValues));

    if (context.reportNotes && context.reportNotes.trim()) {
      findings.push(`${template.analyst_note_prefix}: ${context.reportNotes.trim()}`);
    }

    return {
      report_title: "NeuroScope Prime Segmentation Report",
      generated_at_iso: generatedAtIso,
      generated_at_local: generatedAtDisplay,
      tone,
      tone_label: template.label,
      summary_title: template.summary_title,
      findings_title: template.findings_title,
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
      `## ${report.summary_title || "Executive Summary"}`,
      "",
      report.executive_summary,
      "",
      `## ${report.findings_title || "Findings"}`,
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

  function Loader({ label = "Loading...", size = "md", inline = false }) {
    const sizeClass = size === "sm"
      ? "loader-sm"
      : size === "lg"
        ? "loader-lg"
        : size === "xl"
          ? "loader-xl"
          : "";
    const wrapperClass = inline ? "loader-inline" : "loader-block";

    return html`
      <span className=${wrapperClass}>
        <span className=${`loader ${sizeClass}`.trim()}></span>
        <span>${label}</span>
      </span>
    `;
  }

  function BusyOverlay({ label }) {
    return html`
      <div className="section-loader-overlay" role="status" aria-live="polite" aria-label=${label}>
        <div className="section-loader-card">
          <${Loader} label=${label} size="lg" />
        </div>
      </div>
    `;
  }

  function App() {
    const shellRef = useRef(null);
    const viewerRef = useRef(null);
    const reportCardRef = useRef(null);
    const consumedSceneTokenRef = useRef(0);
    const workflowScaleRef = useRef(1);
    const [activePage, setActivePage] = useState("login");
    const [maxUnlockedPageIndex, setMaxUnlockedPageIndex] = useState(0);

    const [demoLoggedIn, setDemoLoggedIn] = useState(false);
    const [sessionAuthType, setSessionAuthType] = useState("none");
    const [loginUsername, setLoginUsername] = useState("");
    const [loginPassword, setLoginPassword] = useState("");
    const [bootLoading, setBootLoading] = useState(true);
    const [intakeMode, setIntakeMode] = useState("demo");
    const [demoPatients, setDemoPatients] = useState([]);
    const [demoPatientsLoading, setDemoPatientsLoading] = useState(false);
    const [demoPatientsError, setDemoPatientsError] = useState("");
    const [selectedDemoCaseId, setSelectedDemoCaseId] = useState("");

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
      text: "Start with Login or Demo Login, then move through Intake, Generate + Report, and Export.",
      type: "",
    });

    const activePageIndex = useMemo(
      () => WORKFLOW_STEPS.findIndex((step) => step.value === activePage),
      [activePage],
    );

    const selectedDemoPatient = useMemo(
      () => demoPatients.find((item) => item.case_id === selectedDemoCaseId) || null,
      [demoPatients, selectedDemoCaseId],
    );

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
      if (bootLoading) {
        return undefined;
      }

      const revealNodes = document.querySelectorAll(".reveal");
      if (!revealNodes.length) {
        return undefined;
      }

      if (!window.anime) {
        revealNodes.forEach((node) => {
          node.style.opacity = "1";
          node.style.transform = "none";
        });
        return undefined;
      }

      const intro = anime({
        targets: ".reveal",
        translateY: [30, 0],
        scale: [0.95, 1],
        opacity: [0, 1],
        delay: anime.stagger(80),
        duration: 1200,
        easing: "easeOutElastic(1, .8)",
      });

      return () => {
        intro.pause();
      };
    }, [bootLoading]);

    useLayoutEffect(() => {
      const shellNode = shellRef.current;
      if (!shellNode) {
        return undefined;
      }

      let animationFrame = 0;
      let observer = null;

      const fitWorkflowToViewport = () => {
        const currentNode = shellRef.current;
        if (!currentNode) {
          return;
        }

        const baseWidth = Math.max(currentNode.scrollWidth, currentNode.clientWidth);
        const baseHeight = Math.max(currentNode.scrollHeight, currentNode.clientHeight);

        const widthRatio = window.innerWidth / Math.max(baseWidth, 1);
        const heightRatio = window.innerHeight / Math.max(baseHeight, 1);
        const nextScale = clamp(Math.min(widthRatio, heightRatio, 1), 0.5, 1);
        const roundedScale = Number(nextScale.toFixed(4));

        if (Math.abs(roundedScale - workflowScaleRef.current) < 0.001) {
          return;
        }

        workflowScaleRef.current = roundedScale;
        currentNode.style.setProperty("--workflow-scale", String(roundedScale));
      };

      const scheduleFit = () => {
        cancelAnimationFrame(animationFrame);
        animationFrame = requestAnimationFrame(fitWorkflowToViewport);
      };

      shellNode.style.setProperty("--workflow-scale", String(workflowScaleRef.current));
      scheduleFit();

      window.addEventListener("resize", scheduleFit);

      if (window.ResizeObserver) {
        observer = new ResizeObserver(scheduleFit);
        observer.observe(shellNode);
        const frameNode = shellNode.querySelector(".prime-page-frame");
        if (frameNode) {
          observer.observe(frameNode);
        }
      }

      return () => {
        cancelAnimationFrame(animationFrame);
        window.removeEventListener("resize", scheduleFit);
        if (observer) {
          observer.disconnect();
        }
      };
    }, [
      bootLoading,
      activePage,
      intakeMode,
      running,
      demoPatients.length,
      foldEntries.length,
      selectedFoldIndices.length,
      metricCards.length,
      classRows.length,
      markdownPreview.length,
      reportNotes.length,
      status.text,
    ]);

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
      if (activePage !== "generate") {
        return;
      }

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
    }, [sceneModel, sceneToken, activePage]);

    useEffect(() => {
      if (!window.lucide || typeof window.lucide.createIcons !== "function") {
        return;
      }
      window.lucide.createIcons();
    });

    useEffect(() => {
      let cancelled = false;
      const forceReadyTimer = window.setTimeout(() => {
        if (!cancelled) {
          setBootLoading(false);
        }
      }, 8000);

      async function initializeWorkspace() {
        try {
          await Promise.allSettled([refreshInventory(), refreshDemoPatients()]);
        } finally {
          if (!cancelled) {
            setBootLoading(false);
          }
        }
      }

      initializeWorkspace();

      return () => {
        cancelled = true;
        window.clearTimeout(forceReadyTimer);
      };
    }, []);

    async function refreshDemoPatients() {
      setDemoPatientsLoading(true);
      setDemoPatientsError("");

      try {
        const response = await fetch("/api/demo-patients");
        const payload = await response.json();

        if (!response.ok) {
          throw new Error(payload.detail || "Demo patient list request failed.");
        }

        const patients = Array.isArray(payload.patients) ? payload.patients : [];
        setDemoPatients(patients);

        setSelectedDemoCaseId((previous) => {
          if (previous && patients.some((entry) => entry.case_id === previous)) {
            return previous;
          }
          return patients.length ? String(patients[0].case_id) : "";
        });
      } catch (error) {
        setDemoPatients([]);
        setDemoPatientsError(String(error && error.message ? error.message : error));
      } finally {
        setDemoPatientsLoading(false);
      }
    }

    function missingUploadedModalities() {
      return Object.entries(multimodalFiles)
        .filter(([, file]) => !file)
        .map(([key]) => MODALITY_LABELS[key] || key.toUpperCase());
    }

    function activateSession(authType, messageText) {
      setDemoLoggedIn(true);
      setSessionAuthType(authType);
      setMaxUnlockedPageIndex((previous) => Math.max(previous, 1));
      setActivePage("intake");
      setStatus({
        text: messageText,
        type: "good",
      });
    }

    function loginWithCredentials() {
      const username = loginUsername.trim();
      const password = loginPassword;

      if (!username || !password) {
        setStatus({ text: "Enter username and password to use Login.", type: "bad" });
        return;
      }

      activateSession(
        "standard",
        `Login complete for ${username}. Choose demo patient data or upload your own files in Intake.`,
      );
    }

    function loginAsDemo() {
      setLoginUsername("demo");
      setLoginPassword("demo");
      activateSession(
        "demo",
        "Demo session is active. Choose demo patient data or upload your own files in Intake.",
      );
    }

    function canAdvanceFromCurrentStep() {
      if (activePage === "login") {
        if (!demoLoggedIn) {
          setStatus({ text: "Use Login or Demo Login to unlock the workflow.", type: "bad" });
          return false;
        }
        return true;
      }

      if (activePage === "intake") {
        if (intakeMode === "demo") {
          if (!selectedDemoCaseId) {
            setStatus({ text: "Select one demo patient case before continuing.", type: "bad" });
            return false;
          }
          return true;
        }

        const missing = missingUploadedModalities();
        if (missing.length > 0) {
          setStatus({ text: `Missing required modalities: ${missing.join(", ")}.`, type: "bad" });
          return false;
        }
        return true;
      }

      if (activePage === "generate") {
        if (!result) {
          setStatus({ text: "Run generation first, then continue to Export.", type: "bad" });
          return false;
        }
        return true;
      }

      return true;
    }

    function goToStep(stepValue) {
      const targetIndex = WORKFLOW_STEPS.findIndex((step) => step.value === stepValue);
      if (targetIndex < 0) {
        return;
      }

      if (targetIndex > maxUnlockedPageIndex) {
        setStatus({ text: "Complete previous steps first.", type: "bad" });
        return;
      }

      setActivePage(stepValue);
    }

    function goBackStep() {
      if (activePageIndex <= 0) {
        return;
      }

      const previousStep = WORKFLOW_STEPS[activePageIndex - 1];
      setActivePage(previousStep.value);
    }

    function goNextStep() {
      if (!canAdvanceFromCurrentStep()) {
        return;
      }

      if (activePageIndex >= WORKFLOW_STEPS.length - 1) {
        return;
      }

      const nextIndex = activePageIndex + 1;
      const nextStep = WORKFLOW_STEPS[nextIndex];
      setMaxUnlockedPageIndex((previous) => Math.max(previous, nextIndex));
      setActivePage(nextStep.value);
    }

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
      const usingDemoData = intakeMode === "demo";

      if (usingDemoData && !selectedDemoCaseId) {
        setStatus({ text: "Select a demo patient before running generation.", type: "bad" });
        return;
      }

      if (!usingDemoData) {
        const missing = missingUploadedModalities();
        if (missing.length > 0) {
          setStatus({
            text: `Missing required modalities: ${missing.join(", ")}.`,
            type: "bad",
          });
          return;
        }
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
      setStatus({
        text: usingDemoData
          ? `Loading demo patient ${selectedDemoCaseId} and running inference...`
          : "Uploading modalities and running inference...",
        type: "",
      });

      try {
        const formData = new FormData();
        formData.append("engine", engineMode);
        formData.append("threshold", String(thresholdNumeric));

        if (usingDemoData) {
          formData.append("case_id", selectedDemoCaseId);
        } else {
          formData.append("flair_file", multimodalFiles.flair);
          formData.append("t1_file", multimodalFiles.t1);
          formData.append("t1ce_file", multimodalFiles.t1ce);
          formData.append("t2_file", multimodalFiles.t2);
        }

        if (selectedFoldIndices.length > 0) {
          formData.append("ensemble_folds", selectedFoldIndices.join(","));
        }

        const response = await fetch(usingDemoData ? "/api/segment-demo" : "/api/segment", {
          method: "POST",
          body: formData,
        });

        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Segmentation request failed.");
        }

        setResult(payload);
        setActivePage("generate");
        setMaxUnlockedPageIndex((previous) => Math.max(previous, 2));
        setSceneToken((token) => token + 1);

        const inference = payload.inference || {};
        const suffix = Number.isFinite(inference.ensemble_size)
          ? ` | ensemble ${inference.ensemble_size}`
          : "";

        setStatus({
          text: `Inference complete with ${inference.engine || "unknown"}${suffix}. Tumor mesh: ${toNumber(payload.mesh && payload.mesh.vertex_count, 0)} vertices. Continue to Export when ready.`,
          type: "good",
        });

        if (window.anime) {
          anime({
            targets: ".status",
            scale: [0.9, 1],
            opacity: [0, 1],
            translateY: [10, 0],
            duration: 600,
            easing: "easeOutElastic(1, .8)",
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

    function exportReportJson() {
      if (!reportObject) {
        setStatus({ text: "Run generation before exporting report JSON.", type: "bad" });
        return;
      }

      downloadText(
        `${reportObject.case_id}_${timestampLabel()}.json`,
        `${JSON.stringify(reportObject, null, 2)}\n`,
        "application/json;charset=utf-8",
      );
    }

    function exportReportMarkdown() {
      if (!reportObject) {
        setStatus({ text: "Run generation before exporting report Markdown.", type: "bad" });
        return;
      }

      downloadText(
        `${reportObject.case_id}_${timestampLabel()}.md`,
        `${markdownPreview}\n`,
        "text/markdown;charset=utf-8",
      );
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
      const summaryTitle = reportObject.summary_title || "Summary";
      try {
        await navigator.clipboard.writeText(text);
        setStatus({ text: `${summaryTitle} copied to clipboard.`, type: "good" });
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

    const isFinalStep = activePageIndex >= WORKFLOW_STEPS.length - 1;
    const footerPrimaryLabel = activePage === "intake"
      ? (running ? "Generating..." : "Generate")
      : (activePage === "generate" ? "Continue to Export" : "Next");
    const footerPrimaryDisabled = activePage === "intake" ? running : isFinalStep;
    const footerPrimaryAction = activePage === "intake" ? runSegmentation : goNextStep;

    const footerPrimaryContent = activePage === "intake" && running
      ? html`<span className="btn-loader"><${Loader} label="Generating..." size="sm" inline=${true} /></span>`
      : footerPrimaryLabel;

    if (bootLoading) {
      return html`
        <main ref=${shellRef} className="prime-shell prime-shell-app boot-shell">
          <section className="prime-card boot-card" role="status" aria-live="polite">
            <${Loader} label="Loading NeuroScope Prime" size="xl" />
            <h2>Preparing Workspace</h2>
            <p>Checking checkpoints, demo patients, and visualization services.</p>
          </section>
        </main>
      `;
    }

    return html`
      <main ref=${shellRef} className="prime-shell prime-shell-app">
        <header className="prime-card prime-header reveal">
          <div className="header-brand">
            <p className="kicker">NeuroScope Prime</p>
            <h1>Clinical 3D Intelligence Studio</h1>
            <p>Guided workflow: Login or Demo Login, Intake, Generate + Report, then Export.</p>
          </div>

          <nav className="page-switcher" role="tablist" aria-label="Workspace steps">
            ${WORKFLOW_STEPS.map((page, index) => {
              const unlocked = index <= maxUnlockedPageIndex;
              return html`
                <button
                  key=${page.value}
                  type="button"
                  className=${`page-tab ${activePage === page.value ? "active" : ""} ${unlocked ? "" : "locked"}`}
                  disabled=${!unlocked}
                  onClick=${() => goToStep(page.value)}
                >
                  ${page.label}
                </button>
              `;
            })}
          </nav>

          <div className="hero-kpis">
            ${heroKpis.map((item) => html`
              <article className="kpi" key=${item.key}>
                <span>${item.key}</span>
                <strong>${item.value}</strong>
              </article>
            `)}
          </div>
        </header>

        <section className="prime-page-frame">
          ${activePage === "login" ? html`
            <section className="prime-card login-card page-pane page-pane-narrow workflow-pane workflow-login">
              <div className="login-primary">
                <div className="card-head">
                  <div>
                    <h2>Login</h2>
                    <p>Use Login for standard access or Demo Login for a guided showcase.</p>
                  </div>
                  <div className="badge">
                    ${!demoLoggedIn ? "not logged in" : sessionAuthType === "demo" ? "demo active" : "active"}
                  </div>
                </div>

                <div className="control-block login-auth-block">
                  <span className="control-label">User Login</span>
                  <div className="login-auth-grid">
                    <input
                      className="input"
                      type="text"
                      value=${loginUsername}
                      placeholder="Username or email"
                      onInput=${(event) => setLoginUsername(event.target.value)}
                    />
                    <input
                      className="input"
                      type="password"
                      value=${loginPassword}
                      placeholder="Password"
                      onInput=${(event) => setLoginPassword(event.target.value)}
                      onKeyDown=${(event) => {
                        if (event.key === "Enter") {
                          loginWithCredentials();
                        }
                      }}
                    />
                  </div>

                  <div className="login-actions">
                    <button
                      type="button"
                      className="btn btn-primary btn-inline"
                      onClick=${loginWithCredentials}
                    >
                      ${demoLoggedIn && sessionAuthType === "standard" ? "Logged In" : "Login"}
                    </button>
                    <button
                      type="button"
                      className="btn btn-soft btn-inline"
                      onClick=${loginAsDemo}
                    >
                      ${demoLoggedIn && sessionAuthType === "demo" ? "Demo Session Active" : "Login as Demo"}
                    </button>
                  </div>
                </div>

                <div className="control-block">
                  <span className="control-label">Demo Credentials</span>
                  <div className="badge-row">
                    <div className="badge">username: demo</div>
                    <div className="badge">mode: local research</div>
                  </div>
                </div>

                <div className=${`status ${demoLoggedIn ? "good" : ""}`}>
                  ${demoLoggedIn
                    ? sessionAuthType === "demo"
                      ? "Demo login complete. Continue to Intake and select a patient dataset."
                      : `User login complete for ${loginUsername.trim() || "user"}. Continue to Intake and choose a data source.`
                    : "Use Login or Login as Demo to start."}
                </div>
              </div>

              <div className="login-panels">
                <article className="login-panel">
                  <h3>Workflow Preview</h3>
                  <p className="mini-note">Complete these guided stages after sign-in.</p>
                  <div className="login-step-list">
                    ${WORKFLOW_STEPS.map((step) => html`
                      <div className="login-step" key=${`preview-${step.value}`}>
                        <span>${step.label}</span>
                      </div>
                    `)}
                  </div>
                </article>

                <article className="login-panel">
                  <h3>System Readiness</h3>
                  <p className="mini-note">Quick checks before running generation.</p>
                  <div className="login-readiness-grid">
                    <div className="login-readiness-item">
                      <span>Demo cases</span>
                      <strong>
                        ${demoPatientsLoading
                          ? html`<${Loader} label="Loading" size="sm" inline=${true} />`
                          : String(demoPatients.length)}
                      </strong>
                    </div>
                    <div className="login-readiness-item">
                      <span>Ensemble folds</span>
                      <strong>
                        ${inventoryLoading
                          ? html`<${Loader} label="Loading" size="sm" inline=${true} />`
                          : String(foldEntries.length)}
                      </strong>
                    </div>
                    <div className="login-readiness-item">
                      <span>Deep model</span>
                      <strong>${deepAvailable ? "Ready" : "Missing"}</strong>
                    </div>
                    <div className="login-readiness-item">
                      <span>Session</span>
                      <strong>${demoLoggedIn ? "Active" : "Pending"}</strong>
                    </div>
                  </div>
                </article>
              </div>
            </section>
          ` : null}

          ${activePage === "intake" ? html`
            <section className="prime-card intake-card page-pane page-pane-narrow workflow-pane workflow-intake">
              <div className="card-head">
                <div>
                  <h2>Inference Intake</h2>
                  <p>Choose demo data or upload files, then configure generation options.</p>
                </div>
                <div className="badge">${running ? "running" : "idle"}</div>
              </div>

              <div className="field">
                <label>Data source</label>
                <div className="intake-source-switch">
                  <button
                    type="button"
                    className=${`btn btn-soft ${intakeMode === "demo" ? "active-toggle" : ""}`}
                    onClick=${() => setIntakeMode("demo")}
                  >
                    Demo patient dataset
                  </button>
                  <button
                    type="button"
                    className=${`btn btn-soft ${intakeMode === "upload" ? "active-toggle" : ""}`}
                    onClick=${() => setIntakeMode("upload")}
                  >
                    Manual upload
                  </button>
                </div>
              </div>

              ${intakeMode === "demo" ? html`
                <div className="field">
                  <label>Demo patients (3-4 prepared cases)</label>
                  <div className="stack-actions">
                    <button type="button" className="btn btn-soft" disabled=${demoPatientsLoading} onClick=${refreshDemoPatients}>
                      ${demoPatientsLoading
                        ? html`<span className="btn-loader"><${Loader} label="Refreshing..." size="sm" inline=${true} /></span>`
                        : "Refresh demo list"}
                    </button>
                  </div>

                  ${demoPatientsLoading
                    ? html`<div className="inline-progress"><${Loader} label="Fetching demo patients..." inline=${true} /></div>`
                    : null}

                  ${demoPatientsError
                    ? html`<div className="status bad">${demoPatientsError}</div>`
                    : html`
                        <div className="demo-patient-grid">
                          ${demoPatients.length === 0
                            ? html`<div className="empty-chip">No demo patients found in this workspace data directory.</div>`
                            : demoPatients.map((patient) => html`
                                <button
                                  type="button"
                                  key=${patient.case_id}
                                  className=${`demo-patient-card ${selectedDemoCaseId === patient.case_id ? "active" : ""}`}
                                  onClick=${() => setSelectedDemoCaseId(patient.case_id)}
                                >
                                  <strong>${patient.case_id}</strong>
                                  <span>${patient.source || "local"}</span>
                                </button>
                              `)}
                        </div>
                      `}

                  <p className="mini-note">
                    ${selectedDemoPatient
                      ? `Selected demo case: ${selectedDemoPatient.case_id}`
                      : "Select a demo case to continue."}
                  </p>
                </div>
              ` : null}

              ${intakeMode === "upload" ? html`
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
              ` : null}

              <div className="field-grid">
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
                    ${inventoryLoading
                      ? html`<span className="btn-loader"><${Loader} label="Refreshing..." size="sm" inline=${true} /></span>`
                      : "Refresh"}
                  </button>
                </div>

                ${inventoryLoading
                  ? html`<div className="inline-progress"><${Loader} label="Scanning checkpoints..." inline=${true} /></div>`
                  : null}

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

              <div className=${`status ${status.type || ""}`.trim()}>
                ${status.text}
              </div>

              ${running ? html`<${BusyOverlay} label="Generating segmentation..." />` : null}
            </section>
          ` : null}

          ${activePage === "generate" ? html`
            <section className="generate-layout page-pane workflow-pane workflow-generate">
              <section className="prime-card viewer-card">
                <div className="card-head">
                  <div>
                    <h2>3D Brain Stage</h2>
                    <p>Generation + report are combined in this step.</p>
                  </div>
                  <div className="badge-row">
                    <div className="badge">Mode: ${intakeMode}</div>
                    <div className="badge">Case: ${selectedDemoCaseId || "manual"}</div>
                    <div className="badge">Engine: ${result && result.inference ? result.inference.engine : "n/a"}</div>
                  </div>
                </div>

                <div className="viewer-layout">
                  <div className="viewer-stage">
                    <div ref=${viewerRef}></div>
                    ${!result ? html`
                      <div className="viewer-overlay">
                        <div>
                          <strong>Ready to generate</strong>
                          Run generation to produce tumor meshes and report content.
                        </div>
                      </div>
                    ` : null}
                  </div>

                  <aside className="scene-controls">
                    <h3>Scene Controls</h3>
                    <p className="mini-note">
                      Tune transparency, toggle layers, and export high-resolution stage snapshots.
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

              <div className="generate-side">
                <section className="prime-card report-card" ref=${reportCardRef}>
                  <div className="card-head">
                    <div>
                      <h2>Report Forge</h2>
                      <p>Generate and inspect report content before final export.</p>
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

                  <div className=${`status ${status.type || ""}`}>
                    ${status.text}
                  </div>
                </section>

                <section className="prime-card metrics-card">
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
                      ? html`<div className="empty-chip">Run generation to populate the quantitative board.</div>`
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

              ${running ? html`<${BusyOverlay} label="Running generation and report synthesis..." />` : null}
            </section>
          ` : null}

          ${activePage === "export" ? html`
            <section className="prime-card export-card page-pane page-pane-narrow workflow-pane workflow-export">
              <div className="card-head">
                <div>
                  <h2>Export Package</h2>
                  <p>Finalize and export report assets.</p>
                </div>
                <div className="badge">${reportObject ? `Case: ${reportObject.case_id}` : "No report"}</div>
              </div>

              <div className="stack-actions export-actions">
                <button type="button" className="btn" disabled=${!reportObject} onClick=${exportReportJson}>Export JSON</button>
                <button type="button" className="btn" disabled=${!reportObject} onClick=${exportReportMarkdown}>Export Markdown</button>
                <button type="button" className="btn" disabled=${!reportObject || exportingPdf} onClick=${exportReportPdf}>
                  ${exportingPdf
                    ? html`<span className="btn-loader"><${Loader} label="Exporting PDF..." size="sm" inline=${true} /></span>`
                    : "Export PDF"}
                </button>
                <button type="button" className="btn" disabled=${!reportObject} onClick=${copyExecutiveSummary}>Copy Summary</button>
              </div>

              <p className="mini-note">Use Back if you want to regenerate with a different patient or intake settings.</p>

              <div className="preview-box">
                <pre>${markdownPreview}</pre>
              </div>

              ${exportingPdf ? html`<${BusyOverlay} label="Rendering PDF export..." />` : null}
            </section>
          ` : null}
        </section>

        <footer className="workflow-footer">
          <button type="button" className="btn btn-soft" disabled=${activePageIndex <= 0} onClick=${goBackStep}>Back</button>
          <span className="workflow-step-meta">Step ${activePageIndex + 1} of ${WORKFLOW_STEPS.length}</span>
          <button type="button" className="btn" disabled=${footerPrimaryDisabled} onClick=${footerPrimaryAction}>
            ${footerPrimaryContent}
          </button>
        </footer>
      </main>
    `;
  }

  ReactDOM.createRoot(document.getElementById("root")).render(html`<${App} />`);
})();
