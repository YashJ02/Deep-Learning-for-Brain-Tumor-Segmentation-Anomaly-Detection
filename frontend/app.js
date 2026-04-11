const fileInput = document.getElementById("mriFile");
const modalitySelect = document.getElementById("modalityIndex");
const engineSelect = document.getElementById("engineMode");
const thresholdInput = document.getElementById("maskThreshold");
const foldSelectorEl = document.getElementById("foldSelector");
const foldSelectorInfoEl = document.getElementById("foldSelectorInfo");
const foldSelectAllBtn = document.getElementById("foldSelectAllBtn");
const foldClearBtn = document.getElementById("foldClearBtn");
const foldRefreshBtn = document.getElementById("foldRefreshBtn");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const metricsGrid = document.getElementById("metricsGrid");

function setStatus(text, type = "") {
  statusEl.textContent = text;
  statusEl.className = "status " + type;
}

function listOrNA(values) {
  return values && values.length ? values.join(", ") : "N/A";
}

function renderMetrics(metrics, inference, classMetrics = null) {
  const rows = [
    ["Detected", metrics.detected ? "Yes" : "No", metrics.detected ? "good" : "bad"],
    ["Tumor voxels", String(metrics.voxel_count)],
    ["Occupancy", `${metrics.occupancy_percent.toFixed(4)} %`],
    ["Volume", `${metrics.volume_mm3.toFixed(2)} mm³`],
    ["Volume", `${metrics.volume_ml.toFixed(3)} mL`],
    ["Equivalent diameter", `${metrics.equivalent_diameter_mm.toFixed(2)} mm`],
    ["BBox min", `[${metrics.bbox_min.join(", ")}]`],
    ["BBox max", `[${metrics.bbox_max.join(", ")}]`],
    ["Extent", `[${metrics.extent_mm.map((x) => x.toFixed(2)).join(", ")}] mm`],
    ["Centroid voxel", `[${metrics.centroid_voxel.map((x) => x.toFixed(2)).join(", ")}]`],
    ["Centroid mm", `[${metrics.centroid_mm.map((x) => x.toFixed(2)).join(", ")}]`],
    ["Inference engine", String(inference.engine || "unknown")],
    [
      "Ensemble size",
      Number.isFinite(inference.ensemble_size) ? String(inference.ensemble_size) : "N/A",
    ],
    [
      "Probability mean",
      Number.isFinite(inference.probability_mean) ? inference.probability_mean.toFixed(4) : "N/A",
    ],
    [
      "Probability max",
      Number.isFinite(inference.probability_max) ? inference.probability_max.toFixed(4) : "N/A",
    ],
    ["Fold indices used", listOrNA(inference.fold_indices || [])],
    [
      "Checkpoints used",
      Array.isArray(inference.checkpoints)
        ? `${inference.checkpoints.length} selected`
        : inference.checkpoint || "N/A",
    ],
  ];

  const classOrder = ["1", "2", "4"];
  classOrder.forEach((label) => {
    const entry = classMetrics?.[label];
    if (!entry) {
      return;
    }
    rows.push([`${entry.name} voxels`, String(entry.voxel_count), entry.detected ? "good" : ""]);
    rows.push([`${entry.name} volume`, `${Number(entry.volume_ml || 0).toFixed(3)} mL`]);
  });

  metricsGrid.innerHTML = rows
    .map(
      ([k, v, cls = ""]) =>
        `<div class="metric"><div class="k">${k}</div><div class="v ${cls}">${v}</div></div>`,
    )
    .join("");
}

function getSelectedFoldIndices() {
  return Array.from(foldSelectorEl.querySelectorAll("input[type='checkbox']:checked"))
    .map((node) => Number(node.value))
    .filter((value) => Number.isInteger(value) && value >= 0)
    .sort((a, b) => a - b);
}

function getFoldCheckboxes() {
  return Array.from(foldSelectorEl.querySelectorAll("input[type='checkbox']"));
}

function setFoldActionState({ loading = false, hasFolds = false } = {}) {
  foldSelectAllBtn.disabled = loading || !hasFolds;
  foldClearBtn.disabled = loading || !hasFolds;
  foldRefreshBtn.disabled = loading;
}

function setAllFoldCheckboxes(checked) {
  getFoldCheckboxes().forEach((checkbox) => {
    checkbox.checked = checked;
  });
}

function updateFoldSelectionInfo() {
  const total = getFoldCheckboxes().length;
  const selected = getSelectedFoldIndices().length;
  const deepExists = foldSelectorInfoEl.dataset.deepExists === "1";

  if (total === 0) {
    foldSelectorInfoEl.textContent = `No fold checkpoints found. Deep checkpoint available: ${deepExists ? "yes" : "no"}`;
    return;
  }

  foldSelectorInfoEl.textContent = `${selected}/${total} folds selected. Deep checkpoint available: ${deepExists ? "yes" : "no"}`;
}

function renderFoldSelector(inventory) {
  const folds = inventory?.ensemble?.folds || [];
  const deepExists = Boolean(inventory?.deep?.exists);
  foldSelectorInfoEl.dataset.deepExists = deepExists ? "1" : "0";

  if (!folds.length) {
    foldSelectorEl.innerHTML = `<div class="fold-empty">No fold checkpoints found in models/kfold.</div>`;
    setFoldActionState({ loading: false, hasFolds: false });
    updateFoldSelectionInfo();
    return;
  }

  foldSelectorEl.innerHTML = folds
    .map((entry) => {
      const foldLabel = Number.isInteger(entry.fold_index)
        ? `Fold ${entry.fold_index}`
        : `Fold ?`;
      return `
        <label class="fold-option" title="${entry.path}">
          <input type="checkbox" value="${entry.fold_index}" checked />
          <span>${foldLabel}</span>
        </label>
      `;
    })
    .join("");

  getFoldCheckboxes().forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      updateFoldSelectionInfo();
    });
  });

  setFoldActionState({ loading: false, hasFolds: true });
  updateFoldSelectionInfo();
}

async function loadCheckpointInventory() {
  try {
    setFoldActionState({ loading: true, hasFolds: false });
    foldSelectorInfoEl.textContent = "Loading checkpoint inventory...";
    const response = await fetch("/api/checkpoints");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Checkpoint inventory request failed.");
    }
    renderFoldSelector(payload);
  } catch (error) {
    setFoldActionState({ loading: false, hasFolds: false });
    foldSelectorInfoEl.dataset.deepExists = "0";
    foldSelectorEl.innerHTML = `<div class="fold-empty">Could not load checkpoint inventory.</div>`;
    foldSelectorInfoEl.textContent = String(error.message || error);
  }
}

function scaledVertices(vertices, scale = 1) {
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

function buildMeshTrace(mesh, { color, opacity, name, scale = 1 }) {
  const vertices = scaledVertices(mesh.vertices, scale);

  return {
    type: "mesh3d",
    x: vertices.map((v) => v[0]),
    y: vertices.map((v) => v[1]),
    z: vertices.map((v) => v[2]),
    i: mesh.faces.map((f) => f[0]),
    j: mesh.faces.map((f) => f[1]),
    k: mesh.faces.map((f) => f[2]),
    opacity,
    color,
    name,
    flatshading: true,
    showscale: false,
    lighting: {
      ambient: 0.5,
      diffuse: 0.7,
      roughness: 0.6,
      fresnel: 0.1,
      specular: 0.2,
    },
  };
}

function renderMesh(tumorMesh, inputInfo, brainMesh = null, classMeshes = []) {
  const viewer = document.getElementById("viewer");

  const hasTumor = Boolean(tumorMesh?.vertices?.length);
  const hasBrain = Boolean(brainMesh?.vertices?.length);
  const usableClassMeshes = Array.isArray(classMeshes)
    ? classMeshes.filter((entry) => Boolean(entry?.mesh?.vertices?.length))
    : [];
  const hasClassMeshes = usableClassMeshes.length > 0;

  if (!hasTumor && !hasBrain && !hasClassMeshes) {
    Plotly.purge(viewer);
    setStatus("No renderable surfaces were detected in this volume.", "bad");
    return;
  }

  const traces = [];
  if (hasBrain) {
    traces.push(
      buildMeshTrace(brainMesh, {
        color: "#4c96ff",
        opacity: 0.34,
        name: "Brain",
      }),
    );
  }
  if (hasClassMeshes) {
    usableClassMeshes.forEach((entry) => {
      traces.push(
        buildMeshTrace(entry.mesh, {
          color: entry.color || "#f59e0b",
          opacity: 0.78,
          name: entry.name || `Class ${entry.label}`,
          scale: 0.92,
        }),
      );
    });
  } else if (hasTumor) {
    traces.push(
      buildMeshTrace(tumorMesh, {
        color: "#ffb347",
        opacity: 0.82,
        name: "Tumor",
        scale: 0.92,
      }),
    );
  }

  const layout = {
    paper_bgcolor: "#0d1b2c",
    plot_bgcolor: "#0d1b2c",
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: {
      bgcolor: "#0d1b2c",
      xaxis: { title: "X", color: "#9bb2ce" },
      yaxis: { title: "Y", color: "#9bb2ce" },
      zaxis: { title: "Z", color: "#9bb2ce" },
      aspectmode: "data",
    },
    legend: {
      orientation: "h",
      yanchor: "top",
      y: 1,
      xanchor: "right",
      x: 1,
      font: { color: "#c2d3e9", size: 12 },
      bgcolor: "rgba(13, 27, 44, 0.45)",
      bordercolor: "rgba(155, 178, 206, 0.2)",
      borderwidth: 1,
    },
    annotations: [
      {
        xref: "paper",
        yref: "paper",
        x: 0.01,
        y: 0.99,
        showarrow: false,
        font: { color: "#9bb2ce", size: 11 },
        text: `Shape: ${inputInfo.volume_shape.join(" x ")} | Spacing: ${inputInfo.voxel_spacing_mm.map((x) => x.toFixed(2)).join(", ")} mm`,
      },
    ],
  };

  Plotly.newPlot(viewer, traces, layout, { responsive: true, displaylogo: false });
}

runBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    setStatus("Please choose a .nii or .nii.gz file first.", "bad");
    return;
  }

  runBtn.disabled = true;
  setStatus("Uploading volume and running segmentation...", "");

  try {
    const selectedFolds = getSelectedFoldIndices();
    if (engineSelect.value === "ensemble" && selectedFolds.length === 0) {
      throw new Error("Select at least one fold checkpoint for ensemble mode.");
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("modality_index", modalitySelect.value);
    formData.append("engine", engineSelect.value);
    formData.append("threshold", thresholdInput.value);
    if (selectedFolds.length) {
      formData.append("ensemble_folds", selectedFolds.join(","));
    }

    const response = await fetch("/api/segment", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Segmentation request failed.");
    }

    renderMetrics(payload.metrics, payload.inference, payload.class_metrics || {});
    renderMesh(payload.mesh, payload.input, payload.brain_mesh, payload.class_meshes || []);

    const ensembleSuffix = Number.isFinite(payload.inference.ensemble_size)
      ? ` | Ensemble size: ${payload.inference.ensemble_size}`
      : "";
    const taskSuffix = payload.inference.task ? ` | Task: ${payload.inference.task}` : "";
    const classSuffix = Array.isArray(payload.class_meshes) && payload.class_meshes.length
      ? ` | Class meshes: ${payload.class_meshes.length}`
      : "";

    setStatus(
      `Done. Engine: ${payload.inference.engine}${ensembleSuffix}${taskSuffix}${classSuffix} | Vertices: ${payload.mesh.vertex_count} | Faces: ${payload.mesh.face_count} | Modality index: ${payload.input.modality_index}`,
      "good",
    );
  } catch (error) {
    setStatus(`Error: ${error.message}`, "bad");
  } finally {
    runBtn.disabled = false;
  }
});

foldSelectAllBtn.addEventListener("click", () => {
  setAllFoldCheckboxes(true);
  updateFoldSelectionInfo();
});

foldClearBtn.addEventListener("click", () => {
  setAllFoldCheckboxes(false);
  updateFoldSelectionInfo();
});

foldRefreshBtn.addEventListener("click", async () => {
  await loadCheckpointInventory();
});

loadCheckpointInventory();
