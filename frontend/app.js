const fileInput = document.getElementById("mriFile");
const modalitySelect = document.getElementById("modalityIndex");
const engineSelect = document.getElementById("engineMode");
const thresholdInput = document.getElementById("maskThreshold");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const metricsGrid = document.getElementById("metricsGrid");

function setStatus(text, type = "") {
  statusEl.textContent = text;
  statusEl.className = "status " + type;
}

function renderMetrics(data) {
  const rows = [
    ["Detected", data.detected ? "Yes" : "No", data.detected ? "good" : "bad"],
    ["Tumor voxels", String(data.voxel_count)],
    ["Occupancy", `${data.occupancy_percent.toFixed(4)} %`],
    ["Volume", `${data.volume_mm3.toFixed(2)} mm³`],
    ["Volume", `${data.volume_ml.toFixed(3)} mL`],
    ["Equivalent diameter", `${data.equivalent_diameter_mm.toFixed(2)} mm`],
    ["BBox min", `[${data.bbox_min.join(", ")}]`],
    ["BBox max", `[${data.bbox_max.join(", ")}]`],
    ["Extent", `[${data.extent_mm.map((x) => x.toFixed(2)).join(", ")}] mm`],
    ["Centroid voxel", `[${data.centroid_voxel.map((x) => x.toFixed(2)).join(", ")}]`],
    ["Centroid mm", `[${data.centroid_mm.map((x) => x.toFixed(2)).join(", ")}]`],
  ];

  metricsGrid.innerHTML = rows
    .map(
      ([k, v, cls = ""]) =>
        `<div class="metric"><div class="k">${k}</div><div class="v ${cls}">${v}</div></div>`,
    )
    .join("");
}

function renderMesh(mesh, inputInfo) {
  const viewer = document.getElementById("viewer");

  if (!mesh.vertices || mesh.vertices.length === 0) {
    Plotly.purge(viewer);
    setStatus("No tumor region detected in this volume.", "bad");
    return;
  }

  const x = mesh.vertices.map((v) => v[0]);
  const y = mesh.vertices.map((v) => v[1]);
  const z = mesh.vertices.map((v) => v[2]);
  const i = mesh.faces.map((f) => f[0]);
  const j = mesh.faces.map((f) => f[1]);
  const k = mesh.faces.map((f) => f[2]);

  const trace = {
    type: "mesh3d",
    x,
    y,
    z,
    i,
    j,
    k,
    opacity: 0.44,
    color: "#ff9a62",
    flatshading: true,
    lighting: {
      ambient: 0.5,
      diffuse: 0.7,
      roughness: 0.6,
      fresnel: 0.1,
      specular: 0.2,
    },
  };

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

  Plotly.newPlot(viewer, [trace], layout, { responsive: true, displaylogo: false });
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
    const formData = new FormData();
    formData.append("file", file);
    formData.append("modality_index", modalitySelect.value);
    formData.append("engine", engineSelect.value);
    formData.append("threshold", thresholdInput.value);

    const response = await fetch("/api/segment", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Segmentation request failed.");
    }

    renderMetrics(payload.metrics);
    renderMesh(payload.mesh, payload.input);

    setStatus(
      `Done. Engine: ${payload.inference.engine} | Vertices: ${payload.mesh.vertex_count} | Faces: ${payload.mesh.face_count} | Modality index: ${payload.input.modality_index}`,
      "good",
    );
  } catch (error) {
    setStatus(`Error: ${error.message}`, "bad");
  } finally {
    runBtn.disabled = false;
  }
});
