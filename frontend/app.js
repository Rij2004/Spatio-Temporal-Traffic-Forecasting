const state = {
  chart: null,
  sensors: [],
};

const els = {
  apiUrl: document.querySelector("#apiUrl"),
  sensorSelect: document.querySelector("#sensorSelect"),
  trafficInput: document.querySelector("#trafficInput"),
  sampleBtn: document.querySelector("#sampleBtn"),
  randomBtn: document.querySelector("#randomBtn"),
  predictBtn: document.querySelector("#predictBtn"),
  returnAllSensors: document.querySelector("#returnAllSensors"),
  status: document.querySelector("#status"),
  selectedSensor: document.querySelector("#selectedSensor"),
  modelFamily: document.querySelector("#modelFamily"),
  nextStep: document.querySelector("#nextStep"),
  predictionRows: document.querySelector("#predictionRows"),
  chart: document.querySelector("#forecastChart"),
};

function apiBase() {
  return els.apiUrl.value.replace(/\/$/, "");
}

function setStatus(message, isError = false) {
  els.status.textContent = message;
  els.status.classList.toggle("error", isError);
}

function setBusy(isBusy) {
  els.predictBtn.disabled = isBusy;
  els.sampleBtn.disabled = isBusy;
  els.randomBtn.disabled = isBusy;
}

function parseTrafficInput() {
  const values = els.trafficInput.value
    .split(/[\s,;]+/)
    .map((value) => value.trim())
    .filter(Boolean)
    .map(Number);

  if (values.length !== 12 || values.some((value) => Number.isNaN(value))) {
    throw new Error("Enter exactly 12 numeric traffic values.");
  }

  return values;
}

function writeTraffic(values) {
  els.trafficInput.value = values.map((value) => Number(value).toFixed(2)).join(", ");
}

function randomSeries() {
  const base = 45 + Math.random() * 18;
  const slope = -3 + Math.random() * 6;
  return Array.from({ length: 12 }, (_, index) => {
    const wave = Math.sin((index / 11) * Math.PI) * (4 + Math.random() * 5);
    const jitter = (Math.random() - 0.5) * 3;
    return Math.max(5, Math.min(95, base + slope * (index / 11) + wave + jitter));
  });
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed with ${response.status}`);
  }
  return payload;
}

async function loadSensors() {
  try {
    const sensors = await fetchJson(`${apiBase()}/sensors`);
    state.sensors = sensors;
    els.sensorSelect.innerHTML = sensors
      .map((sensor) => `<option value="${sensor.id}">${sensor.name}</option>`)
      .join("");
    setStatus(`Loaded ${sensors.length} sensors.`);
  } catch (error) {
    els.sensorSelect.innerHTML = Array.from({ length: 207 }, (_, id) => (
      `<option value="${id}">Sensor ${String(id).padStart(3, "0")}</option>`
    )).join("");
    setStatus(error.message, true);
  }
}

async function loadSample() {
  setBusy(true);
  try {
    const sensorId = Number(els.sensorSelect.value || 0);
    const sample = await fetchJson(`${apiBase()}/sample?sensor_id=${sensorId}`);
    writeTraffic(sample.traffic_sequence);
    setStatus("Sample loaded.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setBusy(false);
  }
}

function renderRows(predicted) {
  els.predictionRows.innerHTML = predicted
    .map((value, index) => `
      <tr>
        <td>+${(index + 1) * 5} min</td>
        <td>${Number(value).toFixed(3)}</td>
      </tr>
    `)
    .join("");
}

function renderChart(past, predicted) {
  const labels = [
    ...past.map((_, index) => `${(index - past.length + 1) * 5}m`),
    ...predicted.map((_, index) => `+${(index + 1) * 5}m`),
  ];
  const actualLine = [...past, ...Array(predicted.length).fill(null)];
  const predictedLine = [
    ...Array(Math.max(past.length - 1, 0)).fill(null),
    past[past.length - 1],
    ...predicted,
  ];

  if (state.chart) {
    state.chart.destroy();
  }

  state.chart = new Chart(els.chart, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Actual input",
          data: actualLine,
          borderColor: "#2457a6",
          backgroundColor: "rgba(36, 87, 166, 0.12)",
          borderWidth: 2,
          pointRadius: 3,
          spanGaps: false,
          tension: 0.25,
        },
        {
          label: "Predicted",
          data: predictedLine,
          borderColor: "#c84934",
          backgroundColor: "rgba(200, 73, 52, 0.12)",
          borderWidth: 2,
          borderDash: [7, 5],
          pointRadius: 3,
          spanGaps: false,
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Time steps",
          },
        },
        y: {
          title: {
            display: true,
            text: "Traffic speed",
          },
          suggestedMin: 0,
          suggestedMax: 100,
        },
      },
    },
  });
}

async function predict() {
  setBusy(true);
  setStatus("Running model inference...");

  try {
    const sensorId = Number(els.sensorSelect.value || 0);
    const sequence = parseTrafficInput();
    const payload = await fetchJson(`${apiBase()}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        sensor_id: sensorId,
        traffic_sequence: sequence,
        return_all_sensors: els.returnAllSensors.checked,
      }),
    });

    renderChart(payload.past_traffic, payload.predicted_traffic);
    renderRows(payload.predicted_traffic);
    els.selectedSensor.textContent = `Sensor ${payload.sensor_id}`;
    els.modelFamily.textContent = payload.model_family;
    els.nextStep.textContent = Number(payload.predicted_traffic[0]).toFixed(3);
    setStatus(payload.simulated_missing_nodes ? "Prediction complete. Missing nodes were simulated." : "Prediction complete.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setBusy(false);
  }
}

els.sampleBtn.addEventListener("click", loadSample);
els.randomBtn.addEventListener("click", () => {
  writeTraffic(randomSeries());
  setStatus("Random sample ready.");
});
els.predictBtn.addEventListener("click", predict);
els.apiUrl.addEventListener("change", loadSensors);

writeTraffic(randomSeries());
loadSensors().then(loadSample);

