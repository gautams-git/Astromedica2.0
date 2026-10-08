console.log("[app.js] loaded", new Date().toISOString());

const $ = (id) => document.getElementById(id);

let currentJobId = null;
let pollTimer = null;
let trainPollTimer = null;

function showError(msg) {
  const el = $("errorBox");
  if (!el) return;
  if (msg) { el.textContent = msg; el.classList.remove("hidden"); }
  else { el.textContent = ""; el.classList.add("hidden"); }
}

function validateInput(data) {
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    return "Top level must be an object with a 'conditions' key.";
  }
  if (!Array.isArray(data.conditions)) {
    return "Missing 'conditions' array.";
  }
  if (data.chat_log !== undefined && !Array.isArray(data.chat_log)) {
    return "'chat_log' must be an array.";
  }
  for (let i = 0; i < data.conditions.length; i++) {
    const item = data.conditions[i];
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      return `conditions[${i}]: must be an object.`;
    }
    const keys = Object.keys(item);
    if (keys.length !== 1) {
      return `conditions[${i}]: must have exactly one organ key.`;
    }
    const v = item[keys[0]];
    if (!Array.isArray(v)) {
      return `conditions[${i}] ("${keys[0]}"): value must be an array.`;
    }
    if (!v.length) {
      return `conditions[${i}] ("${keys[0]}"): empty.`;
    }
  }
  return null;
}

function escapeHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;")
                  .replace(/>/g,"&gt;").replace(/"/g,"&quot;")
                  .replace(/'/g,"&#39;");
}

function renderPreview(conditions) {
  const list = $("previewList");
  if (!list) return;
  list.innerHTML = "";
  if (!conditions || !conditions.length) {
    list.innerHTML = '<li class="hint">Nothing to preview.</li>';
    return;
  }
  conditions.forEach((item) => {
    if (!item || typeof item !== "object") return;
    for (const [organ, diseases] of Object.entries(item)) {
      const li = document.createElement("li");
      const pills = (Array.isArray(diseases) ? diseases : [])
        .map((d) => `<span class="pill">${escapeHtml(d)}</span>`).join(" ");
      li.innerHTML = `<strong>[${escapeHtml(organ)}]</strong> ${pills}`;
      list.appendChild(li);
    }
  });
}

function setBadge(el, ok, okText, badText) {
  if (!el) return;
  el.textContent = ok ? okText : badText;
  el.className = "badge " + (ok ? "badge-ok" : "badge-muted");
}

async function refreshTrainingStatus() {
  try {
    const r = await fetch("/api/training/status", { cache: "no-store" });
    if (!r.ok) return;
    const s = await r.json();
    setBadge($("pretrainBadge"), !!s.pretrained,
             "ANN: pretrained", "ANN: not trained");
    setBadge($("ppoBadge"), !!s.ppo_trained,
             "PPO: trained", "PPO: not trained");
    const dev = $("deviceBadge");
    if (dev) {
      dev.textContent = "device: " + (s.device || "cpu");
      dev.className = "badge badge-muted";
    }
  } catch (_) {}
}

async function startTraining(kind) {
  const errEl = $("trainError");
  if (errEl) { errEl.textContent = ""; errEl.classList.add("hidden"); }
  const btn = kind === "pretrain" ? $("pretrainBtn") : $("ppoBtn");
  const orig = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Starting…";
  try {
    const r = await fetch(`/api/training/${kind}`, { method: "POST" });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "Training request failed.");
    $("trainJobBox").classList.remove("hidden");
    $("trainJobId").textContent = data.job_id;
    setTrainJobStatus("queued", 0, data.message || "Queued");
    pollTrainJob(data.job_id);
  } catch (e) {
    if (errEl) {
      errEl.textContent = "Error: " + e.message;
      errEl.classList.remove("hidden");
    }
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}

function pollTrainJob(jobId) {
  if (trainPollTimer) clearInterval(trainPollTimer);
  trainPollTimer = setInterval(async () => {
    try {
      const r = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
      if (!r.ok) return;
      const job = await r.json();
      setTrainJobStatus(job.status, job.progress || 0, job.message || "");
      if (["done", "error", "cancelled"].includes(job.status)) {
        clearInterval(trainPollTimer);
        trainPollTimer = null;
        refreshTrainingStatus();
      }
    } catch (_) {}
  }, 1000);
}

function setTrainJobStatus(status, progress, message) {
  const el = $("trainJobStatus");
  if (!el) return;
  el.textContent = String(status).toUpperCase();
  el.className = "status status-" + status;
  $("trainProgressBar").style.width = Math.round((progress || 0) * 100) + "%";
  $("trainJobMessage").textContent = message || "—";
}

document.addEventListener("DOMContentLoaded", () => {
  console.log("[app.js] DOMContentLoaded");

  const submitBtn = $("submitBtn");
  const clearBtn = $("clearBtn");
  const fileInput = $("fileInput");
  const pretrainBtn = $("pretrainBtn");
  const ppoBtn = $("ppoBtn");
  const refreshTrainBtn = $("refreshTrainBtn");

  if (submitBtn) submitBtn.addEventListener("click", async () => {
    showError("");
    const raw = $("jsonInput").value.trim();
    if (!raw) { showError("Please paste the input JSON."); return; }
    let parsed;
    try { parsed = JSON.parse(raw); }
    catch (e) { showError("Invalid JSON: " + e.message); return; }
    const err = validateInput(parsed);
    if (err) { showError(err); return; }
    renderPreview(parsed.conditions);
    await submitBody(raw);
  });

  if (fileInput) fileInput.addEventListener("change", async (e) => {
    showError("");
    const file = e.target.files[0];
    if (!file) return;
    const text = await file.text();
    let parsed;
    try { parsed = JSON.parse(text); }
    catch (err) { showError("File is not valid JSON: " + err.message); return; }
    const vErr = validateInput(parsed);
    if (vErr) { showError(vErr); return; }
    $("jsonInput").value = JSON.stringify(parsed, null, 2);
    renderPreview(parsed.conditions);
    await submitBody(text);
  });

  if (clearBtn) clearBtn.addEventListener("click", () => {
    $("jsonInput").value = "";
    $("fileInput").value = "";
    showError("");
    $("previewList").innerHTML =
      '<li class="hint">Submit JSON to preview.</li>';
    $("jobBox").classList.add("hidden");
    $("noJob").classList.remove("hidden");
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    currentJobId = null;
  });

  if (pretrainBtn) pretrainBtn.addEventListener("click",
                                                () => startTraining("pretrain"));
  if (ppoBtn) ppoBtn.addEventListener("click", () => startTraining("ppo"));
  if (refreshTrainBtn) refreshTrainBtn.addEventListener("click",
                                                       refreshTrainingStatus);

  refreshTrainingStatus();
  setInterval(refreshTrainingStatus, 5000);
});

async function submitBody(bodyText) {
  const btn = $("submitBtn");
  btn.disabled = true;
  const orig = btn.textContent;
  btn.textContent = "Submitting…";
  try {
    const r = await fetch("/api/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: bodyText,
      cache: "no-store",
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || "Upload failed.");
    }
    const data = await r.json();
    currentJobId = data.job_id;
    if (data.chat_summary_preview) {
      const li = document.createElement("li");
      li.innerHTML = `<strong>Chat log detected:</strong> ` +
        `<span class="pill">patient ${escapeHtml(data.chat_summary_preview.patient_id || "?")}</span> ` +
        (data.chat_summary_preview.diseases_detected || [])
          .map(d => `<span class="pill">${escapeHtml(d)}</span>`).join(" ");
      $("previewList").appendChild(li);
    }
    $("jobBox").classList.remove("hidden");
    $("noJob").classList.add("hidden");
    $("jobId").textContent = data.job_id;
    setStatus("queued", 0, data.message || "Queued");
    for (const id of ["downloadBtn", "viewBtn"]) {
      const el = $(id);
      el.classList.add("disabled");
      el.href = "#";
    }
    startPolling();
  } catch (err) {
    showError("Submit error: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollJob, 1000);
  pollJob();
}

async function pollJob() {
  if (!currentJobId) return;
  try {
    const r = await fetch(`/api/jobs/${currentJobId}`, { cache: "no-store" });
    if (!r.ok) return;
    const job = await r.json();
    setStatus(job.status, job.progress || 0, job.message || "");
    if (job.status === "done") {
      clearInterval(pollTimer); pollTimer = null;
      const dl = $("downloadBtn");
      dl.href = `/api/jobs/${currentJobId}/download`;
      dl.classList.remove("disabled");
      const vw = $("viewBtn");
      vw.href = `/api/jobs/${currentJobId}/result`;
      vw.classList.remove("disabled");
    } else if (job.status === "error" || job.status === "cancelled") {
      clearInterval(pollTimer); pollTimer = null;
    }
  } catch (_) {}
}

function setStatus(status, progress, message) {
  const el = $("jobStatus");
  if (!el) return;
  el.textContent = String(status).toUpperCase();
  el.className = "status status-" + status;
  $("progressBar").style.width = Math.round((progress || 0) * 100) + "%";
  $("jobMessage").textContent = message || "—";
}