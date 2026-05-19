/* ───────────────────────────────────────────────────────────────
   VisionAsk — wizard frontend
   ─────────────────────────────────────────────────────────────── */

const BACKEND = ""; // same origin

const QUESTION_TYPES = {
  "Custom":                     "",
  "Counting":                   "How many objects are visible in this image?",
  "Object Recognition":         "What are the main objects present in this image?",
  "Object Presence":            "Is there a person in this image? What other objects are present?",
  "Color":                      "What are the dominant colors in this image?",
  "Scene Understanding":        "Describe the overall scene and setting of this image.",
  "Animal Recognition":         "What animal or animals are visible in this image? Describe their appearance.",
  "Activity Recognition":       "What activity or action is being performed in this image?",
  "Sports Recognition":         "What sport or athletic activity is shown in this image?",
  "Transportation Recognition": "What mode of transportation is visible in this image?",
  "Text Recognition":           "What text, signs, or written words are visible in this image?",
  "Food Recognition":           "What food or dish is shown in this image? Describe it.",
};

const TYPE_COLORS = {
  "Custom":                     "#7E96C6",
  "Counting":                   "#5B8DEF",
  "Object Recognition":         "#7BA5FF",
  "Object Presence":            "#5EE3A8",
  "Color":                      "#FBC156",
  "Scene Understanding":        "#6BC8FF",
  "Animal Recognition":         "#A0CFFF",
  "Activity Recognition":       "#E08DFF",
  "Sports Recognition":         "#4FD8C8",
  "Transportation Recognition": "#FFAB6B",
  "Text Recognition":           "#C9DCFF",
  "Food Recognition":           "#F87B7B",
};

const BOX_PALETTE = [
  "#5B8DEF","#7BA5FF","#6BC8FF","#5EE3A8",
  "#FBC156","#E08DFF","#FFAB6B","#A0CFFF",
];

// ─── State ─────────────────────────────────────────────────────
const state = {
  step: 1,
  file: null,
  fileURL: null,
  selectedType: "Custom",
  lastPreset: "",
  lastResult: null,
  history: [],
  backendOk: false,
  busy: false,
};

// ─── Element refs ──────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const els = {
  // dropzone
  dropzone:   $("dropzone"),
  fileInput:  $("file-input"),
  dzEmpty:    $("dz-empty"),
  dzMeta:     $("dz-meta"),
  dzName:     $("dz-name"),
  dzSize:     $("dz-size"),
  dzThumb:    $("dz-thumb"),
  dzClear:    $("dz-clear"),
  // type + question
  typePill:   $("type-pill"),
  typeSelect: $("type-select"),
  question:   $("question"),
  // step 2
  previewImg:      $("preview-img"),
  confirmImage:    $("confirm-image"),
  confirmType:     $("confirm-type"),
  confirmQuestion: $("confirm-question"),
  // step 3
  annotatedImg: $("annotated-img"),
  originalImg:  $("original-img"),
  dinoCaption:  $("dino-caption"),
  answerCard:   $("answer-card"),
  answerText:   $("answer-text"),
  metaRow:      $("meta-row"),
  confPct:      $("conf-pct"),
  confFill:     $("conf-fill"),
  groundSection: $("ground-section"),
  groundCount:   $("ground-count"),
  boxList:       $("box-list"),
  historyList:   $("history-list"),
  historyEmpty:  $("history-empty"),
  histCount:     $("hist-count"),
  clearBtn:      $("clear-history"),
  // wizard
  stepper:    $("stepper"),
  viewport:   $("viewport"),
  actionBar:  $("action-bar"),
  btnBack:    $("btn-back"),
  btnNext:    $("btn-next"),
  btnNextLabel: $("btn-next-label"),
  btnNextSpin:  $("btn-next-spin"),
  stepStatus: $("step-status"),
  // status
  statusDot:    $("status-dot"),
  backendLabel: $("backend-label"),
  vram:         $("vram"),
  offlineBanner: $("offline-banner"),
  toast: $("toast"),
};

// ─── Helpers ───────────────────────────────────────────────────
const confColor = (c) => c >= 0.75 ? "#5EE3A8" : c >= 0.5 ? "#FBC156" : "#F87B7B";
const confLabel = (c) => c >= 0.75 ? "High" : c >= 0.5 ? "Medium" : "Low";
const fmtSize = (b) => b < 1024 ? `${b} B` : b < 1024*1024 ? `${(b/1024).toFixed(0)} KB` : `${(b/1024/1024).toFixed(1)} MB`;
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const nowTime = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

function toast(msg, kind = "") {
  els.toast.textContent = msg;
  els.toast.className = `toast show ${kind}`;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { els.toast.className = "toast hidden"; }, 3500);
}

// ─── Step machine ──────────────────────────────────────────────
function goToStep(n, direction = "forward") {
  if (state.busy) return;
  if (n < 1 || n > 3) return;

  // Validate forward transitions
  if (n > state.step) {
    if (state.step === 1 && !canLeaveStep1()) {
      toast(stepError(1), "warn");
      return;
    }
    // Step 2 → 3 is gated by ask() succeeding (not via direct goToStep)
    if (state.step === 2 && n === 3 && !state.lastResult) {
      // Trigger ask flow instead
      ask();
      return;
    }
  }

  state.step = n;

  // Swap panel
  document.querySelectorAll(".step-panel").forEach(p => {
    p.classList.remove("active", "from-back");
  });
  const target = document.querySelector(`.step-panel[data-panel="${n}"]`);
  target.classList.add("active");
  if (direction === "back") target.classList.add("from-back");

  // Update stepper
  document.querySelectorAll(".step").forEach(s => {
    const sn = parseInt(s.dataset.step, 10);
    s.classList.toggle("active", sn === n);
    s.classList.toggle("done", sn < n);
  });
  // Step line fills
  document.querySelectorAll(".step-line-fill").forEach(f => {
    const ln = parseInt(f.dataset.line, 10);
    f.classList.toggle("filled", ln < n);
  });

  // Update step-specific UI
  if (n === 2) populateStep2();
  if (n === 3 && state.lastResult) renderResult(state.lastResult);

  updateActionBar();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function canLeaveStep1() {
  return state.file && els.question.value.trim().length > 0;
}
function stepError(s) {
  if (s === 1) {
    if (!state.file) return "Please upload an image first.";
    if (!els.question.value.trim()) return "Enter or pick a question.";
  }
  return "";
}

function updateActionBar() {
  const s = state.step;
  els.btnBack.classList.toggle("hidden", s === 1);
  els.btnBack.textContent = "← Back";

  if (s === 1) {
    els.btnNextLabel.textContent = "Continue →";
    els.btnNext.disabled = !canLeaveStep1();
    els.stepStatus.textContent = state.file
      ? "Image ready — fill in your question to continue."
      : "Start by uploading an image.";
  } else if (s === 2) {
    els.btnNextLabel.textContent = state.busy ? "Running …" : "⚡ Run Analysis";
    els.btnNext.disabled = !state.backendOk || state.busy;
    els.btnNextSpin.classList.toggle("hidden", !state.busy);
    els.stepStatus.textContent = state.backendOk
      ? "Review everything below, then run the models."
      : "Backend is offline — cannot run analysis.";
    els.stepStatus.classList.toggle("error", !state.backendOk);
  } else if (s === 3) {
    els.btnNextLabel.textContent = "↻ Ask Another";
    els.btnNext.disabled = false;
    els.btnNextSpin.classList.add("hidden");
    els.stepStatus.textContent = state.lastResult
      ? `Completed in ${state.lastResult.inference_ms + (state.lastResult.grounding_ms || 0)} ms.`
      : "";
    els.stepStatus.classList.remove("error");
  }
}

function nextAction() {
  if (state.busy) return;
  if (state.step === 1) goToStep(2);
  else if (state.step === 2) ask();
  else if (state.step === 3) restart();
}

function backAction() {
  if (state.busy) return;
  if (state.step > 1) goToStep(state.step - 1, "back");
}

function restart() {
  // Keep history but reset everything else
  state.lastResult = null;
  goToStep(1, "back");
}

// ─── Type select / question ────────────────────────────────────
function buildTypeSelect() {
  els.typeSelect.innerHTML = Object.keys(QUESTION_TYPES)
    .map(k => `<option value="${k}">${k}</option>`).join("");
  els.typeSelect.value = state.selectedType;
}

function applyTypeUI() {
  const c = TYPE_COLORS[state.selectedType] || "#94a3b8";
  els.typePill.textContent = state.selectedType;
  els.typePill.style.background = `${c}26`;
  els.typePill.style.color      = c;
  els.typePill.style.borderColor = `${c}66`;
  els.typePill.style.boxShadow = `0 0 12px ${c}33`;
}

// ─── Step 2 population ─────────────────────────────────────────
function populateStep2() {
  if (state.fileURL) els.previewImg.src = state.fileURL;
  els.confirmImage.textContent = state.file
    ? `${state.file.name} · ${fmtSize(state.file.size)}`
    : "—";
  els.confirmType.textContent = state.selectedType;
  els.confirmQuestion.textContent = els.question.value.trim() || "—";
}

// ─── Health polling ────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(`${BACKEND}/health`, { cache: "no-store" });
    if (!r.ok) throw 0;
    const data = await r.json();
    state.backendOk = true;
    els.statusDot.classList.remove("off");
    els.backendLabel.textContent = "Online";
    els.offlineBanner.classList.add("hidden");
    if (typeof data.vram_used_mb === "number") {
      els.vram.textContent = `VRAM ${Math.round(data.vram_used_mb)}/${Math.round(data.vram_total_mb)} MB`;
      els.vram.classList.remove("hidden");
    } else {
      els.vram.classList.add("hidden");
    }
  } catch {
    state.backendOk = false;
    els.statusDot.classList.add("off");
    els.backendLabel.textContent = "Offline";
    els.offlineBanner.classList.remove("hidden");
    els.vram.classList.add("hidden");
  }
  updateActionBar();
}

// ─── File handling ─────────────────────────────────────────────
function setFile(file) {
  if (!file) { clearFile(); return; }
  if (!/^image\//.test(file.type)) { toast("Please drop an image file.", "warn"); return; }

  state.file = file;
  if (state.fileURL) URL.revokeObjectURL(state.fileURL);
  state.fileURL = URL.createObjectURL(file);
  state.lastResult = null;

  els.dzEmpty.classList.add("hidden");
  els.dzMeta.classList.remove("hidden");
  els.dzName.textContent = file.name;
  els.dzSize.textContent = fmtSize(file.size);
  els.dzThumb.src = state.fileURL;

  const img = new Image();
  img.onload = () => {
    els.dzSize.textContent = `${fmtSize(file.size)} · ${img.naturalWidth}×${img.naturalHeight}`;
  };
  img.src = state.fileURL;

  updateActionBar();
}

function clearFile() {
  state.file = null;
  if (state.fileURL) { URL.revokeObjectURL(state.fileURL); state.fileURL = null; }
  els.fileInput.value = "";
  els.dzThumb.src = "";
  els.dzEmpty.classList.remove("hidden");
  els.dzMeta.classList.add("hidden");
  state.lastResult = null;
  updateActionBar();
}

// ─── Inference ─────────────────────────────────────────────────
async function ask() {
  if (state.busy) return;
  if (!state.file) { toast("Upload an image first.", "warn"); return; }
  const q = els.question.value.trim();
  if (!q) { toast("Enter or select a question.", "warn"); return; }

  setBusy(true);
  try {
    const form = new FormData();
    form.append("image", state.file);
    form.append("question", q);
    form.append("question_type", state.selectedType);
    form.append("run_grounding", "yes");

    const resp = await fetch(`${BACKEND}/ask`, { method: "POST", body: form });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`Backend ${resp.status}: ${text || resp.statusText}`);
    }
    const d = await resp.json();

    const result = {
      answer:         d.answer,
      confidence:     d.confidence,
      inference_ms:   d.inference_ms,
      device:         d.device,
      grounding_ran:  d.grounding_ran,
      grounding_ms:   d.grounding_ms,
      annotated_b64:  d.annotated_image,
      boxes:          d.boxes || [],
      gdino_caption:  d.gdino_caption,
      question:       q,
      qtype:          state.selectedType,
      image_name:     state.file.name,
      timestamp:      nowTime(),
      original_url:   state.fileURL,
    };
    state.lastResult = result;
    state.history.unshift(result);

    setBusy(false);
    goToStep(3);
  } catch (err) {
    console.error(err);
    toast(err.message || "Request failed", "error");
    setBusy(false);
  }
}

function setBusy(on) {
  state.busy = on;
  els.btnNext.disabled = on;
  els.btnBack.disabled = on;
  els.btnNextSpin.classList.toggle("hidden", !on);
  els.btnNextLabel.textContent = on ? "Running …" : (state.step === 2 ? "⚡ Run Analysis" : els.btnNextLabel.textContent);
}

// ─── Rendering ─────────────────────────────────────────────────
function renderResult(r) {
  // Image
  els.annotatedImg.src = `data:image/png;base64,${r.annotated_b64}`;
  els.originalImg.src = r.original_url;
  if (r.gdino_caption) {
    els.dinoCaption.textContent = `Caption sent to DINO: "${r.gdino_caption}"`;
    els.dinoCaption.classList.remove("hidden");
  } else {
    els.dinoCaption.classList.add("hidden");
  }
  switchTab("annotated");

  // Answer card
  els.answerText.textContent = r.answer;
  const chips = [];
  chips.push(`<div class="meta-chip">Confidence <b>${(r.confidence*100).toFixed(1)}% — ${confLabel(r.confidence)}</b></div>`);
  chips.push(`<div class="meta-chip">BLIP-2 <b>${r.inference_ms} ms</b></div>`);
  if (r.grounding_ms != null) chips.push(`<div class="meta-chip">GDino <b>${r.grounding_ms} ms</b></div>`);
  chips.push(`<div class="meta-chip">Device <b>${(r.device || "?").toUpperCase()}</b></div>`);
  els.metaRow.innerHTML = chips.join("");

  const color = confColor(r.confidence);
  els.confPct.textContent = `${(r.confidence*100).toFixed(1)}%`;
  els.confPct.style.color = color;
  els.confFill.style.background = color;
  els.confFill.style.width = "0%";
  requestAnimationFrame(() => { els.confFill.style.width = `${r.confidence*100}%`; });

  // Re-trigger card animation
  els.answerCard.style.animation = "none"; void els.answerCard.offsetWidth; els.answerCard.style.animation = "";

  // Grounding boxes
  if (r.grounding_ran) {
    els.groundSection.classList.remove("hidden");
    const n = r.boxes.length;
    els.groundCount.textContent = `${n} object${n === 1 ? "" : "s"} detected`;
    if (n) {
      els.boxList.innerHTML = r.boxes.map((b, i) => {
        const col = BOX_PALETTE[i % BOX_PALETTE.length];
        return `<div class="box-row">
          <div class="box-dot" style="background:${col}"></div>
          <span class="box-label">${escapeHtml(b.label)}</span>
          <span class="box-conf" style="color:${col}">${Math.round(b.confidence*100)}%</span>
          <span class="box-coords">[${b.x0},${b.y0}→${b.x1},${b.y1}]</span>
        </div>`;
      }).join("");
    } else {
      els.boxList.innerHTML = `<div class="no-boxes">No objects detected above confidence threshold</div>`;
    }
  } else {
    els.groundSection.classList.add("hidden");
  }

  renderHistory();
}

function renderHistory() {
  els.histCount.textContent = `(${state.history.length})`;
  els.clearBtn.classList.toggle("hidden", state.history.length === 0);

  if (state.history.length === 0) {
    els.historyList.innerHTML = `<div class="empty-inline">Q&amp;A history will appear here</div>`;
    return;
  }

  els.historyList.innerHTML = state.history.map((h, idx) => {
    const c = h.confidence;
    const tc = TYPE_COLORS[h.qtype] || "#94a3b8";
    const bc = confColor(c);
    const groundTag = h.grounding_ran
      ? `<span class="hist-ground">🎯 ${h.boxes.length} box${h.boxes.length === 1 ? "" : "es"}</span>`
      : "";
    return `<div class="hist-item" data-idx="${idx}">
      <div class="hist-q">❓ ${escapeHtml(h.question)}</div>
      <div class="hist-a">${escapeHtml(h.answer)}</div>
      <div class="hist-footer">
        <span class="hist-badge" style="background:${tc}26;color:${tc};border:1px solid ${tc}66">${h.qtype}</span>
        ${groundTag}
        <span class="hist-meta" style="color:${bc}">${Math.round(c*100)}% conf</span>
        <span class="hist-meta">${h.timestamp} · ${h.inference_ms}ms</span>
      </div>
    </div>`;
  }).join("");
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.toggle("active", p.dataset.panelTab === name));
}

// ─── Event wiring ──────────────────────────────────────────────
function wire() {
  // Dropzone
  els.fileInput.addEventListener("change", (e) => setFile(e.target.files[0] || null));
  els.dzClear.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); clearFile(); });

  ["dragenter", "dragover"].forEach(evt =>
    els.dropzone.addEventListener(evt, (e) => {
      e.preventDefault(); e.stopPropagation();
      els.dropzone.classList.add("dragging");
    })
  );
  ["dragleave", "drop"].forEach(evt =>
    els.dropzone.addEventListener(evt, (e) => {
      e.preventDefault(); e.stopPropagation();
      els.dropzone.classList.remove("dragging");
    })
  );
  els.dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (f) setFile(f);
  });
  // Paste support
  window.addEventListener("paste", (e) => {
    const f = [...(e.clipboardData?.files || [])][0];
    if (f) { setFile(f); toast("Image pasted from clipboard."); }
  });

  // Type selector
  els.typeSelect.addEventListener("change", () => {
    state.selectedType = els.typeSelect.value;
    const preset = QUESTION_TYPES[state.selectedType];
    els.question.value = preset;
    state.lastPreset = preset;
    applyTypeUI();
    updateActionBar();
  });

  // Question textarea
  els.question.addEventListener("input", () => {
    if (els.question.value !== state.lastPreset && state.selectedType !== "Custom") {
      state.selectedType = "Custom";
      els.typeSelect.value = "Custom";
      applyTypeUI();
    }
    updateActionBar();
  });

  // Cmd/Ctrl + Enter to advance
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      if (!els.btnNext.disabled) nextAction();
    }
  });

  // Action bar
  els.btnNext.addEventListener("click", nextAction);
  els.btnBack.addEventListener("click", backAction);

  // Stepper click — allow jumping to completed/active steps
  els.stepper.addEventListener("click", (e) => {
    const step = e.target.closest(".step");
    if (!step) return;
    const n = parseInt(step.dataset.step, 10);
    // Can jump back freely; can't jump forward past current
    if (n < state.step) goToStep(n, "back");
    else if (n === state.step) return;
    else if (n === state.step + 1) nextAction();
    // skip-ahead beyond +1 not allowed
  });

  // Tabs
  document.querySelectorAll(".tab").forEach(t =>
    t.addEventListener("click", () => switchTab(t.dataset.tab))
  );

  // Clear history
  els.clearBtn.addEventListener("click", () => {
    state.history = [];
    renderHistory();
  });

  // History click — re-show that result
  els.historyList.addEventListener("click", (e) => {
    const item = e.target.closest(".hist-item");
    if (!item) return;
    const idx = parseInt(item.dataset.idx, 10);
    const r = state.history[idx];
    if (r) {
      state.lastResult = r;
      renderResult(r);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  });
}

// ─── Boot ──────────────────────────────────────────────────────
buildTypeSelect();
applyTypeUI();
wire();
checkHealth();
setInterval(checkHealth, 8000);
updateActionBar();
