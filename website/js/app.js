/**
 * Client-side Logistic Regression + exact Linear SHAP + Gemini advice.
 */

import { fetchGeminiAdvice, markdownToHtml } from "./gemini.js";

const TOP_N = 5;
let lastAdviceContext = null;

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function transformFeatures(rawScores, model) {
  const { mean, scale } = model.scaler;
  const prioritySet = new Set(model.priority_factors);
  const weight = model.priority_weight;
  return model.feature_cols.map((col, i) => {
    let z = (rawScores[i] - mean[i]) / scale[i];
    if (prioritySet.has(col)) z *= weight;
    return z;
  });
}

function predictProba(x, model) {
  const { coefficients, intercepts } = model.logistic_regression;
  const logits = coefficients.map(
    (coef, c) => intercepts[c] + coef.reduce((s, w, i) => s + w * x[i], 0)
  );
  return { logits, probabilities: softmax(logits) };
}

/**
 * Exact Linear SHAP in log-odds space for every class:
 * phi[c][i] = coef[c][i] * (x[i] - E[x[i]])
 */
function linearShap(x, model) {
  const means = model.shap.feature_means;
  const { coefficients } = model.logistic_regression;
  return coefficients.map((coef) =>
    coef.map((w, i) => w * (x[i] - means[i]))
  );
}

function pct(v) {
  return `${(v * 100).toFixed(1)}%`;
}

function fmtShap(v) {
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(3)}`;
}

async function loadModel() {
  const res = await fetch("assets/model.json");
  if (!res.ok) throw new Error("Could not load model.json");
  return res.json();
}

function buildFactorUI(model) {
  const employeeRoot = document.getElementById("employee-factors");
  const managerRoot = document.getElementById("manager-factors");
  employeeRoot.innerHTML = "";
  managerRoot.innerHTML = "";

  model.factors.forEach((factor, idx) => {
    const value = model.defaults[factor.key] ?? 3;
    const item = document.createElement("div");
    item.className = "factor-item";
    item.innerHTML = `
      <div class="factor-top">
        <div class="factor-label">
          <span class="factor-code">${factor.code}</span>${factor.label}
          ${factor.priority ? '<span class="priority-pill">Priority ×1.6</span>' : ""}
        </div>
        <div class="factor-value" data-value-for="${idx}">${Number(value).toFixed(1)}</div>
      </div>
      <p class="factor-help">${factor.help}</p>
      <input
        type="range"
        min="${model.score_min}"
        max="${model.score_max}"
        step="${model.score_step}"
        value="${value}"
        data-factor-idx="${idx}"
        aria-label="${factor.label}"
      />
      <div class="scale-ends"><span>1 Low</span><span>5 High</span></div>
    `;
    (factor.group === "Employee" ? employeeRoot : managerRoot).appendChild(item);
  });

  document.querySelectorAll('input[type="range"][data-factor-idx]').forEach((input) => {
    input.addEventListener("input", () => {
      const idx = input.dataset.factorIdx;
      const label = document.querySelector(`[data-value-for="${idx}"]`);
      if (label) label.textContent = Number(input.value).toFixed(1);
    });
  });
}

function readScores(model) {
  return model.feature_cols.map((_, idx) => {
    const input = document.querySelector(`input[data-factor-idx="${idx}"]`);
    return Number(input.value);
  });
}

function setScores(model, scoreMap) {
  model.feature_cols.forEach((key, idx) => {
    const input = document.querySelector(`input[data-factor-idx="${idx}"]`);
    const v = scoreMap[key] ?? model.defaults[key] ?? 3;
    input.value = v;
    const label = document.querySelector(`[data-value-for="${idx}"]`);
    if (label) label.textContent = Number(v).toFixed(1);
  });
}

function renderShapList(el, items, emptyText) {
  el.innerHTML = "";
  if (!items.length) {
    el.innerHTML = `<li><div class="shap-meta"><strong>${emptyText}</strong></div></li>`;
    return;
  }
  items.forEach((item, i) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <span class="rank">${i + 1}</span>
      <div class="shap-meta">
        <strong>${item.code} · ${item.label}</strong>
        <span>Your score: ${item.score.toFixed(1)} / 5${item.priority ? " · priority factor" : ""}</span>
      </div>
      <span class="shap-score">${fmtShap(item.shap)}</span>
    `;
    el.appendChild(li);
  });
}

function setAdviceLoading() {
  const body = document.getElementById("advice-body");
  const btn = document.getElementById("btn-refresh-advice");
  btn.hidden = true;
  body.innerHTML = `
    <div class="advice-loading">
      <span class="advice-spinner" aria-hidden="true"></span>
      <p>Gemini is writing advice from your SHAP results…</p>
    </div>
  `;
}

function setAdviceError(message) {
  const body = document.getElementById("advice-body");
  const btn = document.getElementById("btn-refresh-advice");
  btn.hidden = false;
  const safe = String(message)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  body.innerHTML = `
    <div class="advice-error">
      <p><strong>Could not generate advice.</strong></p>
      <p>${safe}</p>
      <p class="advice-hint">If this keeps happening, wait a minute and click Regenerate.</p>
    </div>
  `;
}

function setAdviceContent(markdown) {
  const body = document.getElementById("advice-body");
  const btn = document.getElementById("btn-refresh-advice");
  btn.hidden = false;
  body.innerHTML = `<div class="advice-content">${markdownToHtml(markdown)}</div>`;
}

async function loadAdvice(context) {
  lastAdviceContext = context;
  setAdviceLoading();
  try {
    const text = await fetchGeminiAdvice(context, (model, attempt) => {
      const status = document.querySelector("#advice-body .advice-loading p");
      if (status && attempt > 0) {
        status.textContent = `Main model is busy, trying backup model (${model})…`;
      }
    });
    setAdviceContent(text);
  } catch (err) {
    console.error(err);
    setAdviceError(err.message || "Unknown Gemini error.");
  }
}

function renderResults(model, rawScores, prediction) {
  const results = document.getElementById("results");
  const body = document.getElementById("results-body");
  const placeholder = document.getElementById("results-placeholder");
  results.classList.remove("is-empty");
  placeholder.hidden = true;
  body.hidden = false;

  const smeName = document.getElementById("sme-name").value.trim();
  const levelEl = document.getElementById("pred-level");
  levelEl.textContent = prediction.predicted;
  levelEl.dataset.level = prediction.predicted;
  document.getElementById("pred-org").textContent = smeName
    ? `Assessment for ${smeName}`
    : "Assessment for your SME";
  document.getElementById("results-sub").textContent =
    "Readiness, SHAP drivers, and Gemini recommendations for your SME.";

  const bars = document.getElementById("prob-bars");
  bars.innerHTML = "";
  const displayOrder = ["Low", "Medium", "High"].filter((n) =>
    model.class_names.includes(n)
  );
  displayOrder.forEach((name) => {
    const i = model.class_names.indexOf(name);
    const row = document.createElement("div");
    row.className = "prob-row";
    const cls = name.toLowerCase();
    row.innerHTML = `
      <span>${name}</span>
      <div class="prob-track"><div class="prob-fill ${cls}" style="width:0%"></div></div>
      <span>${pct(prediction.probabilities[i])}</span>
    `;
    bars.appendChild(row);
    requestAnimationFrame(() => {
      row.querySelector(".prob-fill").style.width = pct(prediction.probabilities[i]);
    });
  });

  const highIdx = model.class_names.indexOf("High");
  const shapHigh = prediction.shapByClass[highIdx];
  const enriched = model.factors.map((f, i) => ({
    ...f,
    score: rawScores[i],
    shap: shapHigh[i],
  }));

  const advantages = enriched
    .filter((f) => f.shap > 0)
    .sort((a, b) => b.shap - a.shap)
    .slice(0, TOP_N);
  const disadvantages = enriched
    .filter((f) => f.shap < 0)
    .sort((a, b) => a.shap - b.shap)
    .slice(0, TOP_N);

  renderShapList(
    document.getElementById("adv-list"),
    advantages,
    "No positive High-readiness drivers for this profile."
  );
  renderShapList(
    document.getElementById("dis-list"),
    disadvantages,
    "No negative High-readiness drivers for this profile."
  );

  const maxAbs = Math.max(...enriched.map((f) => Math.abs(f.shap)), 1e-9);
  const waterfall = document.getElementById("waterfall");
  waterfall.innerHTML = "";
  [...enriched]
    .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
    .forEach((f) => {
      const widthPct = (Math.abs(f.shap) / maxAbs) * 50;
      const pos = f.shap >= 0;
      const row = document.createElement("div");
      row.className = "wf-row";
      row.innerHTML = `
        <div class="wf-label">${f.code} ${f.label}</div>
        <div class="wf-track">
          <span class="wf-mid"></span>
          <span class="wf-fill ${pos ? "pos" : "neg"}" style="width:0%"></span>
        </div>
        <div class="wf-val" style="color:${pos ? "var(--advantage)" : "var(--disadvantage)"}">${fmtShap(f.shap)}</div>
      `;
      waterfall.appendChild(row);
      requestAnimationFrame(() => {
        row.querySelector(".wf-fill").style.width = `${widthPct}%`;
      });
    });

  results.scrollIntoView({ behavior: "smooth", block: "start" });

  loadAdvice({
    smeName,
    predicted: prediction.predicted,
    probabilities: prediction.probabilities,
    classNames: model.class_names,
    advantages,
    disadvantages,
    allFactors: enriched,
  });
}

function runPrediction(model) {
  const raw = readScores(model);
  const x = transformFeatures(raw, model);
  const { probabilities } = predictProba(x, model);
  const shapByClass = linearShap(x, model);
  const predictedIdx = probabilities.indexOf(Math.max(...probabilities));
  renderResults(model, raw, {
    probabilities,
    predicted: model.class_names[predictedIdx],
    shapByClass,
  });
}

async function main() {
  const model = await loadModel();
  window.__SME_MODEL__ = model;
  buildFactorUI(model);

  document.getElementById("btn-predict").addEventListener("click", () => runPrediction(model));
  document.getElementById("btn-reset").addEventListener("click", () => setScores(model, model.defaults));
  document.getElementById("btn-fill-average").addEventListener("click", () => {
    setScores(model, model.defaults);
    document.getElementById("assess").scrollIntoView({ behavior: "smooth" });
  });
  document.getElementById("btn-refresh-advice").addEventListener("click", () => {
    if (lastAdviceContext) loadAdvice(lastAdviceContext);
  });
}

main().catch((err) => {
  console.error(err);
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<div style="padding:1rem;background:#f8e8e4;color:#a33b2b;font-family:sans-serif">
      Failed to load the model. Ensure <code>assets/model.json</code> is deployed with the site.
    </div>`
  );
});
