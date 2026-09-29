/**
 * Gemini client for SME advice based on SHAP analysis.
 * Configure via window.GEMINI_API_KEY and window.GEMINI_MODEL (see config.js).
 */

const GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models";

function getGeminiConfig() {
  const apiKey = window.GEMINI_API_KEY || "";
  const model = window.GEMINI_MODEL || "gemini-3.5-flash";
  const fallbacks = Array.isArray(window.GEMINI_FALLBACK_MODELS)
    ? window.GEMINI_FALLBACK_MODELS
    : [];
  return { apiKey, model, fallbacks };
}

const REQUEST_TIMEOUT_MS = 45000;

function isRetryableGeminiError(status, message) {
  const msg = String(message || "").toLowerCase();
  return (
    [404, 429, 500, 502, 503, 504].includes(status) ||
    msg.includes("high demand") ||
    msg.includes("resource exhausted") ||
    msg.includes("unavailable") ||
    msg.includes("try again")
  );
}

function buildAdvicePrompt({ smeName, predicted, probabilities, classNames, advantages, disadvantages, allFactors }) {
  const org = smeName || "this SME";
  const probs = classNames
    .map((name, i) => `${name}: ${(probabilities[i] * 100).toFixed(1)}%`)
    .join(", ");

  const fmtFactor = (f) =>
    `- ${f.code} ${f.label}: score ${f.score.toFixed(1)}/5, SHAP ${f.shap >= 0 ? "+" : ""}${f.shap.toFixed(3)}${f.priority ? " (priority factor)" : ""}`;

  const advText = advantages.length
    ? advantages.map(fmtFactor).join("\n")
    : "(none)";
  const disText = disadvantages.length
    ? disadvantages.map(fmtFactor).join("\n")
    : "(none)";
  const allText = allFactors
    .slice()
    .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
    .map(fmtFactor)
    .join("\n");

  return `You are an AI adoption advisor for small and medium enterprises (SMEs).
A new SME owner just completed an AI Adoption Readiness assessment.

Organization: ${org}
Predicted readiness level: ${predicted}
Class probabilities: ${probs}

SHAP analysis explains which factors currently help or hurt movement toward HIGH readiness.
Positive SHAP = advantage toward High readiness. Negative SHAP = disadvantage holding readiness back.
Scores are on a 1–5 Likert scale. Priority factors are weighted more heavily in the model.

TOP ADVANTAGES (leverage these):
${advText}

TOP DISADVANTAGES (fix/improve these first):
${disText}

ALL FACTOR CONTRIBUTIONS (sorted by |SHAP|):
${allText}

Write practical advice for the SME owner. Requirements:
1) Start with a short plain-language summary of what the readiness level means for them.
2) Give 3–5 concrete actions to strengthen the disadvantage factors (most important first).
3) Give 2–3 actions to protect and amplify the advantage factors.
4) End with a 30–60 day starter plan (bullet list).
5) Keep language clear, non-technical, and actionable for a busy SME owner.
6) Do not invent scores. Base every recommendation on the SHAP factors and scores above.
7) Use markdown with short headings and bullets. No tables. Keep total length under 450 words.`;
}

function buildGenerationConfig(model) {
  // Thinking tokens count toward maxOutputTokens, so the cap must leave room for the answer.
  const config = { temperature: 0.6, maxOutputTokens: 8192 };
  if (/^gemini-3/.test(model)) {
    config.thinkingConfig = { thinkingLevel: "low" };
  } else if (/^gemini-2\.5/.test(model)) {
    config.thinkingConfig = { thinkingBudget: 512 };
  }
  return config;
}

async function callGeminiOnce(apiKey, model, prompt) {
  const url = `${GEMINI_BASE}/${encodeURIComponent(model)}:generateContent`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let res;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": apiKey,
      },
      body: JSON.stringify({
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: buildGenerationConfig(model),
      }),
      signal: controller.signal,
    });
  } catch (networkErr) {
    const err = new Error(
      networkErr.name === "AbortError"
        ? `${model} took too long to respond.`
        : `Network error while contacting Gemini (${networkErr.message}).`
    );
    err.retryable = true;
    throw err;
  } finally {
    clearTimeout(timer);
  }

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg =
      data?.error?.message ||
      `Gemini request failed (${res.status}). Check the API key and model name.`;
    const err = new Error(msg);
    err.status = res.status;
    err.retryable = isRetryableGeminiError(res.status, msg);
    throw err;
  }

  const candidate = data?.candidates?.[0];
  const text = (candidate?.content?.parts || [])
    .filter((p) => !p.thought)
    .map((p) => p.text || "")
    .join("")
    .trim();

  if (!text) {
    const reason = candidate?.finishReason || data?.promptFeedback?.blockReason;
    const err = new Error(
      reason
        ? `Gemini returned no advice (reason: ${reason}). Try again.`
        : "Gemini returned an empty response. Try again."
    );
    err.retryable = true;
    throw err;
  }
  return text;
}

/**
 * Calls Gemini generateContent and returns plain text advice.
 * Retries with fallback models when Google reports high demand / overload.
 */
export async function fetchGeminiAdvice(context, onAttempt = () => {}) {
  const { apiKey, model, fallbacks } = getGeminiConfig();
  if (!apiKey) {
    throw new Error("Gemini API key is missing. Set window.GEMINI_API_KEY in config.local.js.");
  }

  const prompt = buildAdvicePrompt(context);
  const models = [model, ...fallbacks.filter((m) => m && m !== model)];
  let lastError = null;

  for (let i = 0; i < models.length; i++) {
    onAttempt(models[i], i);
    try {
      return await callGeminiOnce(apiKey, models[i], prompt);
    } catch (err) {
      lastError = err;
      console.warn(`Gemini model ${models[i]} failed:`, err.message);
      if (!err.retryable || i === models.length - 1) break;
      await new Promise((r) => setTimeout(r, 400));
    }
  }

  if (lastError?.retryable) {
    throw new Error(
      "All Gemini models are busy right now (Google reports high demand). Please wait a minute and click Regenerate."
    );
  }
  throw lastError || new Error("Gemini request failed.");
}

/** Minimal markdown → HTML (headings, bold, lists, paragraphs). */
export function markdownToHtml(md) {
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let inList = null;

  const closeList = () => {
    if (inList) {
      html.push(`</${inList}>`);
      inList = null;
    }
  };

  const openList = (tag) => {
    if (inList !== tag) {
      closeList();
      html.push(`<${tag}>`);
      inList = tag;
    }
  };

  const inline = (s) =>
    s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>");

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) {
      closeList();
      continue;
    }
    if (/^###\s+/.test(line)) {
      closeList();
      html.push(`<h4>${inline(line.replace(/^###\s+/, ""))}</h4>`);
      continue;
    }
    if (/^##\s+/.test(line)) {
      closeList();
      html.push(`<h3>${inline(line.replace(/^##\s+/, ""))}</h3>`);
      continue;
    }
    if (/^#\s+/.test(line)) {
      closeList();
      html.push(`<h3>${inline(line.replace(/^#\s+/, ""))}</h3>`);
      continue;
    }
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(line)) {
      closeList();
      html.push("<hr />");
      continue;
    }
    if (/^[-*]\s+/.test(line)) {
      openList("ul");
      html.push(`<li>${inline(line.replace(/^[-*]\s+/, ""))}</li>`);
      continue;
    }
    if (/^\d+[.)]\s+/.test(line)) {
      openList("ol");
      html.push(`<li>${inline(line.replace(/^\d+[.)]\s+/, ""))}</li>`);
      continue;
    }
    if (inList && /^\s+/.test(raw)) {
      html.push(`<li class="md-cont">${inline(line)}</li>`);
      continue;
    }
    closeList();
    html.push(`<p>${inline(line)}</p>`);
  }
  closeList();
  return html.join("\n");
}
