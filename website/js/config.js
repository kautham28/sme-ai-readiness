// Shared defaults (safe to commit). Put your real key in config.local.js (gitignored).
window.GEMINI_API_KEY = window.GEMINI_API_KEY || "";
window.GEMINI_MODEL = window.GEMINI_MODEL || "gemini-3.5-flash";
window.GEMINI_FALLBACK_MODELS = window.GEMINI_FALLBACK_MODELS || [
  "gemini-3.8-flash",
  "gemini-3.1-flash-lite",
  "gemini-2.5-flash",
];
