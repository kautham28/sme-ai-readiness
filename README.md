# SME AI Adoption Readiness — Netlify site

Static website (no backend) that scores an SME across **17 factors** using the final **Logistic Regression** model, then shows:

- AI readiness level: **Low / Medium / High**
- Class probabilities
- **SHAP advantages & disadvantages** (exact Linear SHAP toward High readiness)

Everything runs in the browser from `website/assets/model.json`.

## Quick deploy to Netlify

### Option A — Drag & drop (fastest)

1. Open [https://app.netlify.com/drop](https://app.netlify.com/drop)
2. Drag the **`website`** folder onto the page
3. Netlify gives you a live URL

### Option B — Git-connected site

1. Push this project to GitHub / GitLab / Bitbucket
2. In Netlify: **Add new site → Import an existing project**
3. Build settings (already in root `netlify.toml`):
   - **Publish directory:** `website`
   - **Build command:** leave empty
4. Deploy

### Option C — Netlify CLI

```bash
npm install -g netlify-cli
cd "e:/Research/model/Final model"
netlify login
netlify deploy --dir=website --prod
```

## Local preview

Any static server from the `website` folder:

```bash
cd website
npx --yes serve .
# or: python -m http.server 8080
```

Open the printed URL (do not open `index.html` via `file://` — `fetch` of `model.json` needs HTTP).

## Retrain / refresh the model

From the project root (`Final model`):

```bash
pip install pandas scikit-learn imbalanced-learn openpyxl numpy
python train_export_lr.py
```

This overwrites `website/assets/model.json`. Redeploy the `website` folder.

## Score guide

Each factor is rated **1 (very low) → 5 (very high)**.

Priority factors (weighted ×1.6 after standardization): E2, E3, E7, M2, M3, M4.

## Privacy

No scores are sent to a server. Inference and SHAP run entirely in the visitor’s browser.
