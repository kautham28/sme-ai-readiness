"""
Train final Logistic Regression model and export browser-ready parameters.
Mirrors the notebook pipeline: scale → priority weight → KMeans labels →
group split → SMOTE → tuned Logistic Regression.
"""
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Merged_Employee_Manager_Factor_Scores.xlsx"
OUT_DIR = ROOT / "website" / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

FACTOR_COLS = [
    "E1 PerceivedUsefulness",
    "E2 PerceivedEaseOfUse",
    "E3 TechnicalLiteracy",
    "E5 TrustInAI",
    "E6 FearOfJobLoss",
    "E7 Complexity",
    "E8 JobRelevance",
    "E9 SocioTechnicalAlignment",
    "E10 UserParticipation",
    "M1 TopMgmtSupport",
    "M2 FinancialCapability",
    "M3 ITInfrastructure",
    "M4 DataQualityStrategy",
    "M5 InnovationCulture",
    "M6 CompetitivePressure",
    "M8 RegulatoryEnvironment",
    "M9 VendorReadiness",
]

PRIORITY_COLS = [
    "E2 PerceivedEaseOfUse",
    "E3 TechnicalLiteracy",
    "E7 Complexity",
    "M2 FinancialCapability",
    "M3 ITInfrastructure",
    "M4 DataQualityStrategy",
]
PRIORITY_WEIGHT = 1.6

# Friendly labels + short help text for the SME owner UI
FACTOR_META = {
    "E1 PerceivedUsefulness": {
        "code": "E1",
        "label": "Perceived Usefulness",
        "group": "Employee",
        "help": "How useful employees believe AI will be for their work",
    },
    "E2 PerceivedEaseOfUse": {
        "code": "E2",
        "label": "Perceived Ease of Use",
        "group": "Employee",
        "help": "How easy employees expect AI tools to be to learn and use",
    },
    "E3 TechnicalLiteracy": {
        "code": "E3",
        "label": "Technical Literacy",
        "group": "Employee",
        "help": "Employees' digital / technical skills to work with AI",
    },
    "E5 TrustInAI": {
        "code": "E5",
        "label": "Trust in AI",
        "group": "Employee",
        "help": "Confidence that AI recommendations and outputs are reliable",
    },
    "E6 FearOfJobLoss": {
        "code": "E6",
        "label": "Fear of Job Loss",
        "group": "Employee",
        "help": "Concern that AI may replace roles (higher = more fear)",
    },
    "E7 Complexity": {
        "code": "E7",
        "label": "Perceived Complexity",
        "group": "Employee",
        "help": "How complex AI systems feel to employees (higher = more complex)",
    },
    "E8 JobRelevance": {
        "code": "E8",
        "label": "Job Relevance",
        "group": "Employee",
        "help": "How relevant AI is to day-to-day job tasks",
    },
    "E9 SocioTechnicalAlignment": {
        "code": "E9",
        "label": "Socio-Technical Alignment",
        "group": "Employee",
        "help": "Fit between AI tools, workflows, and team practices",
    },
    "E10 UserParticipation": {
        "code": "E10",
        "label": "User Participation",
        "group": "Employee",
        "help": "How much employees are involved in AI decisions and rollout",
    },
    "M1 TopMgmtSupport": {
        "code": "M1",
        "label": "Top Management Support",
        "group": "Managerial",
        "help": "Leadership commitment and sponsorship for AI adoption",
    },
    "M2 FinancialCapability": {
        "code": "M2",
        "label": "Financial Capability",
        "group": "Managerial",
        "help": "Budget and resources available to invest in AI",
    },
    "M3 ITInfrastructure": {
        "code": "M3",
        "label": "IT Infrastructure",
        "group": "Managerial",
        "help": "Hardware, software, and connectivity readiness",
    },
    "M4 DataQualityStrategy": {
        "code": "M4",
        "label": "Data Quality Strategy",
        "group": "Managerial",
        "help": "Quality, governance, and readiness of organizational data",
    },
    "M5 InnovationCulture": {
        "code": "M5",
        "label": "Innovation Culture",
        "group": "Managerial",
        "help": "Openness to experimentation and continuous improvement",
    },
    "M6 CompetitivePressure": {
        "code": "M6",
        "label": "Competitive Pressure",
        "group": "Managerial",
        "help": "Market pressure to adopt AI to stay competitive",
    },
    "M8 RegulatoryEnvironment": {
        "code": "M8",
        "label": "Regulatory Environment",
        "group": "Managerial",
        "help": "Awareness and readiness around AI-related regulations",
    },
    "M9 VendorReadiness": {
        "code": "M9",
        "label": "Vendor Readiness",
        "group": "Managerial",
        "help": "Access to capable AI vendors / implementation partners",
    },
}


def transform_features(raw: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    scaled = scaler.transform(raw)
    priority_idx = [FACTOR_COLS.index(c) for c in PRIORITY_COLS]
    weighted = scaled.copy()
    weighted[:, priority_idx] *= PRIORITY_WEIGHT
    return weighted


def main():
    df = pd.read_excel(DATA_PATH, sheet_name="Merged_Data")
    X_raw = df[FACTOR_COLS].astype(float).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_weighted = X_scaled.copy()
    priority_idx = [FACTOR_COLS.index(c) for c in PRIORITY_COLS]
    X_weighted[:, priority_idx] *= PRIORITY_WEIGHT

    # K-Means → Low / Medium / High labels (same as notebook)
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    cluster_id = kmeans.fit_predict(X_weighted)
    overall = X_raw.mean(axis=1)
    cluster_order = (
        pd.Series(overall.values, index=df.index)
        .groupby(cluster_id)
        .mean()
        .sort_values()
        .index.tolist()
    )
    label_map = {
        cluster_order[0]: "Low",
        cluster_order[1]: "Medium",
        cluster_order[2]: "High",
    }
    y_labels = np.array([label_map[c] for c in cluster_id])

    le = LabelEncoder()
    # Force Low < Medium < High order in encoding by fitting in that order
    le.fit(["Low", "Medium", "High"])
    y = le.transform(y_labels)
    class_names = list(le.classes_)
    groups = df["SME_Name"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_weighted, y, groups=groups))
    X_train, X_test = X_weighted[train_idx], X_weighted[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_counts = Counter(y_train)
    imbalance_ratio = max(train_counts.values()) / min(train_counts.values())
    if imbalance_ratio > 1.5:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"SMOTE applied. {train_counts} -> {Counter(y_train_res)}")
    else:
        X_train_res, y_train_res = X_train, y_train
        print("SMOTE not applied.")

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
        n_iter=40,
        scoring="accuracy",
        cv=inner_cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train_res, y_train_res)
    model = search.best_estimator_
    print("Best params:", search.best_params_)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    print(
        "Test Acc:",
        round(accuracy_score(y_test, y_pred), 4),
        "| F1:",
        round(f1_score(y_test, y_pred, average="weighted"), 4),
        "| Prec:",
        round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "| Rec:",
        round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "| ROC-AUC:",
        round(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"), 4),
    )
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # Background mean in transformed space (for Linear SHAP)
    feature_means = X_train_res.mean(axis=0).tolist()
    base_logits = (np.array(feature_means) @ model.coef_.T + model.intercept_).tolist()

    payload = {
        "model_name": "Logistic Regression",
        "version": "1.0.0",
        "score_min": 1.0,
        "score_max": 5.0,
        "score_step": 0.5,
        "priority_weight": PRIORITY_WEIGHT,
        "priority_factors": PRIORITY_COLS,
        "feature_cols": FACTOR_COLS,
        "factors": [
            {
                **FACTOR_META[col],
                "key": col,
                "priority": col in PRIORITY_COLS,
            }
            for col in FACTOR_COLS
        ],
        "class_names": class_names,
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "logistic_regression": {
            "C": float(model.C),
            "coefficients": model.coef_.tolist(),  # shape: [n_classes, n_features]
            "intercepts": model.intercept_.tolist(),
            "classes": model.classes_.tolist(),
        },
        "shap": {
            "method": "linear_exact",
            "feature_means": feature_means,
            "base_logits": base_logits,
            "note": (
                "Exact Linear SHAP in log-odds space: "
                "phi_i,c = coef[c,i] * (x_i - E[x_i]). "
                "Advantages/disadvantages use contributions toward High readiness."
            ),
        },
        "metrics": {
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
            "best_params": {k: (None if v is None else v) for k, v in search.best_params_.items()},
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_smes": int(df["SME_Name"].nunique()),
        },
        "defaults": {col: round(float(X_raw[col].mean()), 1) for col in FACTOR_COLS},
    }

    out_path = OUT_DIR / "model.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")

    # Sanity: verify JS-equivalent prediction matches sklearn
    sample = X_raw.iloc[0].values.reshape(1, -1)
    xt = transform_features(sample, scaler)
    logits = xt @ model.coef_.T + model.intercept_
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba_manual = exp / exp.sum(axis=1, keepdims=True)
    proba_sk = model.predict_proba(xt)
    assert np.allclose(proba_manual, proba_sk, atol=1e-9)
    print("Manual softmax matches sklearn.predict_proba OK")


if __name__ == "__main__":
    main()
