from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from sport_agent_bridge import run_infer as run_sport_infer
from study_agent_bridge import run_infer as run_study_infer


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
OUT_NEXT_STUDY_DIR = ROOT / "outputs_next" / "study"


def feature_cols(df: pd.DataFrame, excluded: list[str]) -> list[str]:
    return [col for col in df.columns if col not in excluded]


def score_study() -> Path:
    return run_study_infer(out_dir=OUT_NEXT_STUDY_DIR)


def score_life() -> Path:
    data = pd.read_csv(OUT_DIR / "life" / "feature_dataset.csv")
    model = joblib.load(OUT_DIR / "life" / "models" / "best_life_model.joblib")
    metrics = json.loads((OUT_DIR / "life" / "metrics.json").read_text(encoding="utf-8"))

    id_cols = ["sid", "term_id", "year_start", "semester", "term_order"]
    label_cols = ["club_active_flag", "club_event_count"]
    feats = feature_cols(data, id_cols + label_cols)

    out = data[["sid", "term_id", "club_active_flag"]].copy()
    proba = model.predict_proba(data[feats])[:, 1]
    threshold = float(metrics.get("best_threshold", 0.5))
    out["pred_prob"] = proba
    out["pred_label"] = (proba >= threshold).astype(int)
    target = OUT_DIR / "life" / "predictions_full.csv"
    out.to_csv(target, index=False)
    return target


def score_sport() -> Path:
    try:
        return run_sport_infer(out_dir=OUT_DIR / "sport")
    except Exception:
        pass

    data = pd.read_csv(OUT_DIR / "sport" / "feature_dataset.csv")
    model = joblib.load(OUT_DIR / "sport" / "regression" / "best_sport_regression_model.joblib")

    id_cols = ["sid", "term_id", "year_start", "semester", "term_order"]
    label_cols = ["zf_score", "zf_grade"]
    feats = feature_cols(data, id_cols + label_cols)

    out = data[["sid", "term_id", "zf_score"]].copy()
    out["pred_zf_score"] = model.predict(data[feats])
    # Keep risk field for downstream scripts that can directly consume probabilities.
    scores = pd.to_numeric(out["pred_zf_score"], errors="coerce")
    if scores.notna().any():
        low = float(scores.min())
        high = float(scores.max())
        if high > low:
            out["pred_fail_proba"] = 1 - (scores - low) / (high - low)
        else:
            out["pred_fail_proba"] = 0.5
    else:
        out["pred_fail_proba"] = 0.5
    out["best_cls_model"] = "SportLegacy"
    target = OUT_DIR / "sport" / "predictions_full.csv"
    out.to_csv(target, index=False)
    return target


def main() -> None:
    outputs = {
        "study": str(score_study()),
        "life": str(score_life()),
        "sport": str(score_sport()),
    }
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
