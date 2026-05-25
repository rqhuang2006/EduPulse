from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from study_common import DM_DIR, ensure_dirs, model_params, write_csv


def score(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return np.asarray(model.predict(x), dtype=float)


def score_table(df: pd.DataFrame, bundle: dict, source: str) -> pd.DataFrame:
    config = bundle["config"]
    features = config.get("feature_columns", [])
    x = df.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    fallback_used = False
    try:
        prob = score(bundle["primary_model"], x)
    except Exception:
        prob = score(bundle["fallback_model"], x)
        fallback_used = True
    coverage_threshold = float(model_params().get("prediction", {}).get("min_feature_coverage", 0.3))
    low_coverage = df.get("DATA_QUALITY_FLAG", pd.Series("", index=df.index)).astype("string").eq("LOW_COVERAGE")
    low_coverage = low_coverage | (pd.to_numeric(df.get("SOURCE_COVERAGE", pd.Series(np.nan, index=df.index)), errors="coerce").fillna(0) < coverage_threshold)
    failed = pd.Series(prob).isna()
    status = np.where(failed, "failed", np.where(fallback_used | low_coverage.to_numpy(), "degraded", "success"))
    return pd.DataFrame(
        {
            "XH": df["XH"].astype("string"),
            "TERM_ID": df["TERM_ID"].astype("string"),
            "DOMAIN": config.get("domain", "study"),
            "DOMAIN_SCORE": prob,
            "DOMAIN_CONFIDENCE": np.abs(prob - 0.5) * 2,
            "MODEL_VERSION": config.get("model_version", "study_v1"),
            "FEATURE_VERSION": config.get("feature_version", "study_feature_v1"),
            "STATUS": status,
            "FALLBACK_USED": fallback_used,
            "SOURCE_TABLE": source,
        }
    )


def main() -> None:
    ensure_dirs()
    bundle = joblib.load(DM_DIR / "study_model.pkl")
    outputs = []
    for name, source in [("study_train_table.csv", "train"), ("study_infer_table.csv", "infer")]:
        path = DM_DIR / name
        if path.exists():
            outputs.append(score_table(pd.read_csv(path), bundle, source))
    result = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    write_csv(result, DM_DIR / "study_prediction_output.csv")
    print(f"prediction output written: {DM_DIR / 'study_prediction_output.csv'} rows={len(result)}")


if __name__ == "__main__":
    main()
