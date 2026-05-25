from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from study_common import DM_DIR, ensure_dirs, write_csv


def model_importance(bundle: dict, features: list[str]) -> pd.Series:
    model = bundle["primary_model"]
    estimator = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.ravel(estimator.coef_)
    else:
        values = np.ones(len(features))
    if len(values) != len(features):
        values = np.ones(len(features))
    return pd.Series(values, index=features, dtype="float64").abs()


def contribution_frame(bundle: dict, data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    model = bundle["primary_model"]
    if not features or data.empty:
        return pd.DataFrame(index=data.index, columns=features)
    x = data.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    try:
        import shap

        pipeline = model
        estimator = pipeline.named_steps.get("model", pipeline) if hasattr(pipeline, "named_steps") else pipeline
        matrix = pipeline.named_steps["imputer"].transform(x) if hasattr(pipeline, "named_steps") and "imputer" in pipeline.named_steps else x.fillna(0)
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(matrix)
        if isinstance(values, list):
            values = values[-1]
        if np.asarray(values).ndim == 3:
            values = np.asarray(values)[:, :, -1]
        return pd.DataFrame(values, index=data.index, columns=features).abs()
    except Exception:
        importance = model_importance(bundle, features)
        numeric = x.fillna(0).abs()
        return numeric.mul(importance, axis=1)


def explain_row(row: pd.Series, contribution: pd.Series) -> tuple[list[str], list[object], str]:
    ranked = contribution.fillna(0).sort_values(ascending=False)
    top = [c for c in ranked.head(3).index if ranked[c] > 0]
    while len(top) < 3:
        top.append("")
    values = [row.get(name, "") if name else "" for name in top]
    text_bits = [f"{name}={row.get(name)}" for name in top if name]
    text = "主要影响因素：" + "；".join(text_bits) if text_bits else "主要影响因素暂不足，需补充特征覆盖。"
    return top, values, text


def main() -> None:
    ensure_dirs()
    bundle = joblib.load(DM_DIR / "study_model.pkl")
    config = bundle["config"]
    features = config.get("feature_columns", [])
    prediction = pd.read_csv(DM_DIR / "study_prediction_output.csv")
    tables = []
    for name, source in [("study_train_table.csv", "train"), ("study_infer_table.csv", "infer")]:
        path = DM_DIR / name
        if path.exists():
            df = pd.read_csv(path)
            df["SOURCE_TABLE"] = source
            tables.append(df)
    data = pd.concat(tables, ignore_index=True, sort=False) if tables else pd.DataFrame()
    data = data.merge(prediction[["XH", "TERM_ID", "SOURCE_TABLE", "DOMAIN_SCORE"]], on=["XH", "TERM_ID", "SOURCE_TABLE"], how="inner")
    contributions = contribution_frame(bundle, data, features)

    rows = []
    for _, row in data.iterrows():
        top, values, text = explain_row(row, contributions.loc[row.name] if row.name in contributions.index else pd.Series(dtype=float))
        rows.append(
            {
                "XH": row["XH"],
                "TERM_ID": row["TERM_ID"],
                "DOMAIN": config.get("domain", "study"),
                "DOMAIN_SCORE": row["DOMAIN_SCORE"],
                "TOP_FEATURE_1": top[0],
                "TOP_FEATURE_1_VALUE": values[0],
                "TOP_FEATURE_2": top[1],
                "TOP_FEATURE_2_VALUE": values[1],
                "TOP_FEATURE_3": top[2],
                "TOP_FEATURE_3_VALUE": values[2],
                "EXPLANATION_TEXT": text,
                "MODEL_VERSION": config.get("model_version", "study_v1"),
                "FEATURE_VERSION": config.get("feature_version", "study_feature_v1"),
                "SOURCE_TABLE": row["SOURCE_TABLE"],
            }
        )
    result = pd.DataFrame(rows)
    write_csv(result, DM_DIR / "study_explanation_output.csv")
    print(f"explanation output written: {DM_DIR / 'study_explanation_output.csv'} rows={len(result)}")


if __name__ == "__main__":
    main()
