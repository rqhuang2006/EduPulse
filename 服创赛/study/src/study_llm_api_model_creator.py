from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


ROOT = Path(__file__).resolve().parents[1]
DM_DIR = ROOT / "data" / "dm"
DOCS_DIR = ROOT / "data" / "deliverables" / "study" / "docs"
TRAIN_PATH = ROOT / "data" / "deliverables" / "study" / "data" / "study_train_table.csv"
CONFIG_PATH = ROOT / "data" / "deliverables" / "study" / "model" / "study_model_config.json"
API_RESPONSE_PATH = DM_DIR / "study_llm_api_response.json"
RESULT_PATH = DM_DIR / "study_llm_api_model_result.json"
DELIVERABLE_RESULT_PATH = DOCS_DIR / "study_llm_api_model_result.json"
STACKING_PRED_PATH = DM_DIR / "study_llm_api_stacking_valid_predictions.csv"


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def llm_api_response() -> dict[str, Any]:
    """Simulated LLM API response: a concrete model creation recipe, not a selector."""
    return {
        "api_type": "simulated_llm_model_creator",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "objective": "Create a stronger study risk model from existing train table without using a control-decision model.",
        "feature_blueprint": {
            "base_feature_groups": [
                "all_formal_model_features",
                "grade_course_features",
                "engagement_and_missingness_features",
                "llm_domain_synthesized_features",
            ],
            "generated_features": [
                "FEATURE_LLM_GRADE_RISK_INDEX",
                "FEATURE_LLM_COURSE_PRESSURE_INDEX",
                "FEATURE_LLM_ATTENDANCE_RISK_INDEX",
                "FEATURE_LLM_ASSIGNMENT_EXAM_GAP_INDEX",
                "FEATURE_LLM_CLASS_ENGAGEMENT_GAP_INDEX",
                "FEATURE_LLM_L456_SIGNAL_ABSENCE",
                "FEATURE_LLM_SCORE_VOLATILITY_INDEX",
                "FEATURE_LLM_COVERAGE_ADJUSTED_RISK",
                "FEATURE_LLM_GRADE_PRESSURE_CROSS",
                "FEATURE_LLM_ATTENDANCE_GRADE_CROSS",
                "FEATURE_LLM_EXAM_GRADE_CROSS",
            ],
            "guardrail": "Do not use LABEL, NEXT_TERM_ID, LABEL_REASON, or post-label fields as features.",
        },
        "model_blueprint": {
            "method": "OOF stacking",
            "base_models": [
                {"name": "lgbm_all", "model": "LightGBM", "features": "all_features_plus_llm"},
                {"name": "xgb_all", "model": "XGBoost", "features": "all_features_plus_llm"},
                {"name": "rf_grade", "model": "RandomForest", "features": "grade_plus_llm"},
                {"name": "logit_llm", "model": "LogisticRegression", "features": "llm_only"},
            ],
            "meta_model": "LogisticRegression over OOF base scores",
            "validation": "stratified holdout plus 5-fold OOF on development split",
            "publish_policy": "candidate_only",
        },
    }


def num(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[name], errors="coerce")


def clipped(series: pd.Series, lower: float = 0.0, upper: float = 1.0) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower, upper)


def add_llm_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    grade_count = num(out, "FEATURE_GRADE_COURSE_COUNT").fillna(0.0)
    grade_avg = num(out, "FEATURE_GRADE_AVG_SCORE")
    grade_min = num(out, "FEATURE_GRADE_MIN_SCORE")
    fail_count = num(out, "FEATURE_GRADE_FAIL_COUNT").fillna(0.0)
    selected_count = num(out, "FEATURE_COURSE_SELECTED_COUNT").fillna(0.0)
    course_credit = num(out, "FEATURE_COURSE_CREDIT_SUM").fillna(0.0)
    retake_count = num(out, "FEATURE_COURSE_RETAKE_COUNT").fillna(0.0)
    attendance_events = num(out, "FEATURE_ATTENDANCE_EVENT_COUNT").fillna(0.0)
    attendance_abnormal = num(out, "FEATURE_ATTENDANCE_ABNORMAL_COUNT").fillna(0.0)
    attendance_rate = num(out, "FEATURE_ATTENDANCE_ABNORMAL_RATE").fillna(0.0)
    assignment_count = num(out, "FEATURE_ASSIGNMENT_COUNT")
    assignment_missing = num(out, "FEATURE_ASSIGNMENT_MISSING_COUNT").fillna(0.0)
    assignment_submit = num(out, "FEATURE_ASSIGNMENT_SUBMIT_RATE")
    exam_count = num(out, "FEATURE_EXAM_COUNT")
    exam_missing = num(out, "FEATURE_EXAM_MISSING_COUNT").fillna(0.0)
    exam_score = num(out, "FEATURE_EXAM_SCORE_AVG")
    class_task = num(out, "FEATURE_CLASS_TASK_COUNT")
    class_rate = num(out, "FEATURE_CLASS_TASK_RATE_AVG")
    video_rate = num(out, "FEATURE_CLASS_VIDEO_RATE_AVG")
    source_coverage = num(out, "SOURCE_COVERAGE").fillna(0.0)

    out["FEATURE_LLM_GRADE_RISK_INDEX"] = clipped(
        0.35 * ((100.0 - grade_avg) / 100.0)
        + 0.25 * ((60.0 - grade_min).clip(lower=0.0) / 60.0)
        + 0.25 * (fail_count / (grade_count + 1.0))
        + 0.15 * (retake_count / (selected_count + 1.0))
    )
    out["FEATURE_LLM_COURSE_PRESSURE_INDEX"] = clipped(0.55 * (selected_count / 10.0) + 0.45 * (course_credit / 30.0))
    out["FEATURE_LLM_ATTENDANCE_RISK_INDEX"] = clipped(
        0.55 * attendance_rate + 0.45 * (attendance_abnormal / (attendance_events + 1.0))
    )
    out["FEATURE_LLM_ASSIGNMENT_EXAM_GAP_INDEX"] = clipped(
        0.30 * (assignment_missing / (assignment_count.fillna(0.0) + 1.0))
        + 0.25 * (1.0 - assignment_submit.fillna(1.0))
        + 0.25 * (exam_missing / (exam_count.fillna(0.0) + 1.0))
        + 0.20 * ((60.0 - exam_score).clip(lower=0.0) / 60.0).fillna(0.0)
    )
    out["FEATURE_LLM_CLASS_ENGAGEMENT_GAP_INDEX"] = clipped(
        0.45 * (1.0 - class_rate.fillna(1.0)) + 0.35 * (1.0 - video_rate.fillna(1.0)) + 0.20 * class_task.isna().astype(float)
    )
    out["FEATURE_LLM_SCORE_VOLATILITY_INDEX"] = clipped((grade_avg - grade_min).abs() / 50.0)
    l456_cols = [
        "FEATURE_CLASS_TASK_COUNT",
        "FEATURE_CLASS_TASK_RATE_AVG",
        "FEATURE_CLASS_VIDEO_RATE_AVG",
        "FEATURE_ASSIGNMENT_COUNT",
        "FEATURE_ASSIGNMENT_SCORE_AVG",
        "FEATURE_ASSIGNMENT_MISSING_COUNT",
        "FEATURE_ASSIGNMENT_SUBMIT_RATE",
        "FEATURE_EXAM_COUNT",
        "FEATURE_EXAM_SCORE_AVG",
        "FEATURE_EXAM_MISSING_COUNT",
    ]
    present_l456 = out[[c for c in l456_cols if c in out.columns]].notna().any(axis=1)
    out["FEATURE_LLM_L456_SIGNAL_ABSENCE"] = (~present_l456).astype(float)
    out["FEATURE_LLM_GRADE_PRESSURE_CROSS"] = clipped(out["FEATURE_LLM_GRADE_RISK_INDEX"] * out["FEATURE_LLM_COURSE_PRESSURE_INDEX"])
    out["FEATURE_LLM_ATTENDANCE_GRADE_CROSS"] = clipped(out["FEATURE_LLM_ATTENDANCE_RISK_INDEX"] * out["FEATURE_LLM_GRADE_RISK_INDEX"])
    out["FEATURE_LLM_EXAM_GRADE_CROSS"] = clipped(out["FEATURE_LLM_ASSIGNMENT_EXAM_GAP_INDEX"] * out["FEATURE_LLM_GRADE_RISK_INDEX"])
    out["FEATURE_LLM_COVERAGE_ADJUSTED_RISK"] = clipped(
        out["FEATURE_LLM_GRADE_RISK_INDEX"] * (0.75 + 0.25 * source_coverage)
        + out["FEATURE_LLM_L456_SIGNAL_ABSENCE"] * 0.05
    )
    return out


def make_model(name: str) -> Any:
    if name == "LightGBM":
        if LGBMClassifier is None:
            raise RuntimeError("LightGBM is unavailable")
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=220,
                        learning_rate=0.035,
                        num_leaves=24,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        class_weight="balanced",
                        random_state=42,
                        verbose=-1,
                    ),
                ),
            ]
        )
    if name == "XGBoost":
        if XGBClassifier is None:
            raise RuntimeError("XGBoost is unavailable")
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=220,
                        max_depth=3,
                        learning_rate=0.035,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    if name == "RandomForest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=180,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    if name == "LogisticRegression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=700, class_weight="balanced", random_state=42)),
            ]
        )
    raise ValueError(name)


def predict_score(model: Any, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x))[:, 1]


def metrics(y_true: pd.Series, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (score >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, score)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "positive_rate": float(pred.mean()),
        "threshold": float(threshold),
    }


def best_threshold(y_true: pd.Series, score: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in [round(x, 2) for x in np.arange(0.1, 0.91, 0.02)]:
        item = metrics(y_true, score, threshold)
        rows.append(item)
    table = pd.DataFrame(rows)
    selected = table.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    return float(selected["threshold"]), table


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    formal = [c for c in columns if c.startswith("FEATURE_") and not c.startswith("FEATURE_LLM_")]
    llm = [c for c in columns if c.startswith("FEATURE_LLM_")]
    grade = [c for c in formal if any(key in c for key in ["GRADE", "COURSE", "CET"])] + llm
    return {
        "all_features_plus_llm": formal + llm,
        "grade_plus_llm": grade,
        "llm_only": llm,
    }


def run_creator() -> dict[str, Any]:
    DM_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    api_payload = llm_api_response()
    write_json(API_RESPONSE_PATH, api_payload)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    formal_valid = config.get("metrics", {}).get("valid", {})
    df = pd.read_csv(TRAIN_PATH)
    df = add_llm_features(df)
    y = pd.to_numeric(df[config.get("label_name", "LABEL")], errors="coerce").astype(int)
    dev_idx, holdout_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y)
    dev = df.loc[dev_idx].copy()
    holdout = df.loc[holdout_idx].copy()
    y_dev = y.loc[dev_idx]
    y_holdout = y.loc[holdout_idx]

    sets = feature_sets(list(df.columns))
    base_specs = [
        ("lgbm_all", "LightGBM", "all_features_plus_llm"),
        ("xgb_all", "XGBoost", "all_features_plus_llm"),
        ("rf_grade", "RandomForest", "grade_plus_llm"),
        ("logit_llm", "LogisticRegression", "llm_only"),
    ]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = pd.DataFrame(index=dev.index)
    holdout_scores = pd.DataFrame(index=holdout.index)
    base_results = []

    for base_name, model_name, set_name in base_specs:
        features = sets[set_name]
        oof_score = np.zeros(len(dev))
        fold_holdout = []
        for fold_id, (train_pos, valid_pos) in enumerate(folds.split(dev[features], y_dev), start=1):
            model = make_model(model_name)
            train_index = dev.index[train_pos]
            valid_index = dev.index[valid_pos]
            model.fit(dev.loc[train_index, features], y_dev.loc[train_index])
            oof_score[valid_pos] = predict_score(model, dev.loc[valid_index, features])
            fold_holdout.append(predict_score(model, holdout[features]))
        holdout_score = np.mean(np.vstack(fold_holdout), axis=0)
        oof[base_name] = oof_score
        holdout_scores[base_name] = holdout_score
        threshold, _ = best_threshold(y_dev, oof_score)
        base_result = metrics(y_holdout, holdout_score, threshold)
        base_result.update({"base_name": base_name, "model_name": model_name, "feature_set": set_name, "feature_count": len(features)})
        base_results.append(base_result)

    meta = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
        ]
    )
    meta.fit(oof, y_dev)
    meta_oof = predict_score(meta, oof)
    meta_holdout = predict_score(meta, holdout_scores)
    meta_threshold, threshold_table = best_threshold(y_dev, meta_oof)
    stacking_metrics = metrics(y_holdout, meta_holdout, meta_threshold)

    pred_table = holdout[["XH", "TERM_ID"]].copy()
    pred_table["LABEL"] = y_holdout.to_numpy()
    for col in holdout_scores.columns:
        pred_table[col] = holdout_scores[col].to_numpy()
    pred_table["llm_api_stacking_score"] = meta_holdout
    pred_table["llm_api_stacking_pred"] = (meta_holdout >= meta_threshold).astype(int)
    pred_table.to_csv(STACKING_PRED_PATH, index=False, encoding="utf-8-sig")

    result = {
        "request_id": "study_llm_api_model_creator",
        "status": "candidate_only",
        "api_response_path": str(API_RESPONSE_PATH),
        "prediction_path": str(STACKING_PRED_PATH),
        "formal_model_valid_metrics": formal_valid,
        "base_model_holdout_metrics": base_results,
        "llm_api_stacking_holdout_metrics": stacking_metrics,
        "best_base_auc": max(item["auc"] for item in base_results),
        "best_base_f1": max(item["f1"] for item in base_results),
        "improvement_vs_formal": {
            "auc_delta": float(stacking_metrics["auc"] - formal_valid.get("auc", np.nan)),
            "f1_delta": float(stacking_metrics["f1"] - formal_valid.get("f1", np.nan)),
            "recall_delta": float(stacking_metrics["recall"] - formal_valid.get("recall", np.nan)),
        },
        "publish_recommendation": "do_not_switch_primary_by_auc" if stacking_metrics["auc"] < formal_valid.get("auc", 0) else "eligible_for_candidate_review",
        "note": "This is a simulated LLM API-created OOF stacking model. It does not overwrite study_model.pkl or study_model_config.json.",
    }
    write_json(RESULT_PATH, result)
    write_json(DM_DIR / "study_llm_api_threshold_tuning.json", {"selected_threshold": meta_threshold, "rows": threshold_table.to_dict(orient="records")})
    import shutil

    shutil.copy2(RESULT_PATH, DELIVERABLE_RESULT_PATH)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulated LLM API model creation for study domain.")
    parser.parse_args()
    result = run_creator()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
