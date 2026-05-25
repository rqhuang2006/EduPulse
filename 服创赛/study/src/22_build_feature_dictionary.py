from __future__ import annotations

import pandas as pd

from study_common import DM_DIR, ensure_dirs, feature_registry


DETAILS = {
    "FEATURE_GRADE_COURSE_COUNT": ("study_l1_grade_term.parquet", "SCORE", "count non-null course score rows by XH+TERM_ID", "current_term"),
    "FEATURE_GRADE_AVG_SCORE": ("study_l1_grade_term.parquet", "SCORE", "mean course score by XH+TERM_ID", "current_term"),
    "FEATURE_GRADE_MIN_SCORE": ("study_l1_grade_term.parquet", "SCORE", "min course score by XH+TERM_ID", "current_term"),
    "FEATURE_GRADE_FAIL_COUNT": ("study_l1_grade_term.parquet", "SCORE", "count score < 60 by XH+TERM_ID", "current_term"),
    "FEATURE_GRADE_CREDIT_SUM": ("study_l1_grade_term.parquet", "CREDIT", "sum course credit by XH+TERM_ID", "current_term"),
    "FEATURE_CET_SCORE_MAX": ("study_l1_grade_term.parquet", "SCORE", "max CET score by XH+TERM_ID", "current_or_observed_term"),
    "FEATURE_COURSE_SELECTED_COUNT": ("study_l2_course_load_term.parquet", "COURSE_STD", "nunique selected courses by XH+TERM_ID", "current_term"),
    "FEATURE_COURSE_CREDIT_SUM": ("study_l2_course_load_term.parquet", "CREDIT", "sum selected course credit by XH+TERM_ID", "current_term"),
    "FEATURE_COURSE_RETAKE_COUNT": ("study_l2_course_load_term.parquet", "STATUS", "sum retake/abnormal course flags by XH+TERM_ID", "current_term"),
    "FEATURE_ATTENDANCE_EVENT_COUNT": ("study_l3_attendance_term.parquet", "XH", "count attendance events by XH+TERM_ID", "current_term"),
    "FEATURE_ATTENDANCE_ABNORMAL_COUNT": ("study_l3_attendance_term.parquet", "STATUS", "sum abnormal attendance flags by XH+TERM_ID", "current_term"),
    "FEATURE_ATTENDANCE_ABNORMAL_RATE": ("study_l3_attendance_term.parquet", "STATUS", "abnormal attendance count divided by attendance event count", "current_term"),
    "FEATURE_CLASS_TASK_COUNT": ("study_l4_class_task_term.parquet", "XH", "count class task records by XH+TERM_ID", "current_term"),
    "FEATURE_CLASS_TASK_RATE_AVG": ("study_l4_class_task_term.parquet", "JOB_RATE/TASK_RATE", "mean class task completion rate by XH+TERM_ID", "current_term"),
    "FEATURE_CLASS_VIDEO_RATE_AVG": ("study_l4_class_task_term.parquet", "VIDEOJOB_RATE", "mean video task completion rate by XH+TERM_ID", "current_term"),
    "FEATURE_ASSIGNMENT_COUNT": ("study_l5_assignment_term.parquet", "XH", "count assignment records by XH+TERM_ID", "current_term"),
    "FEATURE_ASSIGNMENT_SCORE_AVG": ("study_l5_assignment_term.parquet", "SCORE", "mean assignment score by XH+TERM_ID", "current_term"),
    "FEATURE_ASSIGNMENT_MISSING_COUNT": ("study_l5_assignment_term.parquet", "STATUS", "sum missing/abnormal assignment flags by XH+TERM_ID", "current_term"),
    "FEATURE_ASSIGNMENT_SUBMIT_RATE": ("study_l5_assignment_term.parquet", "STATUS", "1 - missing assignment count / assignment count", "current_term"),
    "FEATURE_EXAM_COUNT": ("study_l6_exam_quiz_term.parquet", "XH", "count exam/quiz records by XH+TERM_ID", "current_term"),
    "FEATURE_EXAM_SCORE_AVG": ("study_l6_exam_quiz_term.parquet", "SCORE", "mean exam/quiz score by XH+TERM_ID", "current_term"),
    "FEATURE_EXAM_MISSING_COUNT": ("study_l6_exam_quiz_term.parquet", "STATUS", "sum missing/abnormal exam flags by XH+TERM_ID", "current_term"),
    "FEATURE_LIBRARY_VISIT_COUNT": ("study_l7_online_activity_term.parquet", "XH", "sum library visit flags by XH+TERM_ID", "current_term"),
    "FEATURE_ONLINE_ACTIVITY_SCORE_AVG": ("study_l7_online_activity_term.parquet", "SCORE", "mean online activity score by XH+TERM_ID", "current_term"),
    "FEATURE_CLASS_HEAD_UP_RATE_AVG": ("study_l3_attendance_term.parquet", "HEAD_UP_RATE", "mean classroom head-up rate by XH+TERM_ID", "current_term"),
    "FEATURE_CLASS_FRONT_ROW_RATE_AVG": ("study_l3_attendance_term.parquet", "FRONT_ROW_RATE", "mean classroom front-row rate by XH+TERM_ID", "current_term"),
    "FEATURE_MISSING_RATE": ("study_train_table.csv/study_infer_table.csv", "FEATURE_*", "row-level missing ratio across business FEATURE_* columns", "current_row"),
}

EXCLUDED_MODEL_FEATURES = {"FEATURE_MISSING_RATE"}


def write_excel_safely(df: pd.DataFrame, output) -> None:
    temp = output.with_suffix(".tmp.xlsx")
    with pd.ExcelWriter(temp) as writer:
        df.sort_values("feature_name").to_excel(writer, sheet_name="features", index=False)
    try:
        temp.replace(output)
    except PermissionError:
        fallback = output.with_suffix(".generated.xlsx")
        try:
            temp.replace(fallback)
        except PermissionError:
            pass
        print(f"warning: {output} is locked; wrote fallback workbook: {fallback}")


def feature_cn_name(feature: str, description: str) -> str:
    return description or feature.replace("FEATURE_", "").replace("_", " ").title()


def build_row(feature: str, meta: dict, subject: str, used_features: set[str]) -> dict:
    source_file, source_field, aggregation_rule, time_window = DETAILS.get(
        feature,
        ("auto_detected", "auto_detected", "auto detected from train/infer table", "current_term"),
    )
    description = meta.get("description", "") if isinstance(meta, dict) else ""
    used = feature in used_features and feature not in EXCLUDED_MODEL_FEATURES
    return {
        "feature_name": feature,
        "feature_cn_name": feature_cn_name(feature, description),
        "source_file": source_file,
        "source_field": source_field,
        "aggregation_rule": aggregation_rule,
        "time_window": time_window,
        "missing_strategy": "keep_null; model pipeline uses median imputation for used_in_model=true",
        "used_in_model": used,
        "note": "quality/audit feature; not used in model" if feature in EXCLUDED_MODEL_FEATURES else description,
        "FEATURE_NAME": feature,
        "SUBJECT": subject,
        "DOMAIN": "study",
        "FEATURE_VERSION": feature_registry().get("feature_version", "study_feature_v1"),
        "TYPE": meta.get("type", "numeric") if isinstance(meta, dict) else "numeric",
        "DESCRIPTION": description,
    }


def main() -> None:
    ensure_dirs()
    registry = feature_registry()
    used_features = set()
    for filename in ["study_train_table.csv", "study_infer_table.csv"]:
        path = DM_DIR / filename
        if path.exists():
            used_features.update([c for c in pd.read_csv(path, nrows=0).columns if c.startswith("FEATURE_")])

    rows = []
    for subject, spec in registry.get("subjects", {}).items():
        for feature, meta in spec.get("features", {}).items():
            rows.append(build_row(feature, meta, subject, used_features))

    dictionary = pd.DataFrame(rows)
    missing = sorted(used_features - set(dictionary["feature_name"]))
    if missing:
        dictionary = pd.concat(
            [
                dictionary,
                pd.DataFrame([build_row(feature, {"description": "Auto detected from train/infer table"}, "auto_detected", used_features) for feature in missing]),
            ],
            ignore_index=True,
        )
    dictionary = dictionary.drop_duplicates("feature_name", keep="first")

    output = DM_DIR / "study_feature_dictionary.xlsx"
    write_excel_safely(dictionary, output)
    print(f"feature dictionary written: {output} rows={len(dictionary)}")


if __name__ == "__main__":
    main()
