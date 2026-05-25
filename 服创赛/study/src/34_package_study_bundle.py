from __future__ import annotations

from pathlib import Path

from study_common import DELIVERABLE_DIR, DM_DIR, DOCS_DIR, ensure_dirs, copy_if_exists


DATA_FILES = [
    "study_train_table.csv",
    "study_infer_table.csv",
    "study_prediction_output.csv",
    "study_explanation_output.csv",
]

DOC_FILES = [
    "study_feature_dictionary.xlsx",
    "study_quality_report.xlsx",
    "study_label_audit.csv",
    "raw_inventory.xlsx",
    "raw_field_mapping.xlsx",
    "ods_cleaning_log.csv",
]

MODEL_FILES = [
    "study_model.pkl",
    "study_model_config.json",
    "study_model_metrics.json",
]


def write_readme(target: Path) -> None:
    readme = target / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Study Domain Delivery Bundle",
                "",
                "This bundle contains the production data, documentation, and model artifacts for the study domain agent.",
                "",
                "## Layout",
                "",
                "- `data/`: train, inference, prediction, and explanation outputs.",
                "- `docs/`: feature dictionary, quality report, validation report, and raw-data profiling documents.",
                "- `model/`: trained model pickle, model config, and model metrics.",
                "",
                "## Grain",
                "",
                "All train and inference rows are keyed by unique `XH + TERM_ID`.",
                "",
                "## Feature Contract",
                "",
                "All modeling features use the `FEATURE_` prefix. The table also includes `SOURCE_COVERAGE` and `DATA_QUALITY_FLAG` for downstream filtering and audit.",
                "",
                "## Prediction Status Rules",
                "",
                "- `success`: primary model scored successfully and source coverage is acceptable.",
                "- `degraded`: fallback scoring was used, `DATA_QUALITY_FLAG=LOW_COVERAGE`, or `SOURCE_COVERAGE` is below the configured minimum.",
                "- `failed`: scoring could not produce a numeric domain score.",
                "",
                "## Label Audit",
                "",
                "`docs/study_label_audit.csv` keeps `NEXT_TERM_ID` and `LABEL_REASON` for audit. These fields are intentionally excluded from `data/study_train_table.csv`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    target = DELIVERABLE_DIR / "study"
    data_dir = target / "data"
    docs_dir = target / "docs"
    model_dir = target / "model"
    for path in [data_dir, docs_dir, model_dir]:
        path.mkdir(parents=True, exist_ok=True)

    copied = []
    for name in DATA_FILES:
        if copy_if_exists(DM_DIR / name, data_dir / name):
            copied.append(f"data/{name}")
    for name in DOC_FILES:
        if name == "study_feature_dictionary.xlsx" and (DM_DIR / "study_feature_dictionary.generated.xlsx").exists():
            source = DM_DIR / "study_feature_dictionary.generated.xlsx"
        elif name == "study_feature_dictionary.xlsx" and (DM_DIR / "study_feature_dictionary.tmp.xlsx").exists():
            source = DM_DIR / "study_feature_dictionary.tmp.xlsx"
        elif name == "study_quality_report.xlsx" and (DM_DIR / "study_quality_report.generated.xlsx").exists():
            source = DM_DIR / "study_quality_report.generated.xlsx"
        elif name == "study_quality_report.xlsx" and (DM_DIR / "study_quality_report.tmp.xlsx").exists():
            source = DM_DIR / "study_quality_report.tmp.xlsx"
        else:
            source = DM_DIR / name if (DM_DIR / name).exists() else DOCS_DIR / name
        if not source.exists() and name == "study_quality_report.xlsx":
            source = DM_DIR / "study_quality_report.generated.xlsx"
        if not source.exists() and name == "study_quality_report.xlsx":
            source = DM_DIR / "study_quality_report.tmp.xlsx"
        if not source.exists() and name == "study_feature_dictionary.xlsx":
            source = DM_DIR / "study_feature_dictionary.generated.xlsx"
        if not source.exists() and name == "study_feature_dictionary.xlsx":
            source = DM_DIR / "study_feature_dictionary.tmp.xlsx"
        if copy_if_exists(source, docs_dir / name):
            copied.append(f"docs/{name}")
    for name in MODEL_FILES:
        if copy_if_exists(DM_DIR / name, model_dir / name):
            copied.append(f"model/{name}")
    write_readme(target)
    copied.append("README.md")
    print(f"study bundle packaged: {target} files={len(copied)}")


if __name__ == "__main__":
    main()
