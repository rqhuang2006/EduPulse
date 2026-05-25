# Study Domain Delivery Bundle

This bundle contains the production data, documentation, and model artifacts for the study domain agent.

## Layout

- `data/`: train, inference, prediction, and explanation outputs.
- `docs/`: feature dictionary, quality report, validation report, and raw-data profiling documents.
- `model/`: trained model pickle, model config, and model metrics.

## Grain

All train and inference rows are keyed by unique `XH + TERM_ID`.

## Feature Contract

All modeling features use the `FEATURE_` prefix. The table also includes `SOURCE_COVERAGE` and `DATA_QUALITY_FLAG` for downstream filtering and audit.

## Prediction Status Rules

- `success`: primary model scored successfully and source coverage is acceptable.
- `degraded`: fallback scoring was used, `DATA_QUALITY_FLAG=LOW_COVERAGE`, or `SOURCE_COVERAGE` is below the configured minimum.
- `failed`: scoring could not produce a numeric domain score.

## Label Audit

`docs/study_label_audit.csv` keeps `NEXT_TERM_ID` and `LABEL_REASON` for audit. These fields are intentionally excluded from `data/study_train_table.csv`.

## Study Domain Overview

This bundle is the delivery package for the `study` domain. It is produced from learning-related Excel source tables under `data/raw/`, transformed through ODS and DWD layers, and packaged for harness consumption under `data/deliverables/study/`.

The primary grain of the training and inference tables is:

```text
XH + TERM_ID
```

Each row represents one student in one academic term.

## Raw Data Sources

The study domain pipeline uses the following raw source files:

- `学生基本信息.xlsx`: student profile information.
- `学籍异动.xlsx`: student status changes.
- `学生成绩.xlsx`: course grade records.
- `四六级成绩.xlsx`: CET score records.
- `学生选课信息.xlsx`: course selection records.
- `课程信息.xlsx`: course metadata.
- `考勤汇总.xlsx`: attendance summary records.
- `学生签到记录.xlsx`: sign-in records.
- `上课信息统计表.xlsx`: classroom session statistics.
- `课堂任务参与.xlsx`: classroom task participation.
- `学生作业提交记录.xlsx`: assignment submission and scoring records.
- `考试提交记录.xlsx`: exam/quiz submission and scoring records.
- `线上学习（综合表现）.xlsx`: online learning activity.
- `图书馆打卡记录.xlsx`: library visit records.
- Supplementary sources include `奖学金获奖.xlsx`, `学科竞赛.xlsx`, `本科生综合测评.xlsx`, and `毕业去向.xlsx`.

## Feature Layers

The DWD layer is organized by study-domain subject tables:

- `L0 student-term base`: base sample space keyed by `XH + TERM_ID`.
- `L1 grade`: course grade, average score, fail count, credit summary, CET score.
- `L2 course load`: selected course count, selected credits, retake indicators.
- `L3 attendance`: attendance events and abnormal attendance rate.
- `L4 class task`: class task count and participation rates.
- `L5 assignment`: assignment count, score average, missing count, submit rate.
- `L6 exam/quiz`: exam count, score average, missing count.
- `L7 online activity`: library visits and online activity features.

All modeling features use the `FEATURE_` prefix.

## Delivered Files

The harness-facing delivery files are located under:

```text
data/deliverables/study/
```

### Data

- `data/study_train_table.csv`
  - Training table.
  - Grain: `XH + TERM_ID`.
  - Contains `LABEL`, `DOMAIN`, `FEATURE_*`, and quality fields.
  - `NEXT_TERM_ID` and `LABEL_REASON` are intentionally excluded from the main training table.

- `data/study_infer_table.csv`
  - Inference table.
  - Grain: `XH + TERM_ID`.
  - Does not contain `LABEL`.

- `data/study_prediction_output.csv`
  - Prediction output for train and infer rows.
  - Required fields include `XH`, `TERM_ID`, `DOMAIN`, `DOMAIN_SCORE`, `DOMAIN_CONFIDENCE`, `MODEL_VERSION`, `FEATURE_VERSION`, `STATUS`, `FALLBACK_USED`.

- `data/study_explanation_output.csv`
  - Explanation output.
  - Contains `TOP_FEATURE_1~3`, corresponding feature values, and `EXPLANATION_TEXT`.

### Docs

- `docs/study_feature_dictionary.xlsx`
  - Feature dictionary with feature name, source file, source field, aggregation rule, time window, missing strategy, and model usage flag.

- `docs/study_quality_report.xlsx`
  - Quality report containing sample statistics, source coverage, feature missing rates, label distribution, model metrics, known issues, and recommendations.

- `docs/study_label_audit.csv`
  - Label audit file containing `NEXT_TERM_ID` and `LABEL_REASON`.
  - These fields are kept for audit only and are excluded from `study_train_table.csv`.

- `docs/study_validation_report.json`
  - Delivery validation result.
  - Current status: `PASS`.

### Model

- `model/study_model.pkl`
  - Serialized model bundle containing the primary model and fallback model.

- `model/study_model_config.json`
  - Model metadata and feature list.
  - Current model version: `study_v1`.
  - Current feature version: `study_feature_v1`.

- `model/study_model_metrics.json`
  - Model metrics.

## Model Method

The primary model is:

```text
LightGBMClassifier
```

The fallback model is:

```text
LogisticRegression
```

The label is designed to predict next-term study risk. The model uses only business `FEATURE_*` columns listed in `model/study_model_config.json`.

The following quality fields are retained in train/infer tables but are not used as model features:

- `FEATURE_MISSING_RATE`
- `SOURCE_COVERAGE`
- `DATA_QUALITY_FLAG`

## Explanation Method

The explanation output is generated by ranking feature contributions and selecting the top 3 features for each prediction.

Each explanation row contains:

- `TOP_FEATURE_1`
- `TOP_FEATURE_1_VALUE`
- `TOP_FEATURE_2`
- `TOP_FEATURE_2_VALUE`
- `TOP_FEATURE_3`
- `TOP_FEATURE_3_VALUE`
- `EXPLANATION_TEXT`

The top features are real `FEATURE_*` columns from the train/infer tables.

## Current Quality Summary

Current validated model metrics:

```text
valid.auc    = 0.852089926589149
valid.f1     = 0.5763000852514919
valid.recall = 0.6224677716390423
```

The current validation report status is:

```text
PASS
```

## Known Issues and Business Risks

The delivery is structurally valid, but the following business risks should be noted:

1. High degraded ratio:
   - A large share of prediction rows are marked as `degraded`, mainly due to low source coverage.

2. High low-coverage ratio:
   - Many training rows have `DATA_QUALITY_FLAG=LOW_COVERAGE`.

3. L4/L5/L6 coverage is limited:
   - Classroom task, assignment, and exam/quiz features have high missing rates.
   - These dimensions should be interpreted carefully.

4. Metadata fields are retained:
   - Some profile/status fields such as `XB`, `MZMC`, `ZZMMMC`, `CSRQ`, `JG`, `XSM`, `ZYM`, `STATUS_CHANGE_FLAG`, and `INVALID_TERM_FLAG` may appear in train/infer tables.
   - They are not model features unless explicitly listed in `model/study_model_config.json`.

5. `FEATURE_CET_SCORE_MAX` currently has no effective coverage in the delivered tables.

## Recommended Run Order

If rebuilding the project from raw data is required, use the pipeline in the following order:

```text
00_profile_raw.py
01_build_field_registry.py
02_build_ods.py
03_build_student_term_base.py
10_build_l1_grade.py
11_build_l2_course_load.py
12_build_l3_attendance.py
13_build_l4_class_task.py
14_build_l5_assignment.py
15_build_l6_exam_quiz.py
16_build_l7_online_activity.py
20_build_label.py
21_build_study_train_infer.py
22_build_feature_dictionary.py
30_train_study_model.py
31_generate_prediction_output.py
32_generate_explanation_output.py
33_generate_quality_report.py
34_package_study_bundle.py
35_validate_study_bundle.py
```

A convenience entry point exists at:

```text
main.py
```

It supports staged execution, but for final submitted artifacts, harness should consume only:

```text
data/deliverables/study/
```
