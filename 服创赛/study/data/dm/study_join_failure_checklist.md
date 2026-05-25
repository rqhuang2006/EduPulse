# Study Join Failure Checklist

Request: `study_infer_serving_request`

## Executive Summary

- `join_failure_rows`: `1995 / 2508`
- Primary key status: `XH + TERM_ID` valid, duplicate rate `0`
- The bottleneck is not base-key merge failure.
- The bottleneck is that multiple upstream feature layers have almost no coverage for the infer terms, especially `2023-2024-2` and `2024-2025-2`.

## Which Infer Terms Are Broken

| TERM_ID | join_failure_rows | infer_rows | join_failure_ratio |
|---|---:|---:|---:|
| 2023-2024-2 | 1841 | 2265 | 81.28% |
| 2024-2025-2 | 89 | 175 | 50.86% |
| 2021-2022-2 | 60 | 60 | 100.00% |
| 2019-2020-2 | 2 | 2 | 100.00% |
| 2023-2024-1 | 2 | 2 | 100.00% |
| 2020-2021-1 | 1 | 1 | 100.00% |

## Feature Layer Coverage On The 1995 Join-Failure Rows

| Layer | join hit / non-null any rate |
|---|---:|
| grade | 99.80% |
| course | 98.45% |
| attendance | 0.00% |
| class_task | 0.05% |
| assignment | 0.00% |
| exam | 0.00% |
| online/library | 0.00% |

Conclusion:

- `grade` and `course` are basically healthy.
- `attendance`, `class_task`, `assignment`, `exam`, `online/library` are the main reason rows are marked as join failure / low coverage.

## DWD Term Table Coverage

| DWD table | observed terms |
|---|---|
| `study_l1_grade_term` | full historical coverage including `2023-2024-2`, `2024-2025-2` |
| `study_l2_course_load_term` | full historical coverage including `2023-2024-2`, `2024-2025-2` |
| `study_l3_attendance_term` | only `2023-2024-2`, `2024-2025-1`, `2024-2025-2` and row count is tiny (`179`) |
| `study_l4_class_task_term` | only `2020-2021-1`, `2021-2022-1`, `2022-2023-1` |
| `study_l5_assignment_term` | only `2020-2021-1`, `2021-2022-1` |
| `study_l6_exam_quiz_term` | only `2020-2021-1`, `2021-2022-1`, tiny `2022-2023-1` |
| `study_l7_online_activity_term` | spread across terms, but not enough to rescue the failing rows |

## Root Cause By Source Table

### 1. Attendance

- ODS source `attendance_summary` has term IDs, but only `4666` rows total.
- Term distribution is concentrated in:
  - `2024-2025-1`: `3451`
  - `2024-2025-2`: `1132`
  - `2023-2024-2`: `83`
- ODS source `signin` has `748679` rows but `TERM_ID` missing rate is `100%`.
- Result: `study_l3_attendance_term` ends up with only `179` term rows, and on the `1995` failure rows the attendance join hit rate is `0%`.

### 2. Class Task

- ODS `class_task` itself is healthy on keys and term IDs.
- But its available term coverage is only:
  - `2020-2021-1`
  - `2021-2022-1`
  - `2022-2023-1`
- There is effectively no `2023-2024-2` / `2024-2025-2` class-task term coverage.
- Result: the join hit rate on the failing rows is only `0.05%`.

### 3. Assignment

- ODS `assignment` has:
  - `XH` present
  - `COURSE_STD` present
  - `CLAZZ_ID` present
  - but `TERM_ID` missing rate is `100%`
- The enrichment logic in [src/14_build_l5_assignment.py](/abs/path-not-used) maps term from `class_task` or `course_selection`.
- Measured key overlap from ODS:
  - `assignment` -> `class_task` by `XH + COURSE_STD`: `99.88%`
  - `assignment` -> `class_task` by `XH + CLAZZ_ID`: `100%`
  - `assignment` -> `course_selection` by `XH + COURSE_STD`: `0%`
- This means the assignment records can mostly inherit term only from `class_task`.
- Since `class_task` itself only covers old terms, `study_l5_assignment_term` is trapped in old terms too:
  - `2020-2021-1`
  - `2021-2022-1`
- Result: assignment features are absent for current infer terms.

### 4. Exam Quiz

- ODS `exam_quiz` has:
  - `XH` present
  - `COURSE_STD` present
  - `CLAZZ_ID` present
  - but `TERM_ID` missing rate is `100%`
- Measured key overlap from ODS:
  - `exam_quiz` -> `class_task` by `XH + COURSE_STD`: `99.91%`
  - `exam_quiz` -> `class_task` by `XH + CLAZZ_ID`: `99.93%`
  - `exam_quiz` -> `course_selection` by `XH + COURSE_STD`: `0%`
- Same failure mode as assignment:
  - term inheritance depends almost entirely on `class_task`
  - `class_task` only has old terms
- Result: `study_l6_exam_quiz_term` has no useful coverage for current infer terms.

### 5. Online Activity

- ODS `online_activity` has `2498` rows, `2498` distinct students.
- `TERM_ID` missing rate is `100%`.
- Columns do not provide `COURSE_STD` or `CLAZZ_ID`, so there is no implemented bridge to infer term from `class_task` or `course_selection`.
- Result: the online-score part cannot be attached to term-level rows.

### 6. Library Visit

- ODS `library_visit` is one of the few healthy sources:
  - `TERM_ID` missing rate `0%`
  - has real recent terms including `2023-2024-2` and `2024-2025-2`
- But it only contributes `FEATURE_LIBRARY_VISIT_COUNT`, which is too weak to offset the missing attendance / task / assignment / exam layers.
- On the `1995` failure rows, even this layer does not hit.

## Concrete Suspects To Fix First

1. Fix `signin` term inference.
   Current `signin` data is large but unusable because `TERM_ID` is fully missing.

2. Rebuild `class_task` with recent terms.
   If `class_task` really should include `2023-2024-2` and `2024-2025-2`, the current cleaned data is incomplete or term inference is wrong.

3. Stop relying on `class_task` as the only term bridge for `assignment` and `exam_quiz`.
   Right now both layers are downstream victims of `class_task` term sparsity.

4. Add a term-enrichment path for `online_activity`.
   It currently has no usable bridge key beyond `XH`.

5. Add source-level monitoring by term.
   The agent should fail fast when a feature layer suddenly disappears for a serving term.

## Recommended Debug Order

1. Re-open ODS `class_task` and confirm whether `2023-2024-2` / `2024-2025-2` raw rows exist.
2. Re-open ODS `signin` and add term derivation from timestamp or session mapping.
3. For `assignment` and `exam_quiz`, inspect why `course_selection` mapping is `0%` even though `COURSE_STD` exists.
4. Design a term bridge for `online_activity`; otherwise that source will stay outside term-level inference.
5. Rebuild DWD -> infer table and re-check:
   - `study_l3_attendance_term`
   - `study_l4_class_task_term`
   - `study_l5_assignment_term`
   - `study_l6_exam_quiz_term`
   - `study_l7_online_activity_term`
