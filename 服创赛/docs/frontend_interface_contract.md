# Frontend Interface Contract

## Overview

This project now exposes one unified harness API contract and three domain-level API contracts:

- `study`
- `life`
- `sport`
- `harness` for cross-domain orchestration

These APIs are designed for frontend-triggered runs, latest-result polling, and top-level summary rendering.

## Base Principles

- All requests and responses use `application/json`
- `study`, `life`, and `sport` all expose `run`, mode-specific shortcuts, and a latest-result endpoint
- `harness` is the recommended frontend entrypoint when the UI needs a three-domain summary
- Domain `decision` should be rendered as the main business outcome, not `status`
- Metrics should be rendered from `summary_metrics` for domain-level pages and from `final_decision.domains.<domain>.trusted_mainline` for the system summary page

## Recommended Frontend Entry

Use `POST /harness/run` for the main workflow.

Use per-domain endpoints only when the UI needs a single-domain debug page.

## 1. Harness API

File:

- `C:\Users\39527\Documents\Playground\服创赛\harness\harness_api.py`

### 1.1 Health

`GET /health`

Response example:

```json
{
  "status": "ok",
  "service": "harness",
  "available_domains": ["study", "life", "sport"],
  "runs_dir": "C:\\Users\\39527\\Documents\\Playground\\服创赛\\data\\harness\\runs"
}
```

### 1.2 Run Domains

`POST /harness/run`

Request body:

```json
{
  "domains": ["study", "life", "sport"],
  "request": {
    "request_id": "frontend_run_001",
    "domain": "study",
    "run_mode": "review",
    "execution_engine": "harness_v1",
    "input_paths": {}
  }
}
```

Field notes:

- `domains`: required, non-empty array, values should be `study`, `life`, `sport`
- `request`: required, full request JSON object passed into the orchestrator
- The same `request` is used as the root harness request and each adapter will build its own domain request as needed

Response shape:

```json
{
  "run_record_path": "C:\\Users\\39527\\Documents\\Playground\\服创赛\\data\\harness\\runs\\multi_domain_study_life_sport.json",
  "result": {
    "run_type": "multi_domain",
    "system_status": "multi_domain_ready",
    "final_decision": {
      "decision": "multi_domain_completed",
      "domains": {
        "study": {
          "status": "completed_with_hold",
          "decision": "keep_baseline",
          "comparable": false,
          "mainline_task_type": "same_window_classification",
          "mainline_validity": true,
          "blocking_reason": "",
          "trusted_mainline": {},
          "mainline_frozen": false,
          "next_optimization_target": ""
        },
        "life": {
          "status": "success",
          "decision": "eligible_for_comparison",
          "comparable": true,
          "mainline_task_type": "future_window_prediction",
          "mainline_validity": true,
          "blocking_reason": "",
          "trusted_mainline": {
            "auc": 0.8632350795272142,
            "f1": 0.741176,
            "precision": 0.677419,
            "recall": 0.818182
          },
          "mainline_frozen": true,
          "next_optimization_target": "improve_future_window_auc"
        },
        "sport": {
          "status": "completed",
          "decision": "keep_baseline",
          "comparable": true,
          "mainline_task_type": "future_window_prediction",
          "mainline_validity": true,
          "blocking_reason": "",
          "trusted_mainline": {
            "auc": 0.8505509045419554,
            "f1": 0.3193717277486911,
            "precision": 0.23282442748091603,
            "recall": 0.5083333333333333,
            "best_threshold": 0.35000000000000003,
            "state_thresholds": {
              "borderline": 0.35000000000000003,
              "low_participation": 0.37000000000000005,
              "stable": 0.32000000000000006
            }
          },
          "mainline_frozen": true,
          "next_optimization_target": "improve_future_window_auc"
        }
      }
    },
    "domain_results": {}
  }
}
```

Frontend rendering priorities:

- System-level status: `result.system_status`
- System-level final decision: `result.final_decision.decision`
- Per-domain card title status: `result.final_decision.domains.<domain>.decision`
- Per-domain main metrics: `result.final_decision.domains.<domain>.trusted_mainline`

### 1.3 Latest Harness Result

`GET /harness/result/latest`

Response:

```json
{
  "run_record_path": "C:\\Users\\39527\\Documents\\Playground\\服创赛\\data\\harness\\runs\\multi_domain_study_life_sport.json",
  "result": {}
}
```

## 2. Study API

File:

- `C:\Users\39527\Documents\Playground\服创赛\study\src\study_agent_api.py`

Endpoints:

- `GET /health`
- `POST /study/run`
- `POST /study/train`
- `POST /study/infer`
- `POST /study/review`
- `POST /study/publish`
- `POST /study/rollback`
- `GET /study/result/{request_id}`
- `GET /study/evolution/latest`
- `GET /study/registry`
- `GET /study/serving`
- `GET /study/paths`

### 2.1 Study Run Request

```json
{
  "request_id": "study_ui_001",
  "domain": "study",
  "run_mode": "review",
  "execution_engine": "harness_v1",
  "input_paths": {}
}
```

### 2.2 Study Run Response Core Fields

```json
{
  "status": "completed_with_hold",
  "request_id": "study_ui_001",
  "summary_metrics": {
    "auc": 0.8324450440706692,
    "f1": 0.540837336993823,
    "recall": 0.7255985267034991
  },
  "harness_v1": {
    "final_decision": "keep_baseline"
  }
}
```

## 3. Life API

File:

- `C:\Users\39527\Documents\Playground\服创赛\life\src\life_agent_api.py`

Endpoints:

- `GET /health`
- `POST /life/run`
- `POST /life/train`
- `POST /life/infer`
- `POST /life/review`
- `GET /life/result/latest`

### 3.1 Life Run Request

```json
{
  "request_id": "life_ui_001",
  "domain": "life",
  "run_mode": "review",
  "execution_engine": "harness_v1",
  "input_paths": {}
}
```

### 3.2 Life Response Core Fields

```json
{
  "status": "success",
  "summary_metrics": {
    "auc": 0.8632350795272142,
    "f1": 0.741176,
    "precision": 0.677419,
    "recall": 0.818182,
    "positive_rate": 0.282353
  },
  "harness_v1": {
    "decision": "eligible_for_comparison",
    "trusted_mainline": {
      "label_version": "instability_future_v2",
      "feature_bundle": "regularity+volatility+coupling",
      "split_version": "purged_temporal_split"
    }
  }
}
```

## 4. Sport API

File:

- `C:\Users\39527\Documents\Playground\服创赛\sport\src\sport_agent_api.py`

Endpoints:

- `GET /health`
- `POST /sport/run`
- `POST /sport/train`
- `POST /sport/infer`
- `POST /sport/review`
- `GET /sport/result/latest`

### 4.1 Sport Run Request

```json
{
  "request_id": "sport_ui_001",
  "domain": "sport",
  "run_mode": "review",
  "execution_engine": "harness_v1",
  "input_paths": {
    "feature_dataset": "data/deliverables/sport/data/sport_feature_dataset.csv",
    "prediction_output": "data/deliverables/sport/data/sport_prediction_output.csv",
    "prediction_test_output": "data/deliverables/sport/data/sport_prediction_test_output.csv",
    "quality_report": "data/deliverables/sport/docs/sport_quality_report.json",
    "validation_report": "data/deliverables/sport/docs/sport_validation_report.json",
    "model_regression": "data/deliverables/sport/model/sport_regression_model.joblib",
    "model_classification": "data/deliverables/sport/model/sport_classification_model.joblib",
    "model_config": "data/deliverables/sport/model/sport_model_config.json",
    "metrics": "data/deliverables/sport/data/metrics.json"
  }
}
```

### 4.2 Sport Response Core Fields

```json
{
  "status": "success",
  "domain": "sport",
  "run_mode": "review",
  "summary_metrics": {
    "auc": 0.8505509045419554,
    "f1": 0.3193717277486911,
    "precision": 0.23282442748091603,
    "recall": 0.5083333333333333,
    "rows": 3545,
    "eval_rows": 1852,
    "positive_count": 120,
    "future_window_auc": 0.8505509045419554
  },
  "harness_v1": {
    "decision": "keep_baseline",
    "comparable": true,
    "trusted_mainline": {
      "label_version": "future_v3",
      "feature_bundle": "baseline+deviation+trend",
      "structure_version": "two_stage",
      "population_version": "recoverable",
      "best_threshold": 0.35000000000000003,
      "state_thresholds": {
        "borderline": 0.35000000000000003,
        "low_participation": 0.37000000000000005,
        "stable": 0.32000000000000006
      }
    }
  }
}
```

## 5. Decision Vocabulary

The frontend should use these values as display enums:

- `promote_candidate`
- `keep_baseline`
- `eligible_for_comparison`
- `hold_for_review`

Recommended Chinese display mapping:

- `promote_candidate` -> `提升候选`
- `keep_baseline` -> `保持基线`
- `eligible_for_comparison` -> `可进入比较`
- `hold_for_review` -> `暂缓，待复核`

## 6. Frontend Display Mapping

### System Summary Page

Use:

- `result.system_status`
- `result.final_decision.decision`
- `result.final_decision.domains`

### Domain Detail Page

Use:

- `domain_results.<domain>.status`
- `domain_results.<domain>.metrics`
- `domain_results.<domain>.metric_context`
- `domain_results.<domain>.raw_result.harness_v1`

### Model Metrics Block

Preferred field order:

1. `auc`
2. `f1`
3. `precision`
4. `recall`
5. `rows`
6. `eval_rows`
7. `positive_count`

## 7. Important Notes for Frontend

- `status` and `decision` are not the same thing
- `status` means execution state, such as `success` or `completed`
- `decision` means business recommendation, such as `keep_baseline`
- `study` may be valid but still `keep_baseline`
- `life` is the current strongest frontend-ready domain for balanced future-window metrics
- `sport` is a valid future-window domain, but the positive class is sparse, so threshold metrics are lower than life

## 8. Latest Verified Metrics

Latest verified three-domain run:

- `study`: `auc=0.8324450440706692`, `f1=0.540837336993823`, `decision=keep_baseline`
- `life`: `auc=0.8632350795272142`, `f1=0.741176`, `precision=0.677419`, `recall=0.818182`, `decision=eligible_for_comparison`
- `sport`: `auc=0.8505509045419554`, `f1=0.3193717277486911`, `precision=0.23282442748091603`, `recall=0.5083333333333333`, `decision=keep_baseline`

Source run file:

- `C:\Users\39527\Documents\Playground\服创赛\data\harness\runs\multi_domain_study_life_sport.json`
