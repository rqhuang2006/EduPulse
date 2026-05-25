# Harness Domain Onboarding Guide

This document explains how to add a new domain (e.g., `sport`) to the harness multi-domain orchestration framework.

---

## 1. Directory Structure

Create a new domain root (e.g., `sport/`) with this layout:

```
sport/
  conf/
    feature_registry.yaml      # Feature column definitions
    file_registry.yaml         # Raw data file paths (relative to workspace root)
    field_alias.yaml           # Field name mappings (e.g., student_id aliases)
    model_params.yaml          # Model hyperparameters
    package_manifest.yaml      # Deliverables listing
    sport_agent_policy.yaml    # Agent behavior policy (defaults)
    sport_release_policy.yaml  # Release / baseline comparison policy
  input/
    sport_agent_request.harness_v1.review.json   # Minimum request file
  src/
    sport_agent.py             # Domain agent entrypoint (standalone runnable)
  data/
    deliverables/sport/
      data/                    # train_table.csv, infer_table.csv, prediction_output.csv
      docs/                    # quality_report.json, validation_report.json, feature_dictionary.csv
      model/                   # model.pkl, model_config.json, model_metrics.json
    dm/                        # Working outputs: sport_agent_result.json, eval reports
    registry/sport/            # Snapshot versions (baseline_index.json + version dirs)
```

### Key notes:
- `conf/` files are YAML and loaded by the agent at runtime.
- `input/` contains request JSON files that drive the harness workflow.
- `data/deliverables/<domain>/` holds the final production artifacts.
- `data/dm/` holds intermediate run outputs (eval reports, agent results).
- `data/registry/<domain>/` holds frozen baseline snapshots for comparison.

---

## 2. BaseDomainAgent Contract

Every domain agent adapter under `harness/domain_agents/<domain>/` must implement `BaseDomainAgent` from `harness/domain_agents/base/base_domain_agent.py`.

### Required abstract methods:

| Method | Purpose | Return keys |
|--------|---------|-------------|
| `train(request, context)` | Domain-specific training | `status`, `metrics`, `artifacts` |
| `eval(request, context)` | Domain-specific evaluation | `status`, `metrics`, `artifacts` |
| `predict(request, context)` | Domain-specific inference | `status`, `metrics`, `artifacts` |
| `build_candidate(request, context)` | Build candidate artifact for evaluation | `status`, `metrics`, `artifact_ref` |
| `load_baseline(request, context)` | Load active/anchor baselines | `baseline_version_id`, `anchor_baseline_version_id`, `baseline_metrics`, `anchor_baseline_metrics` |
| `get_contract_context(request, context)` | Build `ContractContext` for same-caliber/comparison | `ContractContext` dataclass |
| `get_metric_pack(request, context)` | Return metric pack for policy evaluation | `candidate_metrics`, `baseline_metrics`, `metric_context` |
| `get_local_gain_signals(request, context)` | Domain-specific gain signals | `{signal_name: bool}` dict |
| `export_fusion_payload(request, context)` | Export normalized `FusionInputContract` | `FusionInputContract` dataclass |
| `run_domain_pipeline(request)` | **Single entrypoint** for orchestrator | See below |

### `run_domain_pipeline` return dict:

```python
{
    "domain_name": "sport",
    "status": "success" | "failed" | "stub",
    "system_status": "multi_domain_ready" | "partial_domain_ready" | "completed_with_hold" | "stub",
    "final_decision": "promotion_recommended" | "keep_baseline" | "dry_run_only" | ...,
    "policy_decision": "promotion_recommended" | "keep_baseline" | "dry_run_only" | ...,
    "execution_mode": "dry_run" | "production",
    "decision_stage_reached": "approval_release_stage" | "floor_gate" | ...,
    "metrics": {"auc": 0.82, "f1": 0.75, ...},
    "metric_context": {"eval_scope": "...", "feature_contract_hash": "..."},
    "domain_context": {"domain": "sport", ...},       # domain-specific fields
    "domain_audit": {"stub": False, ...},             # domain-specific audit info
    "validation_summary": {...},
    "warning_summary": [...],
    "artifact_ref": {"model_config": "path", ...},
}
```

### Stub agents:
If a domain is not yet implemented, `run_domain_pipeline` should return a structured stub result (not raise `NotImplementedError`):

```python
def run_domain_pipeline(self, request):
    return {
        "domain_name": "sport",
        "status": "stub",
        "system_status": "stub",
        "final_decision": "dry_run_only",
        "policy_decision": "dry_run_only",
        "execution_mode": "dry_run",
        "decision_stage_reached": "contract_chain_gate",
        "metrics": {},
        "metric_context": {"note": "Stub - no metrics"},
        "domain_context": {"domain_name": "sport", "stub": True},
        "domain_audit": {"stub_reason": "agent_not_implemented"},
        "validation_summary": {"is_stub": True},
        "warning_summary": ["SportAgent is a stub"],
        "artifact_ref": {},
    }
```

### `export_fusion_payload` for stubs:

```python
def export_fusion_payload(self, request, context):
    return FusionInputContract(
        domain_name="sport",
        risk_level="stub",
        warning_summary=["SportAgent is a stub"],
        metric_context={"note": "Stub - not yet implemented"},
    )
```

---

## 3. Request Minimum Format

The request JSON file (e.g., `sport/input/sport_agent_request.harness_v1.review.json`) must contain:

```json
{
  "request_id": "sport_review_request_v1",
  "domain": "sport",
  "term_id": "all",
  "run_mode": "review",
  "execution_engine": "harness_v1",
  "feature_version": "sport_feature_v1",
  "model_version": "sport_v1",
  "enable_fallback": true,
  "enable_explanation": false,
  "input_paths": {
    "train_table": "data/deliverables/sport/data/train_table.csv",
    "infer_table": "data/deliverables/sport/data/infer_table.csv",
    "prediction_output": "data/deliverables/sport/data/prediction_output.csv",
    "quality_report": "data/deliverables/sport/docs/quality_report.json",
    "validation_report": "data/deliverables/sport/docs/validation_report.json",
    "feature_dictionary": "data/deliverables/sport/docs/feature_dictionary.csv",
    "model_file": "data/deliverables/sport/model/model.pkl",
    "model_config": "data/deliverables/sport/model/model_config.json"
  }
}
```

### Required fields:
- `request_id` - unique identifier
- `domain` - domain name (must match `domain_name` property)
- `run_mode` - e.g., `review`, `train`, `infer`
- `execution_engine` - must be `harness_v1` for compatibility
- `input_paths` - paths to domain deliverables (relative to domain root)

### Optional fields:
- `feature_version`, `model_version` - version tracking
- `enable_fallback`, `enable_explanation` - feature toggles
- `llm_*` fields - LLM configuration (if applicable)

---

## 4. Deliverables Minimum Requirements

Each domain must produce these artifacts under `data/deliverables/<domain>/`:

### `data/` directory:
| File | Required | Description |
|------|----------|-------------|
| `train_table.csv` | Yes | Training data with features + label |
| `infer_table.csv` | If inference supported | Holdout/inference data |
| `prediction_output.csv` | Yes | Predictions with risk_score, prediction, risk_level |

### `docs/` directory:
| File | Required | Description |
|------|----------|-------------|
| `quality_report.json` | Yes | Data quality metrics (missing rates, distributions) |
| `validation_report.json` | Yes | Validation status and check results |
| `feature_dictionary.csv` | Yes | Feature names and types |

### `model/` directory:
| File | Required | Description |
|------|----------|-------------|
| `model.pkl` | Yes | Serialized model artifact |
| `model_config.json` | Yes | Model metadata (name, features, label) |
| `model_metrics.json` | Yes | `{"summary_metrics": {"auc": ..., "f1": ..., ...}}` |

---

## 5. Snapshot Contract

Every baseline/candidate snapshot under `data/registry/<domain>/<version_id>/` must contain these 5 files:

| File | Description |
|------|-------------|
| `model_config.json` | Model metadata, version, feature columns |
| `feature_config.json` | Feature contract, column list, hash |
| `contract_context.json` | Task scope, label definition, eval scope |
| `domain_audit.json` | Domain-specific audit information |
| `metrics.json` | `{"summary_metrics": {...}}` |

### Use `ensure_snapshot_contract()` from `harness.registry.snapshot_contract`:

```python
from harness.registry.snapshot_contract import ensure_snapshot_contract

manifest = ensure_snapshot_contract(
    snapshot_dir,
    version_id="sport_v1",
    domain="sport",
    payloads={
        "model_config.json": {...},
        "feature_config.json": {...},
        "contract_context.json": {...},
        "domain_audit.json": {...},
        "metrics.json": {"schema_version": "...", "summary_metrics": {...}},
    },
)
```

This ensures:
- All 5 files are written (no missing files)
- Empty payloads fall back to non-empty defaults
- Manifest paths are returned for indexing

### Registry index:
Create `data/registry/<domain>/baseline_index.json`:

```json
{
  "domain": "sport",
  "active_baseline_version_id": "sport_v1",
  "anchor_baseline_version_id": "sport_v1",
  "updated_at": "2026-04-19T12:00:00"
}
```

---

## 6. Fusion Payload Minimum Requirements

Every domain must export a `FusionInputContract` via `export_fusion_payload()`:

```python
FusionInputContract(
    domain_name="sport",
    candidate_version_id="sport_candidate_v1",
    risk_score=0.82,            # float or None (if stub)
    risk_level="low",           # "low", "medium", "high", "critical", "stub"
    confidence=0.7,             # float or None
    top_features=[...],         # list of feature dicts
    explanations=[...],         # list of explanation dicts
    metric_context={...},       # eval_scope, feature_contract_hash, etc.
    validation_summary={...},   # validation status
    warning_summary=[...],      # list of warning strings
    artifact_ref={...},         # path references to artifacts
    raw_payload={...},          # domain-specific data
)
```

### Validation rules (from `validate_fusion_input()`):
- `domain_name` must be non-empty string
- `risk_level` must be one of: `"low"`, `"medium"`, `"high"`, `"critical"`, `"stub"`, or `None`
- `risk_score` can be `None` (emits warning if `allow_stub=False`)
- `confidence` must be numeric or `None`
- `explanations` must be a list
- `artifact_ref` must be a dict
- `validation_summary` must be a dict
- `warning_summary` must be a list

---

## 7. Field Placement: Harness vs Domain

### Harness-generic fields (top-level in run records / policy / context):

These fields are consumed by harness core and must be present:

- `run_id`
- `pipeline_name`
- `domain`
- `eval_scope`
- `task_scope`
- `feature_contract_hash`
- `label_definition`
- `baseline_version_id`
- `anchor_baseline_version_id`
- `comparison_mode`
- `decision_stage_reached`
- `policy_decision`
- `final_decision`
- `execution_mode`
- `collected_warnings`

### Domain-specific fields (must go under `domain_context`, `domain_audit`, or `metric_context`):

**Do NOT** add domain-specific semantics to harness core dataclasses. Instead, put them under:

- `domain_context` - domain-specific configuration, label strategy, source datasets
- `domain_audit` - domain-specific audit info, degraded conditions, bootstrap flags
- `metric_context` - domain-specific metric metadata, sample counts, feature hashes

Examples of fields that should be domain-specific:
- `study_data_mode`, `row_level_study_data_mode` (study-specific)
- `label_strategy`, `source_datasets` (any domain)
- `bootstrap_mode`, `degraded_conditions` (any domain)
- Future sport-specific metrics or flags

---

## 8. Registration

After implementing the domain agent:

1. Create `harness/domain_agents/<domain>/__init__.py` exporting the adapter class
2. Add the import to `harness/run_multi_domain.py`:
   ```python
   from harness.domain_agents.sport import SportAgentStub
   ```
3. Add to the orchestrator's agent list:
   ```python
   agents=[
       StudyAgentAdapter(root / "study"),
       LifeAgentAdapter(root),
       SportAgentAdapter(root / "sport"),  # new
   ]
   ```

---

## 9. Testing

Add domain-specific tests to `harness/tests/`:

1. **Unit tests** - test the domain agent's methods in isolation
2. **Integration tests** - test the agent through the orchestrator
3. **Smoke tests** - test end-to-end with real data (if available)

Use dummy agents for orchestrator tests that don't require real data.

---

## 10. Quick Start Checklist

- [ ] Create domain directory structure
- [ ] Write conf files (feature, file, policy, model params)
- [ ] Create request JSON in `input/`
- [ ] Implement domain agent (`src/<domain>_agent.py`)
- [ ] Implement harness adapter (`harness/domain_agents/<domain>/<domain>_agent_adapter.py`)
- [ ] Ensure deliverables are produced (data, docs, model)
- [ ] Ensure snapshot is written to registry
- [ ] Register adapter in `run_multi_domain.py`
- [ ] Add tests
- [ ] Run `python -m compileall harness <domain>` to verify syntax
- [ ] Run single-domain test
- [ ] Run multi-domain orchestrator
