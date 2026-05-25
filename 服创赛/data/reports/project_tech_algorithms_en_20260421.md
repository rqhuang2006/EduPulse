# High-Value Technologies and Algorithms Across the Three-Domain Project

Generated at: 2026-04-21 08:48:18

## Executive Summary

This project is no longer just a collection of separate models. It is a multi-domain decision and governance system spanning study, life, and sport under one harness.
Its strongest technical value comes from combining modeling, evaluation, baseline-vs-candidate governance, artifact management, and auditable decision semantics into one workflow.

## 1. What Makes This Project Technically Valuable

- Multi-domain orchestration instead of isolated scripts
- Domain-specific modeling with a shared contract layer
- Baseline / candidate / snapshot / registry governance
- Authenticity-aware evaluation rather than score chasing
- Standardized domain outputs for cross-domain comparison

## 2. Core Engineering and ML Stack

- Python for the full execution pipeline
- pandas and numpy for tabular processing, window statistics, aggregation, and feature construction
- JSON and YAML for request contracts, configuration, metrics, and artifact manifests
- joblib for model persistence and reproducible artifact export
- scikit-learn Pipeline and ColumnTransformer for reusable preprocessing-and-model workflows
- SimpleImputer, StandardScaler, and OneHotEncoder for robust tabular preprocessing
- LogisticRegression, RandomForest, LightGBM, XGBoost, CatBoost, and HistGradientBoosting as candidate model families used across domains

## 3. System-Level In-House Design

### 3.1 Harness-Orchestrated Multi-Domain Execution
The harness runs study, life, and sport under a shared orchestration layer. It injects search contracts, collects normalized outputs, and produces a unified three-domain summary.

### 3.2 Adapter and Contract Architecture
Each domain is integrated through an adapter contract rather than hard-coded orchestration logic. This is what makes the system extensible.

### 3.3 Snapshot and Registry Governance
Every serious candidate is tied to snapshot metadata such as model config, feature config, contract context, domain audit, and metrics. This turns experiments into auditable assets instead of disposable runs.

### 3.4 Artifact Export / Mirror Logic
The project includes explicit artifact export logic so that training outputs are mirrored into the exact infer-time contract paths, with post-copy existence checks.

### 3.5 Standardized Result Normalization
Domain outputs are normalized into a consistent structure, allowing the harness to reason about all domains using shared semantics.

### 3.6 Agent Protocol
Life and sport are no longer treated as simple model runners. They are being shaped into four-stage optimization agents with diagnosis, proposal, comparison, and recommendation.

### 3.7 Unified Decision Vocabulary
The project already uses unified decision labels such as promote_candidate, keep_baseline, eligible_for_comparison, and hold_for_review. This is important for auditability and for external reporting.

## 4. Study Domain: Mature Methodology Source

Study is the most mature domain and the main methodological reference for the others.
- Unified experiment framework with fixed holdout IDs, fixed train IDs, fixed label semantics, and fixed engineering order
- Leakage-aware experimentation, including explicit leakage audit and routing damage decomposition
- Strong baseline governance and clear reasoning when the system should keep the baseline instead of promoting a candidate
- Current candidate metrics: AUC 0.8324, F1 0.5408, Recall 0.7256
- Current baseline metrics: AUC 0.8419, F1 0.6672, Recall 0.7374, Precision 0.6092

## 5. Life Domain: Authenticity-First Searcher

Life used to be vulnerable to suspiciously high in-distribution AUC. The current system now reports the trusted mainline rather than surfacing inflated same-distribution scores as the primary result.
- Multi-source behavioral fusion: student profile, internet, club, library, and gate data
- Feature families centered on regularity, volatility, deviation, rhythm, and coupling
- Source-group ablation and honest-candidate logic to reduce proxy-source dominance
- Temporal generalization and trust-score based gating
- Proxy track retained for reference, instability track used as the main decision track
- Current trusted life mainline: AUC 0.8473, F1 0.8693, Precision 1.0000, Recall 0.7689
- Current life decision: eligible_for_comparison

## 6. Sport Domain: Future-Window Predictability Searcher

Sport has been shifted away from misleading top-line serving-style scores and toward a trusted future-window mainline output.
- Multi-source sport panel construction across physical-test data, PE course data, daily exercise data, and running check-ins
- Term inference and rolling temporal evaluation
- Label evolution and feature-bundle search
- Population search over all, active-courses, and recoverable populations
- Future-window mainline selection under realism gates
- Current trusted sport mainline: AUC 0.8493, F1 0.3246, Precision 0.2366, Recall 0.5167
- Current sport decision: keep_baseline
- Mainline configuration: future_v3, baseline+deviation+trend, recoverable, rows=3545, eval_rows=1852, positive_count=120

## 7. Current Three-Domain Status

- System status: multi_domain_ready
- Ready domains: study, life, sport
- Study decision: keep_baseline
- Life decision: eligible_for_comparison
- Sport decision: keep_baseline

## 8. Submission Readiness Assessment

This version is ready to submit as a strong current-stage deliverable.
Reasons:
- The most important reporting issue has been fixed: life and sport no longer expose suspiciously high AUC values as their primary top-line outputs.
- Life now surfaces the trusted temporal-generalization mainline around 0.847 instead of the old ~0.98 in-distribution reference score.
- Sport now surfaces the trusted future-window mainline around 0.849 instead of the old ~0.99 serving-style reference score.
- The three-domain harness summary is now coherent enough for review, demo, and milestone submission.
Caveat: this is a submit-ready milestone build, not a final production-perfect research endpoint. In particular, sport still has room to improve the future-window classification tradeoff even though its AUC is now reported correctly and credibly.

## 9. Strong Resume / Presentation Statements

- Built a multi-domain modeling harness spanning study, life, and sport with unified request contracts, adapters, normalized outputs, and cross-domain decision summaries.
- Designed candidate/baseline/snapshot/registry governance for auditable model lifecycle management.
- Introduced authenticity-aware evaluation for non-study domains, preventing misleading high AUC from dominating top-line reporting.
- Evolved domain runners into optimization agents capable of diagnosis, proposal, comparison, and recommendation.

## 10. Key Files

- harness/domain_agents/README.md
- harness/contracts/policy.py
- harness/artifact_manager.py
- harness/domain_support/result_normalizer.py
- harness/domain_support/agent_protocol.py
- harness/domain_agents/orchestrator.py
- study/FINAL_EXPERIMENT_REPORT.md
- study/src/40_l3_leakage_audit.py
- study/src/41_routing_damage_decomposition.py
- study/src/42_unified_experiment_manifest.py
- life/src/life_agent.py
- life/src/life_validator.py
- sport/sportagent/train_domain_models.py
- sport/sportagent/sport_agent.py
- data/harness/runs/multi_domain_study_life_sport.json
