# Harness Action Contract

## TrainAction
- Input: training request, train table, label config
- Output: model bundle, model config, model metrics, eval report
- Failure conditions: missing train table, missing label, artifact missing

## EvalAction
- Input: candidate artifacts from TrainAction, active/anchor baseline registry
- Output: overall metrics, subgroup metrics, confidence zone metrics, baseline delta
- Failure conditions: eval report missing, metric floor blocked

## DiagnoseAction
- Input: feature screening, subgroup screening, quality report
- Output: structured diagnostic summary and evidence artifacts
- Failure conditions: diagnostic artifacts missing

## SameCaliberAction
- Input: active baseline, candidate version, release-manager comparability checks
- Output: explicit same-caliber validation result before selection
- Failure conditions: candidate missing, comparability mismatch, chain incompleteness

## PublishAction
- Input: policy decision, candidate version context
- Output: publish dry-run result with structured reasons
- Failure conditions: release manager dry-run failure

## RollbackAction
- Input: active baseline and rollback target context
- Output: rollback dry-run result with recovery target evidence
- Failure conditions: no stable rollback target
