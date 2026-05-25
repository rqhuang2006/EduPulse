# Selection Policy Spec

## Reject
- Any blocking validator fails
- Candidate AUC below hard floor
- Required artifacts missing

## Dry Run Only
- Error-level validations exist
- Overall signal is usable but subgroup or stability evidence is insufficient

## Accept
- Overall AUC is not worse than baseline by more than `0.003`
- At least one local gain is confirmed
- No blocking or error validations remain

## Rollback Recommended
- Reserved for future active-serving degradation handling
- Current v1 always records rollback dry-run evidence
