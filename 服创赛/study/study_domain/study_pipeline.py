from __future__ import annotations

from pathlib import Path

from harness.pipeline_spec import PipelineSpec
from harness.recorder.run_recorder import RunRecorder
from harness.registry.artifact_registry import ArtifactRegistry
from harness.runner import HarnessRunner
from study_domain.actions import CaliberAction, DiagnoseAction, EvalAction, PublishAction, RollbackAction, TrainAction
from study_domain.policy.selection_policy import RuleBasedSelectionPolicy


def build_study_pipeline(root_dir: Path, request: dict | None = None) -> HarnessRunner:
    request = request or {}
    run_mode = request.get("run_mode", "train")
    pre_policy_actions = [EvalAction(root_dir), DiagnoseAction(root_dir), CaliberAction(root_dir)]
    if run_mode == "train":
        pre_policy_actions.insert(0, TrainAction(root_dir))
    spec = PipelineSpec(
        name="study_harness_v1",
        pre_policy_actions=pre_policy_actions,
        post_policy_actions=[
            PublishAction(root_dir),
            RollbackAction(root_dir),
        ],
    )
    return HarnessRunner(
        spec=spec,
        policy=RuleBasedSelectionPolicy(),
        recorder=RunRecorder(root_dir),
        artifact_registry=ArtifactRegistry(root_dir),
    )
