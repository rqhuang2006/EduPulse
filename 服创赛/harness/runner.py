from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from harness.contracts import PipelineContext, RunRecord
from harness.pipeline_spec import PipelineSpec
from harness.recorder.run_recorder import RunRecorder
from harness.registry.artifact_registry import ArtifactRegistry


class HarnessRunner:
    def __init__(self, spec: PipelineSpec, policy, recorder: RunRecorder, artifact_registry: ArtifactRegistry):
        self.spec = spec
        self.policy = policy
        self.recorder = recorder
        self.artifact_registry = artifact_registry

    def run(self, request: dict, root_dir: Path) -> tuple[RunRecord, Path]:
        run_id = request.get("run_id") or f"{self.spec.name}_{uuid4().hex[:12]}"

        domain = request.get("domain")
        if not domain:
            raise ValueError("HarnessRunner requires explicit request['domain']")

        release_cfg = request.get("release_config", {})
        context_config = {
            "dry_run": release_cfg.get("dry_run", True),
            "require_approval": release_cfg.get("require_approval", True),
            "release": release_cfg,
        }

        context = PipelineContext(
            run_id=run_id,
            pipeline_name=self.spec.name,
            domain=domain,
            request=request,
            root_dir=root_dir,
            config=context_config,
            metadata={"started_at": datetime.now().isoformat(timespec="seconds")},
        )

        for action in self.spec.pre_policy_actions:
            result = action.run(context)
            context.add_result(result)
            if result.status == "failed":
                break

        decision = self.policy.decide(context, context.stage_results)
        context.final_decision = decision

        for action in self.spec.post_policy_actions:
            result = action.run(context)
            context.add_result(result)

        self.artifact_registry.write_manifest(context.run_id, context.artifacts)
        return self.recorder.finalize(context, decision)