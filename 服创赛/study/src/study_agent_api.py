from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .study_agent import DELIVERABLE_RESULT_PATH, ROOT, StudyAgent, json_default
from .study_release_manager import CURRENT_SERVING_PATH, MODEL_REGISTRY_PATH, StudyReleaseManager

DM_DIR = ROOT / "data" / "dm"

try:
    from fastapi import FastAPI, HTTPException
except ModuleNotFoundError:  # pragma: no cover
    FastAPI = None
    HTTPException = Exception


class MissingFastAPIApp:
    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            return
        body = json.dumps({"status": "failed", "message": "FastAPI is not installed."}).encode("utf-8")
        await send({"type": "http.response.start", "status": 503, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})


def run_agent_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    with NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
        request_path = Path(handle.name)
    try:
        return StudyAgent(request_path=request_path).run()
    finally:
        request_path.unlink(missing_ok=True)


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


if FastAPI is None:
    app = MissingFastAPIApp()
else:
    app = FastAPI(title="StudyAgent API", description="Controlled self-evolving study domain agent.", version="2.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        serving = StudyReleaseManager({}).current_serving()
        latest_evolution = read_json_if_exists(DM_DIR / "study_evolution_selection.json") if (DM_DIR / "study_evolution_selection.json").exists() else {}
        return {
            "status": "ok",
            "domain": "study",
            "available_modes": ["train", "infer", "review", "publish", "rollback"],
            "current_serving_version": serving.get("current_version_id"),
            "last_evolution_status": "available" if latest_evolution else "missing",
        }

    @app.post("/study/run")
    def study_run(payload: dict[str, Any]) -> dict[str, Any]:
        mode = payload.get("run_mode")
        if mode not in {"train", "infer", "review", "publish", "rollback"}:
            raise HTTPException(status_code=400, detail="run_mode must be train/infer/review/publish/rollback")
        return run_agent_from_payload(payload)

    @app.post("/study/train")
    def study_train(payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["run_mode"] = "train"
        payload.setdefault("evolution_enable", True)
        return run_agent_from_payload(payload)

    @app.post("/study/infer")
    def study_infer(payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["run_mode"] = "infer"
        payload.setdefault("serving_version", "latest")
        return run_agent_from_payload(payload)

    @app.post("/study/review")
    def study_review(payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["run_mode"] = "review"
        payload.setdefault("llm_enable", True)
        return run_agent_from_payload(payload)

    @app.post("/study/publish")
    def study_publish(payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["run_mode"] = "publish"
        payload.setdefault("dry_run", True)
        return run_agent_from_payload(payload)

    @app.post("/study/rollback")
    def study_rollback(payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["run_mode"] = "rollback"
        payload.setdefault("dry_run", True)
        return run_agent_from_payload(payload)

    @app.get("/study/result/{request_id}")
    def study_result(request_id: str) -> dict[str, Any]:
        result = read_json_if_exists(DELIVERABLE_RESULT_PATH)
        if result.get("request_id") != request_id:
            raise HTTPException(status_code=404, detail=f"latest result request_id is {result.get('request_id')}, not {request_id}")
        return result

    @app.get("/study/evolution/latest")
    def latest_evolution() -> dict[str, Any]:
        return {
            "selection": read_json_if_exists(DM_DIR / "study_evolution_selection.json"),
            "publish_candidate": read_json_if_exists(DM_DIR / "study_evolution_publish_candidate.json"),
            "comparison_path": str(DM_DIR / "study_evolution_comparison.csv"),
        }

    @app.get("/study/registry")
    def study_registry() -> dict[str, Any]:
        return StudyReleaseManager({}).registry_summary()

    @app.get("/study/serving")
    def study_serving() -> dict[str, Any]:
        return read_json_if_exists(CURRENT_SERVING_PATH)

    @app.get("/study/paths")
    def study_paths() -> dict[str, Any]:
        return {"root": str(ROOT), "latest_result": str(DELIVERABLE_RESULT_PATH), "registry": str(MODEL_REGISTRY_PATH)}
