from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from sport.sportagent.sport_agent import REPO_ROOT, SportAgent


SNAPSHOT_PATH = REPO_ROOT / "data" / "deliverables" / "sport" / "docs" / "sport_snapshot.json"
METRICS_PATH = REPO_ROOT / "data" / "deliverables" / "sport" / "data" / "metrics.json"
PREDICTION_OUTPUT_PATH = REPO_ROOT / "data" / "deliverables" / "sport" / "data" / "sport_prediction_output.csv"

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
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        request_path = Path(handle.name)
    try:
        agent = SportAgent.from_request_file(request_path)
        return agent.run()
    finally:
        request_path.unlink(missing_ok=True)


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


if FastAPI is None:
    app = MissingFastAPIApp()
else:
    app = FastAPI(title="SportAgent API", description="Sport domain agent API.", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        latest_exists = SNAPSHOT_PATH.exists()
        return {
            "status": "ok",
            "domain": "sport",
            "available_modes": ["train", "infer", "review"],
            "last_snapshot_status": "available" if latest_exists else "missing",
        }

    @app.post("/sport/run")
    def sport_run(payload: dict[str, Any]) -> dict[str, Any]:
        mode = payload.get("run_mode")
        if mode not in {"train", "infer", "review"}:
            raise HTTPException(status_code=400, detail="run_mode must be train/infer/review")
        return run_agent_from_payload(payload)

    @app.post("/sport/train")
    def sport_train(payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["run_mode"] = "train"
        return run_agent_from_payload(payload)

    @app.post("/sport/infer")
    def sport_infer(payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["run_mode"] = "infer"
        return run_agent_from_payload(payload)

    @app.post("/sport/review")
    def sport_review(payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["run_mode"] = "review"
        return run_agent_from_payload(payload)

    @app.get("/sport/result/latest")
    def sport_result_latest() -> dict[str, Any]:
        snapshot = read_json_if_exists(SNAPSHOT_PATH)
        metrics = read_json_if_exists(METRICS_PATH) if METRICS_PATH.exists() else {}
        return {
            "domain": "sport",
            "snapshot": snapshot,
            "metrics": metrics,
            "prediction_output": str(PREDICTION_OUTPUT_PATH),
        }
