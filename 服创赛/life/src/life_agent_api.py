from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .life_agent import RESULT_PATH, LifeAgent


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
        return LifeAgent(request_path=request_path).run()
    finally:
        request_path.unlink(missing_ok=True)


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


if FastAPI is None:
    app = MissingFastAPIApp()
else:
    app = FastAPI(title="LifeAgent API", description="Life domain agent API.", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        latest_exists = RESULT_PATH.exists()
        return {
            "status": "ok",
            "domain": "life",
            "available_modes": ["train", "infer", "review"],
            "last_result_status": "available" if latest_exists else "missing",
        }

    @app.post("/life/run")
    def life_run(payload: dict[str, Any]) -> dict[str, Any]:
        mode = payload.get("run_mode")
        if mode not in {"train", "infer", "review"}:
            raise HTTPException(status_code=400, detail="run_mode must be train/infer/review")
        return run_agent_from_payload(payload)

    @app.post("/life/train")
    def life_train(payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["run_mode"] = "train"
        return run_agent_from_payload(payload)

    @app.post("/life/infer")
    def life_infer(payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["run_mode"] = "infer"
        return run_agent_from_payload(payload)

    @app.post("/life/review")
    def life_review(payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        payload["run_mode"] = "review"
        return run_agent_from_payload(payload)

    @app.get("/life/result/latest")
    def life_result_latest() -> dict[str, Any]:
        return read_json_if_exists(RESULT_PATH)
