from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
except ModuleNotFoundError:  # pragma: no cover
    FastAPI = None
    HTTPException = Exception


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path(os.getenv("HARNESS_RUNS_DIR", str(ROOT / "data" / "harness" / "runs")))
FRONTEND_ROOT = Path(os.getenv("FRONTEND_ROOT", str(ROOT.parent / "1")))
FRONTEND_A14_DIR = Path(os.getenv("FRONTEND_A14_DIR", str(FRONTEND_ROOT / "outputs" / "a14")))
BUNDLE_DIR = Path(os.getenv("FRONTEND_BUNDLE_DIR", str(ROOT / "data" / "harness" / "frontend_bundle")))
BUNDLE_PATH = Path(os.getenv("FRONTEND_BUNDLE_PATH", str(BUNDLE_DIR / "latest_frontend_bundle.json")))


class MissingFastAPIApp:
    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            return
        body = json.dumps({"status": "failed", "message": "FastAPI is not installed."}).encode("utf-8")
        await send({"type": "http.response.start", "status": 503, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})


def _bootstrap() -> None:
    for path in [ROOT / ".deps3", ROOT]:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def _read_latest_run() -> dict[str, Any]:
    if not RUNS_DIR.exists():
        raise FileNotFoundError(str(RUNS_DIR))
    candidates = sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(str(RUNS_DIR))
    preferred_terminal: tuple[Path, dict[str, Any]] | None = None
    preferred_running: tuple[Path, dict[str, Any]] | None = None
    fallback: tuple[Path, dict[str, Any]] | None = None
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if fallback is None and path.name.startswith("multi_domain"):
            fallback = (path, payload)
        final_decision = payload.get("final_decision", {})
        domains = final_decision.get("domains", {}) if isinstance(final_decision, dict) else {}
        if isinstance(domains, dict) and domains:
            status = (
                payload.get("system_status")
                or payload.get("status")
                or (final_decision.get("system_status") if isinstance(final_decision, dict) else "")
                or (final_decision.get("decision") if isinstance(final_decision, dict) else "")
            )
            if str(status or "").strip().lower() not in {"", "running"}:
                preferred_terminal = (path, payload)
                break
            if preferred_running is None:
                preferred_running = (path, payload)
    selected = preferred_terminal or preferred_running or fallback
    if selected is not None:
        path, payload = selected
        return {"path": str(path), "payload": payload}
    if fallback is not None:
        path, payload = fallback
        return {"path": str(path), "payload": payload}
    latest = candidates[0]
    return {"path": str(latest), "payload": json.loads(latest.read_text(encoding="utf-8"))}


def run_harness(domains: list[str], request: dict[str, Any]) -> dict[str, Any]:
    _bootstrap()
    from harness.domain_agents.life import LifeAgentAdapter
    from harness.domain_agents.orchestrator import HarnessOrchestrator
    from harness.domain_agents.registry import DomainAgentRegistry
    from harness.domain_agents.sport import SportAgentAdapter
    from harness.domain_agents.study import StudyAgentAdapter

    registry = DomainAgentRegistry()
    registry.register(StudyAgentAdapter(ROOT / "study"))
    registry.register(LifeAgentAdapter(ROOT))
    registry.register(SportAgentAdapter(ROOT))
    orchestrator = HarnessOrchestrator(registry)
    return orchestrator.run_single_domain(domains[0], request) if len(domains) == 1 else orchestrator.run_multi_domain(domains, request)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    import csv

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    except Exception:
        return []


def build_frontend_bundle() -> dict[str, Any]:
    from harness.frontend_fusion import build_frontend_bundle as build_recomputed_bundle

    return build_recomputed_bundle()


if FastAPI is None:
    app = MissingFastAPIApp()
else:
    app = FastAPI(title="Harness API", description="Unified multi-domain harness API.", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "harness",
            "available_domains": ["study", "life", "sport"],
            "runs_dir": str(RUNS_DIR),
        }

    @app.post("/harness/run")
    def harness_run(payload: dict[str, Any]) -> dict[str, Any]:
        domains = payload.get("domains") or ["study", "life", "sport"]
        request = payload.get("request")
        if not isinstance(domains, list) or not domains:
            raise HTTPException(status_code=400, detail="domains must be a non-empty list")
        if not isinstance(request, dict):
            raise HTTPException(status_code=400, detail="request must be a JSON object")

        result = run_harness([str(d) for d in domains], request)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        run_type = "single_domain" if len(domains) == 1 else "multi_domain"
        domain_part = "_".join([str(d) for d in domains])
        stable_file = os.getenv("HARNESS_STABLE_RUN_FILE", "").lower() in {"1", "true", "yes"}
        run_id = f"{run_type}_{domain_part}" if stable_file else f"{run_type}_{domain_part}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        output_path = RUNS_DIR / f"{run_id}.json"
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            from harness.frontend_fusion import build_frontend_bundle as build_recomputed_bundle

            build_recomputed_bundle(output_path)
        except Exception:
            pass
        return {"run_record_path": str(output_path), "result": result}

    @app.get("/harness/result/latest")
    def harness_result_latest() -> dict[str, Any]:
        latest = _read_latest_run()
        return {"run_record_path": latest["path"], "result": latest["payload"]}

    @app.get("/harness/frontend_bundle/latest")
    def harness_frontend_bundle_latest() -> dict[str, Any]:
        return build_frontend_bundle()
