from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RUNS_DIR = Path(os.getenv("HARNESS_RUNS_DIR", str(ROOT / "data" / "harness" / "runs")))
FRONTEND_ROOT = Path(os.getenv("FRONTEND_ROOT", str(ROOT.parent / "1")))
FRONTEND_A14_DIR = Path(os.getenv("FRONTEND_A14_DIR", str(FRONTEND_ROOT / "outputs" / "a14")))
BUNDLE_DIR = Path(os.getenv("FRONTEND_BUNDLE_DIR", str(ROOT / "data" / "harness" / "frontend_bundle")))
BUNDLE_PATH = Path(os.getenv("FRONTEND_BUNDLE_PATH", str(BUNDLE_DIR / "latest_frontend_bundle.json")))
DEFAULT_DOMAINS = ["study", "life", "sport"]
DEFAULT_REQUEST = {
    "request_id": "frontend_run_001",
    "domain": "study",
    "run_mode": "review",
    "execution_engine": "harness_v1",
    "input_paths": {},
}


def bootstrap() -> None:
    for path in [ROOT / ".deps3", ROOT]:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def has_domain_summary(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    final_decision = payload.get("final_decision", {})
    domains = final_decision.get("domains", {}) if isinstance(final_decision, dict) else {}
    return isinstance(domains, dict) and bool(domains)


def is_terminal_result(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    final_decision = payload.get("final_decision", {})
    status = (
        payload.get("system_status")
        or payload.get("status")
        or (final_decision.get("system_status") if isinstance(final_decision, dict) else "")
        or (final_decision.get("decision") if isinstance(final_decision, dict) else "")
    )
    return str(status or "").strip().lower() not in {"", "running"}


def read_latest_run() -> dict[str, Any]:
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
        if has_domain_summary(payload):
            if is_terminal_result(payload):
                preferred_terminal = (path, payload)
                break
            if preferred_running is None:
                preferred_running = (path, payload)

    selected = preferred_terminal or preferred_running or fallback
    if selected is not None:
        path, payload = selected
        return {"run_record_path": str(path), "result": payload}
    if fallback is not None:
        path, payload = fallback
        return {"run_record_path": str(path), "result": payload}
    path = candidates[0]
    return {"run_record_path": str(path), "result": json.loads(path.read_text(encoding="utf-8"))}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    except Exception:
        return []


def build_frontend_bundle() -> dict[str, Any]:
    bootstrap()
    from harness.frontend_fusion import build_frontend_bundle as build_recomputed_bundle

    return build_recomputed_bundle()


def run_harness(domains: list[str], request: dict[str, Any]) -> dict[str, Any]:
    bootstrap()
    from harness.harness_api import run_harness as run_harness_impl

    return run_harness_impl(domains, request)


def build_run_output_path(domains: list[str]) -> Path:
    run_type = "single_domain" if len(domains) == 1 else "multi_domain"
    domain_part = "_".join(domains)
    stable_file = os.getenv("HARNESS_STABLE_RUN_FILE", "").lower() in {"1", "true", "yes"}
    run_id = f"{run_type}_{domain_part}" if stable_file else f"{run_type}_{domain_part}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    return RUNS_DIR / f"{run_id}.json"


def write_run_result(output_path: Path, result: dict[str, Any]) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def build_pending_result(domains: list[str], request: dict[str, Any]) -> dict[str, Any]:
    run_type = "single_domain" if len(domains) == 1 else "multi_domain"
    domain_results = {
        domain_name: {
            "domain_name": domain_name,
            "status": "running",
            "message": "Harness run accepted and executing in background.",
            "agent_trace": {
                "invoked_via_harness": True,
                "adapter_name": "HarnessOrchestrator",
                "execution_path": "http_handler -> background_thread -> adapter.run_domain_pipeline",
                "domain_pipeline_called": False,
                "request_domain": domain_name,
            },
        }
        for domain_name in domains
    }
    final_domains = {
        domain_name: {
            "status": "running",
            "decision": "",
            "decision_semantics": "run_in_progress",
            "comparable": False,
            "mainline_task_type": "",
            "mainline_validity": False,
            "blocking_reason": "run_in_progress",
            "trusted_mainline": {},
            "mainline_frozen": False,
            "next_optimization_target": "",
        }
        for domain_name in domains
    }
    return {
        "run_type": run_type,
        "domain_results": domain_results,
        "fusion_inputs": [],
        "final_decision": {
            "decision": "running",
            "summary": "Harness run is executing in background.",
            "system_status": "running",
            "decision_vocabulary": {},
            "stub_domains": [],
            "ready_domains": [],
            "domains": final_domains,
            "domain_decisions": {domain_name: "" for domain_name in domains},
        },
        "warnings": [],
        "validations": [],
        "status": "running",
        "system_status": "running",
        "domain_timings": {},
        "domain_start_order": [],
        "domain_end_order": [],
        "adapter_names": {},
        "domain_statuses": {domain_name: "running" for domain_name in domains},
        "slowest_domain": None,
        "request_id": str(request.get("request_id", "")),
        "accepted_at": datetime.now().isoformat(timespec="seconds"),
        "agent_trace": {
            "invoked_via_harness": True,
            "adapter_name": "HarnessHTTPServer",
            "execution_path": "http_handler -> background_thread",
            "domain_pipeline_called": False,
            "request_domain": request.get("domain"),
        },
    }


def build_failed_result(domains: list[str], request: dict[str, Any], exc: Exception) -> dict[str, Any]:
    domain_results = {
        domain_name: {
            "domain_name": domain_name,
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "agent_trace": {
                "invoked_via_harness": True,
                "adapter_name": "HarnessHTTPServer",
                "execution_path": "http_handler -> background_thread",
                "domain_pipeline_called": False,
                "request_domain": domain_name,
            },
        }
        for domain_name in domains
    }
    final_domains = {
        domain_name: {
            "status": "failed",
            "decision": "",
            "decision_semantics": "needs_review_or_followup",
            "comparable": False,
            "mainline_task_type": "",
            "mainline_validity": False,
            "blocking_reason": str(exc),
            "trusted_mainline": {},
            "mainline_frozen": False,
            "next_optimization_target": "",
        }
        for domain_name in domains
    }
    return {
        "run_type": "single_domain" if len(domains) == 1 else "multi_domain",
        "domain_results": domain_results,
        "fusion_inputs": [],
        "final_decision": {
            "decision": "failed",
            "summary": str(exc),
            "system_status": "failed",
            "decision_vocabulary": {},
            "stub_domains": [],
            "ready_domains": [],
            "domains": final_domains,
            "domain_decisions": {domain_name: "" for domain_name in domains},
        },
        "warnings": [f"{type(exc).__name__}: {exc}"],
        "validations": [],
        "status": "failed",
        "system_status": "failed",
        "domain_timings": {},
        "domain_start_order": [],
        "domain_end_order": [],
        "adapter_names": {},
        "domain_statuses": {domain_name: "failed" for domain_name in domains},
        "slowest_domain": None,
        "request_id": str(request.get("request_id", "")),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "agent_trace": {
            "invoked_via_harness": True,
            "adapter_name": "HarnessHTTPServer",
            "execution_path": "http_handler -> background_thread",
            "domain_pipeline_called": False,
            "request_domain": request.get("domain"),
        },
    }


def execute_harness_run(domains: list[str], request: dict[str, Any], output_path: Path) -> None:
    try:
        result = run_harness(domains, request)
    except Exception as exc:
        result = build_failed_result(domains, request, exc)
    write_run_result(output_path, result)
    try:
        from harness.frontend_fusion import build_frontend_bundle as build_recomputed_bundle

        build_recomputed_bundle(output_path)
    except Exception:
        pass


class HarnessRequestHandler(BaseHTTPRequestHandler):
    server_version = "HarnessStdlibHTTP/1.0"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "harness",
                    "available_domains": DEFAULT_DOMAINS,
                    "runs_dir": str(RUNS_DIR),
                    "server": "stdlib",
                },
            )
            return

        if self.path == "/harness/result/latest":
            try:
                self._send_json(200, read_latest_run())
            except Exception as exc:
                self._send_json(404, {"status": "failed", "message": str(exc)})
            return

        if self.path == "/harness/frontend_bundle/latest":
            try:
                self._send_json(200, build_frontend_bundle())
            except Exception as exc:
                self._send_json(500, {"status": "failed", "message": str(exc), "exception_type": type(exc).__name__})
            return

        self._send_json(404, {"status": "failed", "message": f"unknown path: {self.path}"})

    def do_POST(self) -> None:
        if self.path != "/harness/run":
            self._send_json(404, {"status": "failed", "message": f"unknown path: {self.path}"})
            return

        try:
            payload = self._read_json_body()
            domains = payload.get("domains") or DEFAULT_DOMAINS
            request = payload.get("request") or DEFAULT_REQUEST
            if not isinstance(domains, list) or not domains:
                raise ValueError("domains must be a non-empty list")
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            domain_names = [str(domain) for domain in domains]
            output_path = build_run_output_path(domain_names)
            pending = build_pending_result(domain_names, request)
            write_run_result(output_path, pending)
            worker = threading.Thread(
                target=execute_harness_run,
                args=(domain_names, request, output_path),
                daemon=True,
            )
            worker.start()
            self._send_json(202, {"run_record_path": str(output_path), "result": pending, "accepted": True})
        except Exception as exc:
            self._send_json(500, {"status": "failed", "message": str(exc), "exception_type": type(exc).__name__})

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frontend-facing harness HTTP service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bootstrap()
    server = ThreadingHTTPServer((args.host, args.port), HarnessRequestHandler)
    print(f"Harness service listening on http://{args.host}:{args.port}", flush=True)
    print("Endpoints: GET /health, GET /harness/result/latest, GET /harness/frontend_bundle/latest, POST /harness/run", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
