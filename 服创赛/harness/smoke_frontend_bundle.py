from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "run_harness_server.py"
PORT = 8771
BASE_URL = f"http://127.0.0.1:{PORT}"
EXPECTED_STATUS = {"fresh", "fallback", "rule_derived", "missing"}


def http_json(method: str, path: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_until_ready() -> None:
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            health = http_json("GET", "/health", timeout=3)
            if health.get("status") == "ok":
                return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("harness server did not become ready")


def assert_bundle(bundle: dict[str, Any], expected_run_file: str) -> None:
    required = [
        "built_from_run_id",
        "built_from_run_file",
        "built_from_domains",
        "generated_at",
        "student_artifacts_fresh",
        "artifact_consistency_ok",
        "artifact_freshness",
    ]
    missing = [key for key in required if key not in bundle]
    if missing:
        raise AssertionError(f"bundle missing fields: {missing}")
    if Path(str(bundle["built_from_run_file"])) != Path(expected_run_file):
        raise AssertionError("bundle built_from_run_file does not match POST /harness/run response")
    if bundle["built_from_run_id"] != Path(expected_run_file).stem:
        raise AssertionError("bundle built_from_run_id does not match run file stem")
    domains = set(str(domain) for domain in bundle.get("built_from_domains", []))
    if not {"study", "life", "sport"}.issubset(domains):
        raise AssertionError(f"bundle domains are incomplete: {sorted(domains)}")
    if not bundle.get("artifact_consistency_ok"):
        raise AssertionError(f"bundle consistency failed: {bundle.get('artifact_consistency_errors')}")
    freshness = bundle.get("artifact_freshness", {})
    for key in ["master_table", "interventions", "reports", "group_profile", "pattern_summary", "life_shap", "sport_shap", "map_scores"]:
        value = freshness.get(key)
        if value not in EXPECTED_STATUS:
            raise AssertionError(f"invalid freshness status for {key}: {value!r}")
    if "single_domain" in str(bundle.get("built_from_run_id", "")):
        raise AssertionError("frontend_bundle/latest selected a single-domain run")


def main() -> int:
    process = subprocess.Popen(
        [sys.executable, "-B", str(SERVER), "--port", str(PORT)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_until_ready()
        run_payload = {
            "domains": ["study", "life", "sport"],
            "request": {
                "request_id": "frontend_bundle_smoke",
                "domain": "study",
                "run_mode": "review",
                "execution_engine": "harness_v1",
                "input_paths": {},
            },
        }
        run_response = http_json("POST", "/harness/run", run_payload, timeout=360)
        run_file = str(run_response.get("run_record_path", ""))
        if not run_file:
            raise AssertionError("POST /harness/run did not return run_record_path")
        bundle = http_json("GET", "/harness/frontend_bundle/latest", timeout=60)
        assert_bundle(bundle, run_file)
        print(
            json.dumps(
                {
                    "ok": True,
                    "run_id": bundle["built_from_run_id"],
                    "domains": bundle["built_from_domains"],
                    "students": bundle.get("counts", {}).get("students"),
                    "artifact_consistency_ok": bundle["artifact_consistency_ok"],
                    "artifact_freshness": bundle["artifact_freshness"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
