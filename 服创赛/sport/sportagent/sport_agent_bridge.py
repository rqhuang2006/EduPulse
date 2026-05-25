from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STUDYMODEL_WORKSPACE = ROOT / "studymodel"
SPORT_ROOT = STUDYMODEL_WORKSPACE / "sport"
SPORT_AGENT_SCRIPT = SPORT_ROOT / "src" / "sport_agent.py"
SPORT_INPUT_DIR = SPORT_ROOT / "input"
GENERATED_REQUEST_DIR = SPORT_INPUT_DIR / "generated"
DEFAULT_OUT_DIR = ROOT / "outputs" / "sport"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SportAgent for training/inference and export compatible outputs.")
    parser.add_argument("--mode", choices=["train", "infer", "both"], default="infer", help="Execution mode for SportAgent.")
    parser.add_argument("--term-id", default="all", help="Term selector written to SportAgent request.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Project-compatible output directory.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun in infer mode even if compatible prediction file already exists.",
    )
    return parser.parse_args()


def _load_request_template(run_mode: str) -> dict[str, Any]:
    template_map = {
        "train": SPORT_INPUT_DIR / "sport_agent_request.train.json",
        "infer": SPORT_INPUT_DIR / "sport_agent_request.infer.json",
    }
    template_path = template_map[run_mode]
    if not template_path.exists():
        raise FileNotFoundError(f"Missing SportAgent request template: {template_path}")
    return json.loads(template_path.read_text(encoding="utf-8"))


def build_request(run_mode: str, term_id: str = "all", out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, Any]:
    request = _load_request_template(run_mode)
    request["run_mode"] = run_mode
    request["term_id"] = term_id
    request["request_id"] = f"project_sport_{run_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    request["llm_enable"] = False
    request["fallback_to_mock"] = True
    request["llm_required"] = False

    out_dir = out_dir.resolve()
    request["input_paths"] = {
        "feature_dataset": str(out_dir / "feature_dataset.csv"),
        "prediction_output": str(out_dir / "predictions_full.csv"),
        "prediction_test_output": str(out_dir / "predictions_test.csv"),
        "quality_report": str(out_dir / "docs" / "sport_quality_report.json"),
        "validation_report": str(out_dir / "docs" / "sport_validation_report.json"),
        "model_regression": str(out_dir / "regression" / "best_sport_regression_model.joblib"),
        "model_classification": str(out_dir / "classification" / "best_sport_classification_model.joblib"),
        "model_config": str(out_dir / "models" / "sport_model_config.json"),
        "metrics": str(out_dir / "metrics.json"),
    }
    return request


def _write_request(request: dict[str, Any], run_mode: str) -> Path:
    GENERATED_REQUEST_DIR.mkdir(parents=True, exist_ok=True)
    request_path = GENERATED_REQUEST_DIR / f"sport_agent_request.generated.{run_mode}.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    return request_path


def _pythonpath_env() -> dict[str, str]:
    env = os.environ.copy()
    entries = [
        str(STUDYMODEL_WORKSPACE / ".deps3"),
        str(ROOT),
        str(SPORT_ROOT),
    ]
    existing = env.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run_agent_request(request_path: Path) -> None:
    cmd = [sys.executable, str(SPORT_AGENT_SCRIPT), "--request", str(request_path)]
    completed = subprocess.run(
        cmd,
        cwd=str(SPORT_ROOT),
        env=_pythonpath_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        message = (
            "SportAgent execution failed.\n"
            f"request={request_path}\n"
            f"stdout:\n{completed.stdout[-2000:]}\n"
            f"stderr:\n{completed.stderr[-2000:]}"
        )
        raise RuntimeError(message)


def run_train(term_id: str = "all", out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    request = build_request("train", term_id=term_id, out_dir=out_dir)
    request_path = _write_request(request, "train")
    _run_agent_request(request_path)
    return Path(request["input_paths"]["prediction_output"])


def run_infer(term_id: str = "all", out_dir: Path = DEFAULT_OUT_DIR, force: bool = False) -> Path:
    compatible_path = out_dir / "predictions_full.csv"
    if compatible_path.exists() and not force:
        return compatible_path
    request = build_request("infer", term_id=term_id, out_dir=out_dir)
    request_path = _write_request(request, "infer")
    _run_agent_request(request_path)
    return Path(request["input_paths"]["prediction_output"])


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        result = run_train(term_id=args.term_id, out_dir=args.out_dir)
        print(f"SportAgent train completed. Compatible predictions: {result}")
        return
    if args.mode == "infer":
        result = run_infer(term_id=args.term_id, out_dir=args.out_dir, force=args.force)
        print(f"SportAgent infer completed. Compatible predictions: {result}")
        return

    run_train(term_id=args.term_id, out_dir=args.out_dir)
    result = run_infer(term_id=args.term_id, out_dir=args.out_dir, force=True)
    print(f"SportAgent train+infer completed. Compatible predictions: {result}")


if __name__ == "__main__":
    main()
