from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run_step(script: str, *args: str) -> None:
    cmd = [sys.executable, str(SRC / script), *args]
    completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Step failed: {script}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full A14 deliverable pipeline.")
    parser.add_argument("--mode", choices=["rule", "llm"], default="rule", help="Multi-agent report mode.")
    parser.add_argument("--skip-check", action="store_true", help="Skip API connectivity check in llm mode.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of students for multi-agent report generation.")
    parser.add_argument("--student-id", type=str, default="", help="Optional single-student run for llm mode.")
    parser.add_argument("--all", action="store_true", help="Allow processing all students in llm mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_step("generate_a14_deliverables.py")
    if args.mode == "llm" and not args.skip_check:
        run_step("check_llm_connection.py")
    agent_args = ["--mode", args.mode]
    if args.student_id:
        agent_args += ["--student-id", args.student_id]
        print(f"LLM pipeline targeting single student: {args.student_id}")
    elif args.limit > 0:
        agent_args += ["--limit", str(args.limit)]
    elif args.mode == "llm":
        if args.all:
            agent_args += ["--all"]
            print("LLM mode full-run requested explicitly.")
        else:
            agent_args += ["--limit", "10"]
            print("LLM mode defaulting to 10 students to save quota. Use --limit, --student-id, or --all to override.")
    run_step("a14_multi_agent.py", *agent_args)
    run_step("generate_a14_demo_html.py")
    print(f"A14 pipeline completed in mode={args.mode}. Output dir: {ROOT / 'outputs' / 'a14'}")


if __name__ == "__main__":
    main()
