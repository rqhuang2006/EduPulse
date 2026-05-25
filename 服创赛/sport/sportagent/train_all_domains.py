from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
AAA_DIR = ROOT / "AAA"


def run_script(script_name: str, args: list[str]) -> None:
    script = SRC_DIR / script_name
    cmd = [sys.executable, str(script)] + args
    completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Script failed: {script_name}")


def run_aaa(stage: str) -> None:
    script = AAA_DIR / "main.py"
    cmd = [sys.executable, str(script), "--stage", stage]
    completed = subprocess.run(cmd, cwd=str(AAA_DIR), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"AAA study pipeline failed: stage={stage}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train learning, life and sport domain models.")
    parser.add_argument("--skip-learning", action="store_true", help="Skip existing learning model training.")
    parser.add_argument("--skip-life-sport", action="store_true", help="Skip life/sport domain training.")
    parser.add_argument(
        "--study-backend",
        choices=["studymodel", "aaa", "legacy"],
        default="studymodel",
        help="Backend for learning(study) domain. studymodel uses StudyAgent; aaa uses AAA pipeline; legacy uses train_multitask.py.",
    )
    parser.add_argument(
        "--aaa-stage",
        choices=["profile", "ods", "features", "model", "report", "all"],
        default="all",
        help="AAA stage to execute when --study-backend=aaa.",
    )
    parser.add_argument(
        "--study-out-dir",
        default=str(ROOT / "outputs_next" / "study"),
        help="Directory for exported study outputs when using studymodel or AAA backend.",
    )
    parser.add_argument("--study-term-id", default="all", help="Term selector passed to StudyAgent request.")
    parser.add_argument("--sport-term-id", default="all", help="Term selector passed to SportAgent request.")
    parser.add_argument(
        "--sport-backend",
        choices=["agent", "legacy"],
        default="agent",
        help="Backend for sport domain. agent uses SportAgent bridge; legacy uses train_domain_models.py.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="Device mode for learning model.")
    parser.add_argument("--life-active-threshold", type=int, default=3, help="Active threshold for life model label.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_learning:
        if args.study_backend == "studymodel":
            run_script(
                "study_agent_bridge.py",
                ["--mode", "train", "--term-id", args.study_term_id, "--out-dir", args.study_out_dir],
            )
            print(f"Study domain trained via StudyAgent(harness_v1) and exported to: {args.study_out_dir}")
        elif args.study_backend == "aaa":
            run_aaa(args.aaa_stage)
            run_script("export_study_from_aaa.py", ["--out-dir", args.study_out_dir])
            print(f"Study domain trained via AAA and exported to: {args.study_out_dir}")
        else:
            run_script("train_multitask.py", ["--device", args.device])
            print("Study domain trained via legacy pipeline.")

    if not args.skip_life_sport:
        run_script(
            "train_domain_models.py",
            ["--life-active-threshold", str(args.life_active_threshold), "--skip-sport"],
        )
        if args.sport_backend == "agent":
            run_script(
                "sport_agent_bridge.py",
                ["--mode", "train", "--term-id", args.sport_term_id, "--out-dir", str(ROOT / "outputs" / "sport")],
            )
            print("Sport domain trained via SportAgent bridge.")
        else:
            run_script("train_domain_models.py", ["--skip-life"])
            print("Sport domain trained via legacy train_domain_models.py.")

    print("All requested domain trainings completed.")


if __name__ == "__main__":
    main()
