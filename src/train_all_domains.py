from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"


def run_script(script_name: str, args: list[str]) -> None:
    script = SRC_DIR / script_name
    cmd = [sys.executable, str(script)] + args
    completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Script failed: {script_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train learning, life and sport domain models.")
    parser.add_argument("--skip-learning", action="store_true", help="Skip existing learning model training.")
    parser.add_argument("--skip-life-sport", action="store_true", help="Skip life/sport domain training.")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="Device mode for learning model.")
    parser.add_argument("--life-active-threshold", type=int, default=3, help="Active threshold for life model label.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_learning:
        run_script("train_multitask.py", ["--device", args.device])

    if not args.skip_life_sport:
        run_script("train_domain_models.py", ["--life-active-threshold", str(args.life_active_threshold)])

    print("All requested domain trainings completed.")


if __name__ == "__main__":
    main()
