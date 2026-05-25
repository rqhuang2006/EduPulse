from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
VENDOR_DIR = WORKSPACE_ROOT / ".deps3"

PIPELINE = [
    "src/00_profile_raw.py",
    "src/01_build_field_registry.py",
    "src/02_build_ods.py",
    "src/03_build_student_term_base.py",
    "src/10_build_l1_grade.py",
    "src/11_build_l2_course_load.py",
    "src/12_build_l3_attendance.py",
    "src/13_build_l4_class_task.py",
    "src/14_build_l5_assignment.py",
    "src/15_build_l6_exam_quiz.py",
    "src/16_build_l7_online_activity.py",
    "src/20_build_label.py",
    "src/21_build_study_train_infer.py",
    "src/22_build_feature_dictionary.py",
    "src/30_train_study_model.py",
    "src/31_generate_prediction_output.py",
    "src/32_generate_explanation_output.py",
    "src/33_generate_quality_report.py",
    "src/34_package_study_bundle.py",
    "src/35_validate_study_bundle.py",
]

STAGES = {
    "profile": PIPELINE[:2],
    "ods": PIPELINE[2:3],
    "features": PIPELINE[3:13],
    "model": PIPELINE[13:16],
    "report": PIPELINE[16:],
    "all": PIPELINE,
}


def run_script(script: str) -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    pythonpath_entries = [str(VENDOR_DIR), str(WORKSPACE_ROOT), str(ROOT)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    path = ROOT / script
    print(f"\n===== RUN {script} =====", flush=True)
    completed = subprocess.run([sys.executable, str(path)], cwd=ROOT, env=env)
    if completed.returncode != 0:
        raise SystemExit(f"FAILED: {script}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the study domain data pipeline.")
    parser.add_argument("--stage", choices=STAGES.keys(), default="all")
    args = parser.parse_args()

    for script in STAGES[args.stage]:
        run_script(script)

    print("\nStudy pipeline completed.")


if __name__ == "__main__":
    main()
