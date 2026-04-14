from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from llm_client import LLMClient


ROOT = Path(__file__).resolve().parents[1]
A14_DIR = ROOT / "outputs" / "a14"
DOMAIN_LABELS = {"study": "学习", "life": "生活", "sport": "运动", "unknown": "未知"}
MAP_LABELS = {"M": "动机", "A": "能力", "P": "提示"}


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class RiskAgent:
    def analyze(self, row: Dict[str, str]) -> Dict[str, object]:
        return {
            "risk_level": row.get("risk_level", "未知"),
            "total_risk": row.get("total_risk", ""),
            "dominant_dimension": DOMAIN_LABELS.get(row.get("dominant_dimension", "unknown"), "未知"),
            "dimension_risks": {
                "life_risk": row.get("life_risk", ""),
                "study_risk": row.get("study_risk", ""),
                "sport_risk": row.get("sport_risk", ""),
            },
        }


class BehaviorAgent:
    def analyze(self, row: Dict[str, str]) -> Dict[str, object]:
        return {
            "pattern_label": row.get("pattern_label", ""),
            "pattern_reason": row.get("pattern_reason", ""),
            "top_behaviors": {
                "life": [row.get("life_shap_top1", ""), row.get("life_shap_top2", ""), row.get("life_shap_top3", "")],
                "study": [row.get("study_shap_top1", ""), row.get("study_shap_top2", ""), row.get("study_shap_top3", "")],
                "sport": [row.get("sport_shap_top1", ""), row.get("sport_shap_top2", ""), row.get("sport_shap_top3", "")],
            },
        }


class MechanismAgent:
    def analyze(self, row: Dict[str, str]) -> Dict[str, object]:
        dominant_map = row.get("dominant_MAP", "")
        return {
            "dominant_map": dominant_map,
            "dominant_map_label": MAP_LABELS.get(dominant_map, dominant_map),
            "map_scores": {
                "M_score": row.get("M_score", ""),
                "A_score": row.get("A_score", ""),
                "P_score": row.get("P_score", ""),
            },
        }


class InterventionAgent:
    def analyze(self, row: Dict[str, str]) -> Dict[str, object]:
        return {
            "intervention_type": row.get("intervention_type", ""),
            "intervention_text": row.get("intervention_text", ""),
            "priority": row.get("priority", ""),
        }


class RuleReportAgent:
    def compose(
        self,
        student_id: str,
        row: Dict[str, str],
        risk: Dict[str, object],
        behavior: Dict[str, object],
        mechanism: Dict[str, object],
        intervention: Dict[str, object],
    ) -> Dict[str, object]:
        return {
            "student_id": student_id,
            "summary": row.get("profile_text", ""),
            "risk": risk,
            "behavior": behavior,
            "mechanism": mechanism,
            "intervention": intervention,
        }


class LLMReportAgent:
    def __init__(self) -> None:
        self.client = LLMClient()

    def compose(
        self,
        student_id: str,
        row: Dict[str, str],
        risk: Dict[str, object],
        behavior: Dict[str, object],
        mechanism: Dict[str, object],
        intervention: Dict[str, object],
    ) -> Dict[str, object]:
        system_prompt = (
            "你是高校学生行为分析系统中的报告代理。"
            "请基于输入信息生成严格 JSON，字段必须包含: "
            "student_id, summary, risk, behavior, mechanism, intervention, narrative."
        )
        user_prompt = json.dumps(
            {
                "student_id": student_id,
                "profile_text": row.get("profile_text", ""),
                "risk": risk,
                "behavior": behavior,
                "mechanism": mechanism,
                "intervention": intervention,
            },
            ensure_ascii=False,
        )
        return self.client.complete_json(system_prompt, user_prompt)


class A14MultiAgentSystem:
    def __init__(self, mode: str = "rule") -> None:
        self.risk_agent = RiskAgent()
        self.behavior_agent = BehaviorAgent()
        self.mechanism_agent = MechanismAgent()
        self.intervention_agent = InterventionAgent()
        self.report_agent = LLMReportAgent() if mode == "llm" else RuleReportAgent()
        self.mode = mode

    def run(self, rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
        reports: List[Dict[str, object]] = []
        total = len(rows)
        for index, row in enumerate(rows, start=1):
            student_id = row.get("student_id", "")
            if total <= 20 or index == 1 or index == total or index % 10 == 0:
                print(f"[{self.mode}] processing student {index}/{total}: {student_id}")
            risk = self.risk_agent.analyze(row)
            behavior = self.behavior_agent.analyze(row)
            mechanism = self.mechanism_agent.analyze(row)
            intervention = self.intervention_agent.analyze(row)
            reports.append(self.report_agent.compose(student_id, row, risk, behavior, mechanism, intervention))
        return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble A14 student reports with a multi-agent pipeline.")
    parser.add_argument("--input", type=Path, default=A14_DIR / "fusion_student_master_table.csv", help="Input fusion master table.")
    parser.add_argument("--out-file", type=Path, default=A14_DIR / "student_full_report_multi_agent.json", help="Output JSON report path.")
    parser.add_argument("--student-id", type=str, default="", help="Optional student_id filter.")
    parser.add_argument("--mode", choices=["rule", "llm"], default="rule", help="Report agent mode.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of students to process after filtering.")
    parser.add_argument("--all", action="store_true", help="Allow processing all students in llm mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv_dicts(args.input)
    if args.student_id:
        rows = [row for row in rows if row.get("student_id") == args.student_id]
    total_available = len(rows)
    if args.limit > 0:
        rows = rows[: args.limit]
    elif args.mode == "llm" and not args.student_id and not args.all:
        rows = rows[:10]
        print(
            f"LLM mode quota protection enabled: processing 10/{total_available} students by default. "
            "Use --limit N for a custom batch, --student-id for a single student, or --all for full processing."
        )
    elif args.mode == "llm" and args.all:
        print(f"LLM full-run explicitly enabled: processing all {total_available} students.")
    print(f"Loaded {len(rows)} students for multi-agent mode={args.mode}.")
    system = A14MultiAgentSystem(mode=args.mode)
    reports = system.run(rows)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Multi-agent reports generated in: {args.out_file}")


if __name__ == "__main__":
    main()
