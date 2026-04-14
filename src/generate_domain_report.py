from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_domain_comparison() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    learning = _read_json(OUT_DIR / "metrics.json")
    if learning:
        reg = learning.get("regression", {})
        cls = learning.get("classification", {})
        rows.append(
            {
                "domain": "learning",
                "task": "regression(avg_score)",
                "best_model": reg.get("best_model"),
                "primary_metric": "r2",
                "metric_value": reg.get("r2"),
            }
        )
        rows.append(
            {
                "domain": "learning",
                "task": "classification(fail_flag)",
                "best_model": cls.get("best_model"),
                "primary_metric": "f1",
                "metric_value": cls.get("f1"),
            }
        )

    life = _read_json(OUT_DIR / "life" / "metrics.json")
    if life:
        rows.append(
            {
                "domain": "life",
                "task": "classification(club_active_flag)",
                "best_model": life.get("best_model"),
                "primary_metric": "f1",
                "metric_value": life.get("f1"),
            }
        )

    sport = _read_json(OUT_DIR / "sport" / "metrics.json")
    if sport:
        sport_reg = sport.get("regression", {})
        sport_cls = sport.get("classification", {})
        rows.append(
            {
                "domain": "sport",
                "task": "regression(zf_score)",
                "best_model": sport_reg.get("best_model"),
                "primary_metric": "r2",
                "metric_value": sport_reg.get("r2"),
            }
        )
        rows.append(
            {
                "domain": "sport",
                "task": "classification(zf_grade)",
                "best_model": sport_cls.get("best_model"),
                "primary_metric": "macro_f1",
                "metric_value": sport_cls.get("macro_f1"),
            }
        )

    return pd.DataFrame(rows)


def write_summary(df: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# 三域模型汇总报告")
    lines.append("")

    if df.empty:
        lines.append("未发现可汇总的模型指标，请先运行训练脚本。")
        (OUT_DIR / "domain_report_summary.md").write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("## 1. 任务与最优模型")
    for _, row in df.iterrows():
        lines.append(
            f"- 域={row['domain']} | 任务={row['task']} | 最优模型={row['best_model']} | "
            f"{row['primary_metric']}={row['metric_value']}"
        )

    lines.append("")
    lines.append("## 2. 产物目录")
    lines.append("- 学习域输出：outputs/")
    lines.append("- 生活域输出：outputs/life/")
    lines.append("- 运动域输出：outputs/sport/")
    lines.append("")
    lines.append("## 3. 说明")
    lines.append("- 运动域当前同时提供分数回归与等级分类两个专用模型。")
    lines.append("- 生活域模型标签由社团活动次数阈值构造，可通过训练参数调整。")

    (OUT_DIR / "domain_report_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = build_domain_comparison()
    df.to_csv(OUT_DIR / "domain_model_comparison.csv", index=False)
    write_summary(df)
    print("Done. Generated outputs/domain_model_comparison.csv and outputs/domain_report_summary.md")


if __name__ == "__main__":
    main()
