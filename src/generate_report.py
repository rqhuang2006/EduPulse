from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "report_figures"
FIG_DIR.mkdir(exist_ok=True)

FAIRNESS_GROUPS = ["XB", "MZMC", "ZZMMMC", "JG", "ZYM"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _save_plot(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=160, bbox_inches="tight")
    plt.close()


def plot_model_comparison() -> Dict[str, str]:
    out = {}
    reg = _read_csv(OUT_DIR / "model_comparison_regression.csv")
    cls = _read_csv(OUT_DIR / "model_comparison_classification.csv")

    if not reg.empty:
        plt.figure(figsize=(9, 4.5))
        sns.barplot(data=reg, x="model", y="val_r2", hue="model", legend=False, errorbar=None, palette="Blues")
        plt.title("Regression Model Comparison (Validation R2)")
        plt.xlabel("Model")
        plt.ylabel("Validation R2")
        _save_plot("model_compare_regression.png")
        out["regression"] = "report_figures/model_compare_regression.png"

    if not cls.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        sns.barplot(
            data=cls,
            x="model",
            y="val_f1",
            hue="model",
            legend=False,
            errorbar=None,
            palette="Greens",
            ax=axes[0],
        )
        axes[0].set_title("Classification Val F1")
        axes[0].set_xlabel("Model")
        axes[0].set_ylabel("Validation F1")

        sns.barplot(
            data=cls,
            x="model",
            y="best_val_threshold",
            hue="model",
            legend=False,
            errorbar=None,
            palette="Oranges",
            ax=axes[1],
        )
        axes[1].set_title("Optimized Threshold")
        axes[1].set_xlabel("Model")
        axes[1].set_ylabel("Threshold")

        _save_plot("model_compare_classification.png")
        out["classification"] = "report_figures/model_compare_classification.png"

    return out


def plot_feature_importance() -> Dict[str, str]:
    out = {}
    for task, fname, color in [
        ("regression", "feature_importance_regression.csv", "#1f77b4"),
        ("classification", "feature_importance_classification.csv", "#2ca02c"),
    ]:
        df = _read_csv(OUT_DIR / fname)
        if df.empty:
            continue
        top = df.head(20).sort_values("importance", ascending=True)
        plt.figure(figsize=(10, 7))
        plt.barh(top["feature"], top["importance"], color=color)
        plt.title(f"Top 20 Feature Importance ({task})")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        out_name = f"feature_importance_{task}.png"
        _save_plot(out_name)
        out[task] = f"report_figures/{out_name}"
    return out


def plot_cluster_profile() -> str:
    cluster = _read_csv(OUT_DIR / "cluster_summary.csv")
    if cluster.empty:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.barplot(data=cluster, x="cluster", y="mean_score", hue="cluster", legend=False, palette="mako", ax=axes[0])
    axes[0].set_title("Cluster Mean Score")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Mean Score")

    sns.barplot(
        data=cluster,
        x="cluster",
        y="fail_rate",
        hue="cluster",
        legend=False,
        palette="rocket",
        ax=axes[1],
    )
    axes[1].set_title("Cluster Fail Rate")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Fail Rate")

    _save_plot("cluster_profile.png")
    return "report_figures/cluster_profile.png"


def fairness_evaluation() -> pd.DataFrame:
    pred = _read_csv(OUT_DIR / "predictions_test.csv")
    feat = _read_csv(OUT_DIR / "feature_dataset.csv")
    if pred.empty or feat.empty:
        return pd.DataFrame()

    data = pred.merge(feat[["sid", "term_id"] + [c for c in FAIRNESS_GROUPS if c in feat.columns]], on=["sid", "term_id"], how="left")

    rows: List[Dict[str, object]] = []
    for g in FAIRNESS_GROUPS:
        if g not in data.columns:
            continue
        temp = data[[g, "fail_flag", "pred_fail_flag"]].copy()
        temp[g] = temp[g].fillna("UNKNOWN").astype(str)
        for value, sub in temp.groupby(g):
            y_true = sub["fail_flag"].astype(int)
            y_pred = sub["pred_fail_flag"].astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0) if len(sub) > 0 else np.nan
            rows.append(
                {
                    "group_field": g,
                    "group_value": value,
                    "sample_count": int(len(sub)),
                    "true_positive_rate": float(y_true.mean()) if len(sub) > 0 else np.nan,
                    "pred_positive_rate": float(y_pred.mean()) if len(sub) > 0 else np.nan,
                    "group_f1": float(f1),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["group_field", "sample_count"], ascending=[True, False]).reset_index(drop=True)
        out.to_csv(OUT_DIR / "fairness_group_metrics.csv", index=False)

        # A compact disparity view: max-min pred positive rate per field.
        disparity = (
            out.groupby("group_field", as_index=False)
            .agg(max_pred=("pred_positive_rate", "max"), min_pred=("pred_positive_rate", "min"))
        )
        disparity["pred_rate_gap"] = disparity["max_pred"] - disparity["min_pred"]
        disparity.to_csv(OUT_DIR / "fairness_disparity_summary.csv", index=False)

        plt.figure(figsize=(8, 4.5))
        sns.barplot(
            data=disparity,
            x="group_field",
            y="pred_rate_gap",
            hue="group_field",
            legend=False,
            palette="viridis",
        )
        plt.title("Fairness Disparity by Group Field")
        plt.xlabel("Group Field")
        plt.ylabel("Pred Positive Rate Gap")
        _save_plot("fairness_disparity.png")

    return out


def generate_markdown_summary() -> None:
    metrics_path = OUT_DIR / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    reg = metrics.get("regression", {})
    cls = metrics.get("classification", {})

    model_fig = plot_model_comparison()
    imp_fig = plot_feature_importance()
    cluster_fig = plot_cluster_profile()
    fairness_df = fairness_evaluation()

    lines: List[str] = []
    lines.append("# 自动化报告摘要")
    lines.append("")
    lines.append("## 1. 模型效果")
    lines.append(f"- 回归 RMSE: {reg.get('rmse', 'NA')}")
    lines.append(f"- 回归 MAE: {reg.get('mae', 'NA')}")
    lines.append(f"- 回归 R2: {reg.get('r2', 'NA')}")
    lines.append(f"- 回归最优模型: {reg.get('best_model', 'NA')}")
    lines.append(f"- 分类 AUC: {cls.get('auc', 'NA')}")
    lines.append(f"- 分类 F1: {cls.get('f1', 'NA')}")
    lines.append(f"- 分类最优模型: {cls.get('best_model', 'NA')}")
    lines.append(f"- 分类最优阈值: {cls.get('best_threshold', 'NA')}")
    lines.append("")

    lines.append("## 2. 可汇报图表")
    if model_fig:
        lines.append(f"- 模型对比图: {', '.join(model_fig.values())}")
    if imp_fig:
        lines.append(f"- 特征重要性图: {', '.join(imp_fig.values())}")
    if cluster_fig:
        lines.append(f"- 分群画像图: {cluster_fig}")
    if not fairness_df.empty:
        lines.append("- 公平性图: report_figures/fairness_disparity.png")
    lines.append("")

    lines.append("## 3. 结论建议")
    lines.append("- 以验证集指标选模，分类任务通过阈值搜索提升 F1。")
    lines.append("- 时间窗特征可降低跨学期信息混杂，建议持续补齐行为日志时间戳。")
    if not fairness_df.empty:
        gap = (
            fairness_df.groupby("group_field")["pred_positive_rate"].agg(["max", "min"]).assign(gap=lambda d: d["max"] - d["min"])
        )
        max_gap_field = gap["gap"].idxmax()
        max_gap_value = float(gap.loc[max_gap_field, "gap"])
        lines.append(f"- 公平性上，{max_gap_field} 字段的预测阳性率差距最大，约为 {max_gap_value:.4f}，建议做分组再校准。")

    (OUT_DIR / "report_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid")
    generate_markdown_summary()
    print("Done. Report files generated in:", OUT_DIR)


if __name__ == "__main__":
    main()
