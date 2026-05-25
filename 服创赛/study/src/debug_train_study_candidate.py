import json
from pathlib import Path

import pandas as pd

from src.study_candidate_generator import train_study_candidate_pool


def main():
    # 你按自己真实文件路径改
    train_path = Path("data/dm/study_train_sample.parquet")
    valid_path = Path("data/dm/study_valid_sample.parquet")

    if not train_path.exists():
        raise FileNotFoundError(f"train file not found: {train_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"valid file not found: {valid_path}")

    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)

    # 这里按你实际标签列改
    label_col = "label"
    if label_col not in train_df.columns:
        raise KeyError(f"label column not found: {label_col}")

    result = train_study_candidate_pool(
        train_df=train_df,
        valid_df=valid_df,
        label_col=label_col,
        id_col="XH",
        term_col="TERM_ID",
        out_dir="data/dm",
        baseline_recall=0.62,   # 你当前 serving recall 大约 0.62，这里先卡住不让它乱降
        random_state=42,
    )

    print("\n===== STUDY CANDIDATE RESULT =====")
    print(json.dumps({
        "version_id": result.version_id,
        "model_name": result.model_name,
        "threshold": result.threshold,
        "metrics": result.metrics,
        "composite_score": result.composite_score,
        "model_path": result.model_path,
        "feature_columns_path": result.feature_columns_path,
        "comparison_path": result.comparison_path,
        "selection_path": result.selection_path,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()