# 多任务学生学习分析 V2

本项目面向高校学生多源行为数据，构建三大领域模型并输出可直接汇报的评估结果：

- 学习类模型：预测平均成绩（回归）与挂科风险（分类）
- 生活类模型：预测社团活跃度（分类）
- 运动健康类模型：预测体测分数（回归）与体测等级（分类）

## 1. 主要脚本

- `src/train_multitask.py`：学习类模型训练与评估（含聚类和学习力评分）
- `src/train_domain_models.py`：生活类 + 运动健康类专用模型训练
- `src/train_all_domains.py`：一键运行学习/生活/运动三域训练
- `src/generate_report.py`：学习类自动化图表与公平性报告
- `src/generate_domain_report.py`：三域模型对比摘要报告

## 2. 环境安装

```bash
pip install -r requirements.txt
```

核心依赖：`numpy`、`pandas`、`scikit-learn`、`openpyxl`、`joblib`、`matplotlib`、`seaborn`、`lightgbm`、`xgboost`。

## 3. 快速开始

### 3.1 仅训练学习类模型

```bash
python src/train_multitask.py
```

可选设备参数：

```bash
python src/train_multitask.py --device auto
python src/train_multitask.py --device cpu
python src/train_multitask.py --device gpu
```

### 3.2 训练生活类 + 运动健康类模型

```bash
python src/train_domain_models.py
```

可调社团活跃阈值（默认每学期 `>=3` 次活动视为活跃）：

```bash
python src/train_domain_models.py --life-active-threshold 3
```

### 3.3 一键训练全部模型

```bash
python src/train_all_domains.py
python src/train_all_domains.py --skip-learning
python src/train_all_domains.py --skip-life-sport
```

### 3.4 生成报告

```bash
python src/generate_report.py
python src/generate_domain_report.py
```

## 4. 输出目录说明

### 4.1 学习类输出（`outputs/`）

- `feature_dataset.csv`：学习类训练样本
- `predictions_test.csv`：学习类测试集预测
- `metrics.json`：学习类核心指标
- `model_comparison_regression.csv`、`model_comparison_classification.csv`
- `feature_importance_regression.csv`、`feature_importance_classification.csv`
- `cluster_labels.csv`、`cluster_summary.csv`、`learning_score.csv`
- `models/`：学习类最佳模型与元数据

### 4.2 生活类输出（`outputs/life/`）

- `feature_dataset.csv`
- `predictions_test.csv`
- `model_comparison.csv`
- `metrics.json`
- `models/best_life_model.joblib`

### 4.3 运动健康类输出（`outputs/sport/`）

- `feature_dataset.csv`
- `metrics.json`：双任务汇总指标
- `metrics_regression.json`：体测分数回归指标
- `metrics_classification.json`：体测等级分类指标
- `regression/`：回归模型与预测结果
- `classification/`：分类模型与预测结果

### 4.4 报告与跨域汇总输出（`outputs/`）

- `report_summary.md`：学习类自动化结论摘要
- `fairness_group_metrics.csv`、`fairness_disparity_summary.csv`
- `report_figures/`：模型对比、特征重要性、公平性等图表
- `domain_models_summary.json`：生活+运动汇总
- `domain_model_comparison.csv`：学习/生活/运动任务对比
- `domain_report_summary.md`：三域模型摘要

## 5. 最近一次运行结果（当前仓库）

### 5.1 学习类

- 数据规模：`total=20081`，`train=19749`，`val=157`，`test=175`
- 回归（`avg_score`）：`best_model=RandomForest`，`RMSE=17.1108`，`MAE=9.8450`，`R2=-0.3952`
- 分类（`fail_flag`）：`best_model=LightGBM`，`AUC=0.9402`，`F1=0.3243`，`best_threshold=0.29`

### 5.2 生活类

- 分类（`club_active_flag`）：`best_model=RandomForest`，`F1=0.4497`，`AUC=0.5090`，`Accuracy=0.2901`，`samples=2506`

### 5.3 运动健康类

- 回归（`zf_score`）：`best_model=RandomForest`，`RMSE=8.5761`，`MAE=6.3563`，`R2=0.2935`，`samples=9751`
- 分类（`zf_grade`）：`best_model=RandomForest`，`Accuracy=0.5465`，`Macro-F1=0.3520`，`samples=86`

## 6. 指标解读建议

- 回归任务重点看 `R2`、`RMSE`、`MAE` 的综合表现
- 分类任务重点看 `F1`、`AUC`（二分类）与 `Macro-F1`（多分类）
- 公平性分析需结合分组样本量，避免被极小样本组误导

## 7. 已知边界

- 个别日志表时间字段缺失时，会回退到可解析字段，可能降低时间窗精度
- 运动等级分类测试集样本较少，指标波动较大
- LightGBM/XGBoost 未安装时会自动回退到可用模型
