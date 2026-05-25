# 自动化报告摘要

## 1. 模型效果
- 回归 RMSE: 17.110754806415
- 回归 MAE: 9.845016488415164
- 回归 R2: -0.39522499692285984
- 回归最优模型: RandomForest
- 分类 AUC: 0.9401981599433828
- 分类 F1: 0.32432432432432434
- 分类最优模型: LightGBM
- 分类最优阈值: 0.29000000000000004

## 2. 可汇报图表
- 模型对比图: report_figures/model_compare_regression.png, report_figures/model_compare_classification.png
- 特征重要性图: report_figures/feature_importance_regression.png, report_figures/feature_importance_classification.png
- 分群画像图: report_figures/cluster_profile.png
- 公平性图: report_figures/fairness_disparity.png

## 3. 结论建议
- 以验证集指标选模，分类任务通过阈值搜索提升 F1。
- 时间窗特征可降低跨学期信息混杂，建议持续补齐行为日志时间戳。
- 公平性上，JG 字段的预测阳性率差距最大，约为 1.0000，建议做分组再校准。