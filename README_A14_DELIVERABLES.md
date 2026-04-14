# A14 交付层说明

当前仓库已具备三部分基础能力：

- 多域建模：`study / life / sport`
- SHAP 解释：个体与全局关键行为特征
- MAP 解释：`M / A / P` 行为机制归因

后续重点不再是重复建设解释层，而是把已有结果组织成赛题要求的完整交付物。为此新增两个脚本：

```bash
python src/generate_a14_deliverables.py
python src/a14_multi_agent.py
```

默认读取：

- 本仓库 `outputs/` 下的学习、生活、运动预测结果
- 同级目录 `fuchuang_shapmapl/fuchuang_final/output/` 下的 SHAP 与 MAP 结果

生成目录：

- `outputs/a14/`

核心产物包括：

- `fusion_student_master_table.csv`
- `student_pattern_label.csv`
- `pattern_summary.csv`
- `01_risk_distribution.csv` 到 `10_student_intervention.csv`
- `group_profile.json`
- `student_profile.csv`
- `student_intervention.csv`
- `student_full_report.json`
- `student_full_report_multi_agent.json`
- `demo_case_student.json`

多 Agent 编排采用固定职责拆分：

- `RiskAgent`
- `BehaviorAgent`
- `MechanismAgent`
- `InterventionAgent`
- `ReportAgent`

它们共同读取 `fusion_student_master_table.csv`，输出面向报告和 Demo 的统一 JSON 结果。

如果需要切换到大模型版 Agent：

1. 在仓库根目录创建 `.env`
2. 参考 `.env.example` 填入 API 信息
3. 运行：

```bash
python src/a14_multi_agent.py --mode llm
```

如果暂时没有 API，则继续使用：

```bash
python src/a14_multi_agent.py --mode rule
```
