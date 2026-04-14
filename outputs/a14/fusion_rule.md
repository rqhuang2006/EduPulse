# 融合规则说明

- 综合风险：`total_risk = mean(life_risk, study_risk, sport_risk)`
- 学习风险：直接使用学习域模型输出的 `pred_fail_proba`
- 生活风险：直接使用生活域模型输出的 `pred_prob`
- 运动风险：对 `pred_zf_score` 做反向 Min-Max 归一化，分数越低风险越高
- 风险等级：
  - 高风险：综合风险位于前 20%
  - 中风险：综合风险位于中间 60%
  - 低风险：综合风险位于后 20%
- 主导维度：`life_risk / study_risk / sport_risk` 中最大者
- MAP 汇总逻辑：对同一学生在不同子模型中的 `M/A/P` 取均值，再以最大值作为 `dominant_MAP`
- 行为模式：按综合风险、主导维度与 `dominant_MAP` 联合打标，保证输出不少于 4 类模式
