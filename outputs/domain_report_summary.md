# 三域模型汇总报告

## 1. 任务与最优模型
- 域=learning | 任务=regression(avg_score) | 最优模型=RandomForest | r2=-0.39522499692285984
- 域=learning | 任务=classification(fail_flag) | 最优模型=LightGBM | f1=0.32432432432432434
- 域=life | 任务=classification(club_active_flag) | 最优模型=RandomForest | f1=0.44970414201183434
- 域=sport | 任务=regression(zf_score) | 最优模型=RandomForest | r2=0.29345592545550236
- 域=sport | 任务=classification(zf_grade) | 最优模型=RandomForest | macro_f1=0.3519745525579431

## 2. 产物目录
- 学习域输出：outputs/
- 生活域输出：outputs/life/
- 运动域输出：outputs/sport/

## 3. 说明
- 运动域当前同时提供分数回归与等级分类两个专用模型。
- 生活域模型标签由社团活动次数阈值构造，可通过训练参数调整。