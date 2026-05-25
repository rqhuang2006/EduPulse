# 项目阅读报告

生成时间：2026-04-21  
项目目录：`E:\connect421\1`

## 1. 项目概览

本项目是一个面向学生行为分析、风险预警和个性化干预的 Streamlit 应用，页面标题为“知行镜 - A14 学生行为分析系统”。应用以学习、生活、运动三域数据为基础，融合模型预测结果、SHAP 解释、MAP 行为机制归因和规则化干预建议，最终形成面向教师/管理端与学生自助端的交互式分析平台。

项目当前不是 Git 仓库，根目录没有 README。业务入口集中在 `app.py`，配套运行脚本位于 `src`，模型、数据和分析结果主要落在 `outputs` 与 `fuchuang_shapmapl/fuchuang_final`。

## 2. 技术栈与依赖

`requirements.txt` 声明的主要依赖如下：

- 数据处理：`numpy`、`pandas`、`openpyxl`
- 机器学习：`scikit-learn`、`joblib`、`lightgbm`、`xgboost`
- 可解释性：`shap`
- 可视化：`matplotlib`、`seaborn`、`altair`
- Web 应用：`streamlit`

项目还内置了 `streamlit_runtime` 目录，里面是解压后的 Streamlit 及其依赖包。`app.py` 与 `src/run_streamlit_app.py` 都会把该目录加入 Python 的 site 路径，因此该项目可以在缺少全局 Streamlit 环境时优先使用本地运行时。

## 3. 目录结构

核心目录如下：

| 路径 | 作用 |
| --- | --- |
| `app.py` | Streamlit 主应用，约 2538 行，包含 80 个顶层函数 |
| `src/run_streamlit_app.py` | 启动 Streamlit 应用的封装脚本 |
| `src/bootstrap_streamlit_runtime.py` | 从 `wheelhouse` 解压本地 Streamlit 运行时的脚本 |
| `requirements.txt` | Python 依赖清单 |
| `assets/risk_portrait.svg` | 群体画像页使用的视觉资产 |
| `outputs` | 主模型、三域数据集、评估指标、预测结果、A14 分析成果 |
| `outputs/a14` | 风险融合、画像、干预、演示链路等前端主要消费数据 |
| `fuchuang_shapmapl/fuchuang_final` | 另一套整理后的数据、模型、SHAP/MAP 输出 |
| `streamlit_runtime` | 本地打包的 Streamlit 及依赖运行时，不属于业务源码 |

目录规模粗略统计：

| 目录 | 文件数 | 大小 |
| --- | ---: | ---: |
| `assets` | 1 | 约 0 MB |
| `fuchuang_shapmapl` | 23 | 约 667.57 MB |
| `outputs` | 75 | 约 1105.69 MB |
| `src` | 8 | 约 0.11 MB |
| `streamlit_runtime` | 5899 | 约 236.02 MB |

非运行时业务文件类型统计：CSV 54 个、JSON 16 个、Joblib 12 个、PNG 7 个、Markdown 4 个、Python 3 个。

## 4. 运行方式

推荐从项目目录运行：

```powershell
cd E:\connect421\1
python .\src\run_streamlit_app.py
```

该脚本会：

1. 把用户 site-packages 和 `streamlit_runtime` 加入依赖搜索路径。
2. 如果本地 Streamlit runtime 不存在，则调用 `bootstrap_streamlit_runtime.py` 从 `wheelhouse` 解压。
3. 等价执行 `streamlit run app.py --server.headless true --browser.gatherUsageStats false`。

如果本机已安装依赖，也可以直接运行：

```powershell
cd E:\connect421\1
streamlit run .\app.py
```

## 5. 应用主流程

`app.py` 的整体流程是：

1. 设置路径常量、标签映射、字段别名、模型输入 schema。
2. 通过 `prepare_data()` 加载数据。
3. 初始化 Streamlit session state。
4. 渲染侧边栏、全局筛选、角色切换和页面导航。
5. 根据当前页面调用对应的 `render_*` 函数。

数据加载采用“合同接口优先，本地数据回退”的策略：

- 默认后端地址：`http://127.0.0.1:8000`
- 拉取最新结果：`GET /harness/result/latest`
- 触发三域运行：`POST /harness/run`
- 后端不可达或返回不足时，使用 `outputs/a14` 和 `outputs` 下的本地文件。

核心数据入口在 `prepare_data()`，主要读取：

- `outputs/a14/fusion_student_master_table.csv`
- `outputs/a14/group_profile.json`
- `outputs/a14/demo_case_student.json`
- `outputs/a14/pattern_summary.csv`
- `outputs/a14/student_intervention.csv`
- `outputs/a14/student_full_report_multi_agent.json`
- `outputs/feature_dataset.csv`
- `outputs/life/feature_dataset.csv`
- `outputs/sport/feature_dataset.csv`
- `outputs/sport/predictions_full.csv`

## 6. 页面功能

教师/管理端包含 8 个页面：

| 页面 | 主要功能 |
| --- | --- |
| 全局监控中枢 | 展示合同接口状态、本地/实时数据来源、总体高风险学生、主导机制、重点学生 |
| 项目指标体系 | 展示分析成果清单、模型评估、画像与干预成果验证 |
| 群体行为态势 | 按行为模式查看群体画像、风险结构、模式分布 |
| 学生数字档案 | 查看单个学生的风险、画像、解释和干预建议 |
| 精准预警干预 | 聚焦高风险学生与干预优先级 |
| 演化推进预测 | 支持手动录入或上传表格，对新增学生做三域风险推演 |
| AI 辅导员助理 | 基于规则和本地数据生成解释、干预和报告类回答 |
| 底层链路追踪 | 展示从子模型风险、融合、SHAP、MAP 到干预输出的链路 |

学生自助端包含 3 个页面：

| 页面 | 主要功能 |
| --- | --- |
| 我的成长主页 | 面向学生展示个人状态概览 |
| 状态自评与倾诉 | 主观状态填写/表达入口 |
| 我的AI树洞 | 学生侧对话入口 |

## 7. 数据与模型资产

主要数据规模：

| 文件 | 行数 | 列数 | 说明 |
| --- | ---: | ---: | --- |
| `outputs/feature_dataset.csv` | 20081 | 34 | 学习域特征数据 |
| `outputs/life/feature_dataset.csv` | 2506 | 11 | 生活域特征数据 |
| `outputs/sport/feature_dataset.csv` | 9751 | 25 | 运动域特征数据 |
| `outputs/a14/fusion_student_master_table.csv` | 2501 | 27 | 前端核心融合主表 |
| `outputs/a14/student_profile.csv` | 2501 | 5 | 个体画像 |
| `outputs/a14/student_intervention.csv` | 2501 | 4 | 个性化干预建议 |

主要模型文件：

| 文件 | 说明 |
| --- | --- |
| `outputs/models/best_classification_model.joblib` | 学习域分类模型 |
| `outputs/models/best_regression_model.joblib` | 学习域回归模型 |
| `outputs/models/cluster_model.joblib` | 学习域聚类模型 |
| `outputs/life/models/best_life_model.joblib` | 生活域模型 |
| `outputs/sport/regression/best_sport_regression_model.joblib` | 运动域回归模型 |
| `outputs/sport/classification/best_sport_classification_model.joblib` | 运动域分类模型 |
| `outputs/sport/models/best_sport_model.joblib` | 运动域模型副本/汇总模型 |

`fuchuang_shapmapl/fuchuang_final/models` 下存在与 `outputs` 高度重复的模型产物，会明显增加项目体积。

## 8. 模型效果摘要

来自 `outputs/metrics.json` 与各域指标文件：

### 学习域

- 样本总数：20081
- 训练集：19749
- 验证集：157
- 测试集：175
- 特征数：25
- 回归最优模型：RandomForest
- 回归测试 RMSE：17.3646
- 回归测试 MAE：10.0184
- 回归测试 R2：-0.4369
- 分类最优模型：RandomForest
- 分类测试 AUC：0.8935
- 分类测试 F1：0.2446
- 分类最优阈值：0.17
- 聚类数：4

### 生活域

- 样本数：2506
- 最优模型：RandomForest
- 最优阈值：0.05
- F1：0.4497
- AUC：0.5069
- Accuracy：0.2901

### 运动域

- 回归样本数：9751
- 回归最优模型：RandomForest
- 回归 RMSE：8.5665
- 回归 MAE：6.3540
- 回归 R2：0.2950
- 分类最优模型：RandomForest
- 分类 Accuracy：0.5581
- 分类 Macro-F1：0.3572
- 分类标签：A、B、C、D、E

### A14 融合画像

`outputs/a14/group_profile.json` 显示：

- 样本数：2501
- 高风险占比：20.03%
- 中风险占比：59.94%
- 低风险占比：20.03%
- 主导维度：生活 66.61%、运动 27.11%、学习 6.28%
- 平均风险：生活 0.5457、学习 0.0761、运动 0.4186、综合 0.3244
- 行为模式占比最高的是“提示缺失型”，占 45.18%

## 9. 新增学生预测逻辑

“演化推进预测”页面支持两种输入：

1. 手动录入：按学习、生活、运动三域字段输入。
2. 表格上传：支持 CSV/XLSX/XLS，自动根据中文字段名或特征字段名匹配列。

预测流程：

1. 将输入标准化到三域 schema。
2. 根据历史特征表补齐模型需要但用户未输入的字段，数值列取中位数，类别列取众数。
3. 加载本地 joblib 模型。
4. 学习域输出失败风险概率和预测成绩。
5. 生活域输出风险概率。
6. 运动域输出预测体测分，并根据历史预测分区间换算运动风险。
7. 三域风险取平均得到综合风险，再根据现有主表的 20%/80% 分位点划分低/中/高风险。

该逻辑复用现有模型，不重新训练模型。

## 10. 主要产物

A14 分析成果清单包括：

- 综合风险分布：`01_risk_distribution.csv`
- 三维风险对比：`02_dimension_comparison.csv`
- 风险等级分层：`03_risk_level_stats.csv`
- 主导维度分布：`04_dominant_dimension_stats.csv`
- SHAP 全局特征：`05_shap_global_summary.csv`
- MAP 机制占比：`06_map_summary.csv`
- 行为模式发现：`07_pattern_summary.csv`
- 群体画像：`08_group_profile.json`
- 个体画像：`09_student_profile.csv`
- 干预建议：`10_student_intervention.csv`

可汇报图表位于 `outputs/report_figures`：

- `cluster_profile.png`
- `fairness_disparity.png`
- `feature_importance_classification.png`
- `feature_importance_regression.png`
- `model_compare_classification.png`
- `model_compare_regression.png`

## 11. 代码质量与风险点

1. 主应用过于集中  
   `app.py` 约 2538 行，包含数据加载、模型推理、业务规则、页面渲染、CSS、接口调用和对话逻辑。后续维护建议拆分为 `data_loader`、`predictor`、`pages`、`components`、`services` 等模块。

2. 缺少 README 和项目级运行说明  
   当前需要通过阅读代码才能知道启动方式、数据来源和页面结构。建议补齐 README，说明运行命令、目录含义、模型产物来源、接口依赖和常见问题。

3. 项目不是 Git 仓库  
   当前无法追踪变更历史、分支和提交。建议初始化 Git 并增加 `.gitignore`，尤其排除 `__pycache__`、`.DS_Store`、临时文件和可再生成的大模型/运行时产物。

4. 模型文件重复且体积较大  
   `outputs` 与 `fuchuang_shapmapl/fuchuang_final` 中存在重复模型，项目体积超过 2GB。建议明确唯一模型源，另一个目录只保留引用说明或生成脚本。

5. `streamlit_runtime` 被提交为源码目录  
   该目录有 5899 个文件，约 236MB。适合交付包，但不适合作为常规源码仓库内容。建议改为 wheelhouse/bootstrap 或环境安装说明。

6. 部分路径是可选或历史遗留  
   `MODELING_ROOT = ROOT / "fuchuangsai2" / "output"`，但当前项目中没有 `fuchuangsai2` 目录；因此“高置信度核心预警模型引擎”页面的 `eval_df` 可能为空。`outputs_next/study` 也是可选路径，当前不存在时会走 legacy 模型。

7. 模型指标需要谨慎解释  
   学习域回归 R2 为负，说明测试集回归效果弱于简单均值基线；生活域 AUC 接近 0.5，区分能力有限；分类 F1 整体偏低。应用可以展示结果，但对外汇报时应明确模型效果边界。

8. 没有发现测试代码  
   目录中带 `test` 的文件主要是预测测试集 CSV，没有单元测试、集成测试或页面测试。建议至少补充数据加载、字段匹配、新增学生预测和核心规则的测试。

9. 运行时 pycache 写入权限异常  
   `python -m py_compile` 在当前环境中会因写入/重命名 `.pyc` 被拒绝而失败；改用内存编译读取源码后，`app.py`、`src/bootstrap_streamlit_runtime.py`、`src/run_streamlit_app.py` 均通过语法检查。

## 12. 建议的下一步

优先级较高的改进：

1. 补 README：包含项目定位、启动方式、目录说明、接口说明、数据产物说明。
2. 拆分 `app.py`：先把数据加载、模型推理、页面组件分出去，降低单文件维护成本。
3. 清理大文件策略：确认模型和运行时是否作为交付包保留；源码仓库应使用 `.gitignore` 或外部制品管理。
4. 修正/说明历史路径：明确 `fuchuangsai2`、`outputs_next` 是否仍然需要。
5. 增加测试：优先覆盖 `prepare_data()`、`standardize_record()`、`score_new_student()`、字段上传识别和风险等级划分。
6. 复核模型汇报口径：对 R2、AUC、F1 偏弱的域给出限制说明，避免把演示系统指标包装成高置信生产模型。

总体判断：该项目已经具备较完整的演示型产品形态，数据产物、模型产物和 Streamlit 前端已经打通；当前最大问题不在功能缺失，而在工程组织、产物体积、可复现说明和模型效果边界表达。
