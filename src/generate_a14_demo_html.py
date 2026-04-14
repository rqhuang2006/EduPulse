from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
A14_DIR = ROOT / "outputs" / "a14"


def load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    group_profile = load_json(A14_DIR / "group_profile.json")
    demo_case = load_json(A14_DIR / "demo_case_student.json")
    reports = load_json(A14_DIR / "student_full_report_multi_agent.json")
    reports = reports if isinstance(reports, list) else []
    preview_reports = reports[:20]

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>A14 Demo Dashboard</title>
  <style>
    :root {{
      --bg: #f6f4ee;
      --card: #fffdf8;
      --ink: #1f2933;
      --muted: #66707a;
      --line: #d8d2c4;
      --accent: #b6542a;
      --accent-soft: #f3d5c7;
      --accent-deep: #7f2f10;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
      background:
        radial-gradient(circle at top left, #f5e7d6 0, transparent 35%),
        radial-gradient(circle at bottom right, #eadfd0 0, transparent 30%),
        var(--bg);
      color: var(--ink);
    }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 28px 20px 60px; }}
    .hero, .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 10px 24px rgba(52, 35, 18, 0.06);
    }}
    .hero {{ padding: 28px; }}
    .card {{ padding: 20px; }}
    h1,h2,h3 {{ margin: 0 0 10px; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.65; }}
    .grid {{ display: grid; grid-template-columns: repeat(12,1fr); gap: 18px; margin-top: 18px; }}
    .span-3 {{ grid-column: span 3; }}
    .span-4 {{ grid-column: span 4; }}
    .span-6 {{ grid-column: span 6; }}
    .span-12 {{ grid-column: span 12; }}
    .metric {{ font-size: 32px; font-weight: 700; color: var(--accent-deep); margin-top: 8px; }}
    .row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }}
    .tag, .chip {{
      display: inline-block; padding: 8px 12px; border-radius: 999px; background: #fff;
      border: 1px solid var(--line); font-size: 13px;
    }}
    .tag {{ background: var(--accent-soft); color: var(--accent-deep); border-color: transparent; }}
    .list {{ margin: 14px 0 0; padding-left: 18px; line-height: 1.75; }}
    .mono {{
      font-family: Consolas, monospace; font-size: 12px; white-space: pre-wrap;
      background: #fcfaf5; border: 1px solid var(--line); border-radius: 14px; padding: 12px; margin-top: 12px;
    }}
    .report {{ border-top: 1px dashed var(--line); padding-top: 14px; margin-top: 14px; }}
    .report:first-child {{ border-top: 0; padding-top: 0; margin-top: 0; }}
    .searchbox {{
      display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px;
    }}
    .searchbox input {{
      flex: 1; min-width: 240px; padding: 12px 14px; border-radius: 14px;
      border: 1px solid var(--line); background: #fff;
    }}
    .searchbox button {{
      padding: 12px 16px; border: 0; border-radius: 14px; background: var(--accent); color: #fff; cursor: pointer;
    }}
    .hidden {{ display: none; }}
    @media (max-width: 960px) {{
      .span-3,.span-4,.span-6,.span-12 {{ grid-column: span 12; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>A14 学生行为分析与干预 Demo</h1>
      <p>当前页面用于答辩展示“分析-解释-画像-干预-展示”闭环，并支持按学生 ID 检索多 Agent 报告。</p>
    </section>

    <section class="grid">
      <div class="card span-3">
        <h2>样本数</h2>
        <div class="metric">{group_profile.get("sample_count", 0)}</div>
        <p>融合主表中的学生数量</p>
      </div>
      <div class="card span-3">
        <h2>高风险占比</h2>
        <div class="metric">{group_profile.get("risk_level_ratio", {}).get("高风险", 0)}</div>
        <p>综合风险位于前 20% 的学生比例</p>
      </div>
      <div class="card span-3">
        <h2>模式数</h2>
        <div class="metric">{len(group_profile.get("pattern_ratio", {}))}</div>
        <p>当前规则识别出的行为模式类别数</p>
      </div>
      <div class="card span-3">
        <h2>机制数</h2>
        <div class="metric">{len(group_profile.get("dominant_map_ratio", {}))}</div>
        <p>当前群体中覆盖的主导机制类别数</p>
      </div>

      <div class="card span-4">
        <h2>风险分层</h2>
        <div class="row">
          {"".join(f'<span class="tag">{k}: {v}</span>' for k, v in group_profile.get("risk_level_ratio", {}).items())}
        </div>
      </div>
      <div class="card span-4">
        <h2>主导维度分布</h2>
        <div class="row">
          {"".join(f'<span class="tag">{k}: {v}</span>' for k, v in group_profile.get("dominant_dimension_ratio", {}).items())}
        </div>
      </div>
      <div class="card span-4">
        <h2>MAP 机制分布</h2>
        <div class="row">
          {"".join(f'<span class="tag">{k}: {v}</span>' for k, v in group_profile.get("dominant_map_ratio", {}).items())}
        </div>
      </div>

      <div class="card span-6">
        <h2>行为模式分布</h2>
        <ul class="list">
          {"".join(f'<li>{k}: {v}</li>' for k, v in group_profile.get("pattern_ratio", {}).items())}
        </ul>
      </div>
      <div class="card span-6">
        <h2>群体高频 SHAP 特征</h2>
        <ul class="list">
          {"".join(f'<li>{k}: {"、".join(v)}</li>' for k, v in group_profile.get("high_frequency_shap_features", {}).items())}
        </ul>
      </div>

      <div class="card span-12">
        <h2>答辩 Demo 单案例链路</h2>
        <div class="mono">{json.dumps(demo_case, ensure_ascii=False, indent=2)}</div>
      </div>

      <div class="card span-12">
        <h2>学生检索</h2>
        <p>输入学生 ID，可快速查看单学生多 Agent 报告。</p>
        <div class="searchbox">
          <input id="studentSearch" type="text" placeholder="例如：pjwtqxbj965" />
          <button onclick="searchStudent()">搜索</button>
        </div>
        <div id="studentResult" class="mono">请输入学生 ID 进行检索。</div>
      </div>

      <div class="card span-12">
        <h2>多 Agent 报告预览</h2>
        <p>默认展示前 20 条样例，完整结果见 `student_full_report_multi_agent.json`。</p>
        {"".join(
          f'''
          <div class="report">
            <h3>{item.get("student_id", "")}</h3>
            <p>{item.get("summary", "")}</p>
            <div class="row">
              <span class="chip">风险等级: {item.get("risk", {}).get("risk_level", "")}</span>
              <span class="chip">主导维度: {item.get("risk", {}).get("dominant_dimension", "")}</span>
              <span class="chip">模式: {item.get("behavior", {}).get("pattern_label", "")}</span>
              <span class="chip">优先级: {item.get("intervention", {}).get("priority", "")}</span>
            </div>
          </div>
          '''
          for item in preview_reports
        )}
      </div>
    </section>
  </div>
  <script>
    const reports = {json.dumps(reports, ensure_ascii=False)};
    function searchStudent() {{
      const keyword = document.getElementById('studentSearch').value.trim().toLowerCase();
      const target = document.getElementById('studentResult');
      if (!keyword) {{
        target.textContent = '请输入学生 ID 进行检索。';
        return;
      }}
      const found = reports.find(item => (item.student_id || '').toLowerCase() === keyword);
      if (!found) {{
        target.textContent = '未找到该学生，请检查 student_id 是否正确。';
        return;
      }}
      target.textContent = JSON.stringify(found, null, 2);
    }}
  </script>
</body>
</html>
"""

    output = A14_DIR / "demo_dashboard.html"
    output.write_text(html, encoding="utf-8")
    print(f"Demo dashboard generated in: {output}")


if __name__ == "__main__":
    main()
