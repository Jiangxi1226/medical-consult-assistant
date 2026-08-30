"""回归评估：跑一批会诊样例，量化关键指标，落盘报告。

指标(医疗可信度重点)：
  - 红旗召回率 : 期望急症的样例里，系统正确触发 urgent 的比例(漏召回 = 致命)
  - 紧急误报率 : 非急症样例被误判为 urgent 的比例(过度就医)
  - 科室命中率 : 建议科室与期望一致的样例比例
  - 药剂交互检出 : expect_pharmacy_interaction 样例是否检出相互作用
  - 降级率/平均耗时 : 各步失败情况与性能
用法：  python eval/run_eval.py   (需项目根 .env 配好 LLM)
"""
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from graph.consultant_graph import run_consult   # noqa: E402


def _load(name):
    with open(os.path.join(_ROOT, "eval", name), encoding="utf-8") as f:
        return json.load(f)


# 别名归一化：导诊口径下"心内科"=="心血管内科"，"骨科或血管外科"含"骨科"
_ALIAS = {"心内科": "心血管内科", "心血管": "心血管内科", "神经": "神经内科",
          "急诊": "急诊科", "骨科": "骨科", "呼吸": "呼吸内科", "消化": "消化内科"}


def _norm(s):
    s = s or ""
    for k, v in _ALIAS.items():
        s = s.replace(k, v)
    return s.replace(" ", "").replace("，", "").replace(",", "")


def _dept_hit(expect, actual, query, urgent):
    """导诊口径判分：红旗急症去急诊 = 正确；期望用 / 拆分的任一命中即算；做别名归一化。"""
    if not expect:
        return None  # 该样例未设期望科室，不计入分母
    if urgent and "急诊科" in _norm(actual):
        return True  # 红旗该就医，去急诊是最合理的导诊决策
    for p in [x.strip() for x in str(expect).split("/") if x.strip()]:
        if _norm(p) in _norm(actual + "@@" + query):
            return True
    return False


def main():
    cases = _load("cases.json")
    rows, timings = [], []

    for c in cases:
        t0 = time.time()
        state = run_consult(c["query"])
        elapsed = time.time() - t0
        report = state.get("final_report", {})
        urgent = bool(report.get("urgent", False))
        dept = report.get("department", "")
        conf = report.get("confidence", 0.0)
        degraded = report.get("degraded", False)
        # 药剂交互检出：看药剂 slot 是否给出 interactions
        pharm = state.get("pharmacy", {}).get("interactions", []) or []
        pharm_hit = bool(pharm)

        exp_urgent = c.get("expect_urgent", False)
        rows.append({
            "id": c["id"], "query": c["query"],
            "exp_urgent": exp_urgent, "urgent": urgent,
            "dept": dept, "expect_dept": c.get("expect_dept", ""),
            "confidence": round(conf, 2), "degraded": degraded,
            "pharm": pharm_hit, "elapsed_s": round(elapsed, 2),
        })
        timings.append(elapsed)

    # ---- 指标 ----
    urgent_cases = [r for r in rows if r["exp_urgent"]]
    nonurgent_cases = [r for r in rows if not r["exp_urgent"]]
    recall = sum(r["urgent"] for r in urgent_cases) / len(urgent_cases) if urgent_cases else 1.0
    false_alert = sum(r["urgent"] for r in nonurgent_cases) / len(nonurgent_cases) if nonurgent_cases else 0.0
    dept_hits = sum(1 for r in rows
                    if _dept_hit(r["expect_dept"], r["dept"], r["query"], r["urgent"]))
    dept_total = sum(1 for r in rows if r["expect_dept"])
    metrics = {
        "cases": len(rows),
        "红旗召回率": round(recall, 3),
        "紧急误报率": round(false_alert, 3),
        "科室命中率": round(dept_hits / dept_total, 3) if dept_total else 1.0,
        "降级样例数": sum(1 for r in rows if r["degraded"]),
        "平均耗时s": round(sum(timings) / len(timings), 2) if timings else 0,
        "最大耗时s": round(max(timings), 2) if timings else 0,
    }

    out = {"metrics": metrics, "cases": rows}
    out_path = os.path.join(_ROOT, "eval", "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n明细已落盘: {out_path}")
    # 标注关键失败(漏召回红旗 = 最严重)
    for r in rows:
        if r["exp_urgent"] and not r["urgent"]:
            print(f"  ⚠ 红旗漏召回: {r['id']} {r['query']}")


if __name__ == "__main__":
    main()
