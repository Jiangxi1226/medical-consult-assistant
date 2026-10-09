"""会诊编排节点。

每个节点：
- 先用知识库规则(kb_loader)做硬校验/预筛(e.g. 红旗、禁忌)。
- 再交给 strict_json(Model→GATE1结构→GATE2业务)做结构化裁决。
- strict_json 失败 → 显式降级(标记 degraded 并说明原因)，绝不静默吞掉。
- 全程写 state["audit"] 供审计落盘。
"""
import time
from concurrent.futures import ThreadPoolExecutor

from core.llm import LLM
from core.prompts import (
    _TRIAGE, _DEPT, _PHARMACY, _RISK, _ARBITER,
)
from core.schemas import (
    TriageResult, DepartmentVerdict, PharmacyVerdict, RiskVerdict, ArbitrationResult,
)
from utils.json_utils import strict_json
from utils.kb_loader import check_red_flags, find_interactions, match_symptoms, extract_drugs
from utils.dept_router import route_departments, route_reason, ALL_DEPTS


def _audit(state, agent, ok, error="", extra=None):
    state["audit"].append({
        "agent": agent, "ok": ok, "error": error[:200] if error else "", **(extra or {}),
    })


# ---------------- 分诊 ----------------
def triage_node(state: dict) -> dict:
    t0 = time.perf_counter()
    ok, e = False, ""
    text = state["user_input"]
    rf = check_red_flags(text)
    state["triage"]["red_flags"] = rf
    if rf:
        state["triage"]["emergency"] = True
        state["triage"]["advice"] = rf[0]["advice"]
    try:
        llm = LLM()
        res = strict_json(
            llm,
            f"{_TRIAGE}\n\n患者主诉：{text}\n请输出分诊结果。",
            output_model=TriageResult,
        )
        ok = res.ok
        if ok:
            state["triage"].update(res.data.model_dump())
    except Exception as ex:
        ok, e = False, str(ex)
    _audit(state, "triage", ok, e, {"duration_ms": round((time.perf_counter() - t0) * 1000)})
    return state


# ---------------- 内科 / 外科(并行) ----------------
def _dept_agent(state: dict, slot: str, dept: str) -> None:
    text = state["user_input"]
    candidates = match_symptoms(text, dept)
    cand_text = "; ".join(
        f"{c['name']}(症状依据:{'/'.join(c['matched'])})" for c in candidates
    ) or "知识库无命中"
    try:
        llm = LLM()
        res = strict_json(
            llm,
            f"{_DEPT.format(dept='内科' if dept == 'internal' else '外科/骨科')}\n\n"
            f"患者主诉：{text}\n知识库候选：{cand_text}\n请给出该科会诊结论。",
            output_model=DepartmentVerdict,
        )
        if res.ok:
            state[slot]["result"] = res.data.model_dump()
            state[slot]["candidates"] = res.data.candidates
            state[slot]["evidence"] = res.data.evidence
            state[slot]["done"] = True
            state[slot]["degraded"] = False
            state[slot]["kb_candidates"] = candidates
        else:
            state[slot]["done"], state[slot]["degraded"] = True, True
            state[slot]["error"] = res.reason
        _audit(state, dept, res.ok, "" if res.ok else res.reason)
    except Exception as e:
        state[slot]["done"], state[slot]["degraded"] = True, True
        state[slot]["error"] = str(e)
        _audit(state, dept, False, str(e))


def _pharmacy_agent(state: dict) -> None:
    text = state["user_input"]
    # 规则预检：从主诉/用药文本里识别候选药物，找已知相互作用
    inter = find_interactions(extract_drugs(text))
    try:
        llm = LLM()
        res = strict_json(
            llm,
            f"{_PHARMACY}\n\n患者主诉/用药：{text}\n"
            f"知识库检出的相互作用：{inter or '无'}\n请输出药剂核对结论。",
            output_model=PharmacyVerdict,
            business_validator=lambda d: None if d.severity in ("none", "low", "medium", "high") else (False, "severity 非法"),
            temperature=0.1,
        )
        if res.ok:
            state["pharmacy"]["result"] = res.data.model_dump()
            state["pharmacy"]["interactions"] = res.data.interactions
            state["pharmacy"]["risk_notes"] = res.data.risk_notes
            state["pharmacy"]["done"], state["pharmacy"]["degraded"] = True, False
        else:
            state["pharmacy"]["done"], state["pharmacy"]["degraded"] = True, True
            state["pharmacy"]["error"] = res.reason
        _audit(state, "pharmacy", res.ok, "" if res.ok else res.reason)
    except Exception as e:
        state["pharmacy"]["done"], state["pharmacy"]["degraded"] = True, True
        state["pharmacy"]["error"] = str(e)
        _audit(state, "pharmacy", False, str(e))


def _risk_agent(state: dict) -> None:
    text = state["user_input"]
    rf = check_red_flags(text)
    try:
        llm = LLM()
        res = strict_json(
            llm,
            f"{_RISK}\n\n患者主诉：{text}\n知识库红旗命中：{rf or '无'}\n请输出风险核对结论。",
            output_model=RiskVerdict,
        )
        if res.ok:
            state["risk"]["result"] = res.data.model_dump()
            state["risk"]["red_flags"] = res.data.red_flags or [r["name"] for r in rf]
            state["risk"]["urgent"] = bool(res.data.urgent or rf)
            state["risk"]["advice"] = res.data.advice or (rf[0]["advice"] if rf else "")
            state["risk"]["done"], state["risk"]["degraded"] = True, False
        else:
            state["risk"]["done"], state["risk"]["degraded"] = True, True
            state["risk"]["error"] = res.reason
        _audit(state, "risk", res.ok, "" if res.ok else res.reason)
    except Exception as e:
        state["risk"]["done"], state["risk"]["degraded"] = True, True
        state["risk"]["error"] = str(e)
        _audit(state, "risk", False, str(e))


def parallel_consult(state: dict) -> dict:
    """并行会诊：临床科室按分诊结果动态选路 + 药剂/风险常驻横切关卡。

    两层次（对齐真实多学科会诊 MDT 结构）：
      - 临床科室（内科 / 外科·骨科）是【变量】：由分诊的科室倾向决定跑哪几路。
        分诊的价值正在于把不相关的科室筛掉——四路无条件全跑时分诊等于装饰。
      - 药剂审方、红旗预警是【常量】：任何病例都要过，属流程安全关卡，不参与路由。
    未被选中的科室显式标记 skipped，避免仲裁把"没跑"误读成"该科没意见"。
    """
    triage = state["triage"]
    has_rf = bool(triage.get("red_flags"))
    is_simple = bool(triage.get("is_simple", False))
    # 知识库症状命中（确定性）作为路由的第一依据，LLM 分诊文本仅作回落。
    kb_depts = []
    if match_symptoms(state["user_input"], "internal"):
        kb_depts.append("internal")
    if match_symptoms(state["user_input"], "surgical"):
        kb_depts.append("surgical")
    depts = route_departments(
        triage.get("department_hint", ""), triage.get("chief_symptom", ""),
        is_simple, has_rf, kb_depts,
    )
    reason = route_reason(depts, has_rf, is_simple)
    state["routing"] = {
        "depts": depts, "reason": reason,
        "skipped": [d for d in ALL_DEPTS if d not in depts],
    }
    for slot in ALL_DEPTS:
        if slot not in depts:
            state[slot]["done"] = True
            state[slot]["skipped"] = True
    _audit(state, "routing", True, "", {"depts": depts, "reason": reason})

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {slot: ex.submit(_dept_agent, state, slot, slot) for slot in depts}
        futures["pharmacy"] = ex.submit(_pharmacy_agent, state)
        futures["risk"] = ex.submit(_risk_agent, state)
        for f in futures.values():
            f.result()   # 收集异常
    return state


# ---------------- 仲裁 ----------------
def _slot_line(state: dict, key: str, label: str) -> str:
    """把一个会诊槽位渲染成仲裁可读的一行。

    关键：区分「本次未启用（分诊筛掉）」「会诊失败」「正常但结论为空」三态。
    若用 `if v` 过滤空值，失败/未跑的科室会从仲裁视野里静默消失，
    仲裁 LLM 便把"没有这一路"误读成"该科没意见"。
    """
    slot = state.get(key) or {}
    if slot.get("skipped"):
        return f"{label}：本次未启用（分诊筛掉的科室，非该科意见，勿据此断定该科无异常）"
    if slot.get("degraded"):
        return f"{label}：会诊失败，无有效结论（该科意见缺失，请在置信度上体现）"
    res = slot.get("result") or {}
    return f"{label}：{res}" if res else f"{label}：结论为空"


def arbiter_node(state: dict) -> dict:
    t0 = time.perf_counter()
    ok, e = False, ""
    text = state["user_input"]
    # 分诊节点把 TriageResult 字段铺在 state["triage"] 顶层(没有 "result" 键)，
    # 故这里不能像各科一样读 .get("result")，否则分诊信息恒为空。直接读顶层字段。
    _triage = state["triage"]
    triage_line = "分诊：" + str({
        "chief_symptom": _triage.get("chief_symptom", ""),
        "department_hint": _triage.get("department_hint", ""),
        "is_simple": _triage.get("is_simple", False),
        "emergency": _triage.get("emergency", False),
    })
    parts = [
        triage_line,
        _slot_line(state, "internal", "内科"),
        _slot_line(state, "surgical", "外科/骨科"),
        _slot_line(state, "pharmacy", "药剂"),
        _slot_line(state, "risk", "风险"),
    ]
    # 路由依据一并交给仲裁：让它知道"某科缺席"是分诊筛掉的，而非漏诊。
    routing = state.get("routing") or {}
    if routing.get("reason"):
        parts.append(f"本轮路由：{routing['reason']}")
    brief = "\n".join(parts)
    try:
        llm = LLM()
        res = strict_json(
            llm,
            f"{_ARBITER}\n\n患者主诉：{text}\n各方会诊结论：\n{brief}\n请仲裁。",
            output_model=ArbitrationResult,
        )
        ok = res.ok
        if ok:
            data = res.data.model_dump()
            state["arbitration"].update(data)
            # degraded 只表示「仲裁/会诊节点解析或结构化失败」，与「是否紧急」无关。
            # 紧急走 final_report["urgent"]，不反算为降级，避免与评估口径混淆。
            state["arbitration"]["degraded"] = False
            state["final_report"] = _build_report(state, data)
    except Exception as ex:
        ok, e = False, str(ex)
    _audit(state, "arbiter", ok, e, {"duration_ms": round((time.perf_counter() - t0) * 1000)})
    return state


_DEFAULT_DISCLAIMER = "辅助导诊建议，不做诊断，请以线下医生为准。"


def _build_report(state: dict, arb: dict) -> dict:
    # 免责声明为「必带」项：即便 LLM 显式返回空串也会被写成 default 之外的值，
    # 故此处无论 True/False 都强制兜底——只要为空就用默认文案。
    disclaimer = (arb.get("disclaimer") or "").strip() or _DEFAULT_DISCLAIMER
    report = {
        "summary": arb.get("primary", ""),
        "primary": arb.get("primary", ""),
        "confidence": arb.get("confidence", 0.0),
        "department": arb.get("department", ""),
        "conflicts": arb.get("conflicts", []),
        "check_items": arb.get("check_items", []),
        "disclaimers": [disclaimer],
    }
    if state["risk"].get("urgent"):
        report["urgent"] = True
        report["urgent_advice"] = state["risk"].get("advice", "请立即就医。")
    if any(s.get("degraded") for s in (state["internal"], state["surgical"],
                                       state["pharmacy"], state["risk"], state["arbitration"])):
        report["degraded"] = True
    return report
