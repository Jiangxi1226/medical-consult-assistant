"""运行时流式会诊：并发多科流式打字机 + 规则层安全兜底 + 仲裁（分层严格 JSON）。

面向前端实时 SSE「问一句马上答」体验：
  - 内科/外科/药剂/风险 四个智能体各自流式生成、并发滚动输出（多 agent 并行）。
  - 规则层（红旗/禁忌/症状）并行做安全预检，命中红旗尽早发出 urgent 高亮。
  - 各科流式要点汇总后，由仲裁用 strict_json（MODEL→GATE1→GATE2→ACCEPT）生成
    结构化 final_report，保证结果可评估、可审计，且绝不静默失败。

与 graph/consultant_graph.py 的 run_consult 并存：本模块服务实时打字机，
run_consult 服务于回归评估/审计（结构化完整链路）。
"""
import queue
import re
from concurrent.futures import ThreadPoolExecutor

from core.llm import LLM
from core.prompts import _ARBITER, _CLARIFY
from core.schemas import ArbitrationResult, ClarifyResult
from utils.json_utils import strict_json
from utils.kb_loader import check_red_flags, find_interactions, match_symptoms, extract_drugs
from utils.dept_router import route_departments, route_reason, ALL_DEPTS, NAME


def _fmt(cands) -> str:
    return "; ".join(f"{c['name']}(依据:{'/'.join(c['matched'])})" for c in cands) or "无"


_STREAM_PROMPT = (
    "你是医疗导诊助手中的{dept}会诊智能体。基于下方患者主诉与知识库线索，"
    "用通俗口语逐步分析，说明该科视角的可能方向、需要进一步确认或检查什么。"
    "硬性要求：不得给出明确诊断；若命中紧急红旗须建议立即就医；"
    "若涉及药物禁忌或相互作用须明确提醒。每段简短、口语化、可分点。\n"
    "患者主诉：{text}\n知识库线索：{cand}"
)


def _stream_one(phase: str, prompt: str, out_queue: queue.Queue,
                stop_event=None) -> None:
    """单科流式生成器：逐 token 送入队列，结束时回传该科完整文本供仲裁。

    stop_event：客户端断开的协同取消信号。每个 token 之间检查一次——一旦置位
    立刻停止继续拉取，避免用户已经离开、后端还在烧 LLM。默认 None = 行为不变。
    """
    text = ""
    try:
        llm = LLM()
        for tok in llm.chat_stream([{"role": "user", "content": prompt}], temperature=0.6):
            if stop_event is not None and stop_event.is_set():
                break
            text += tok
            out_queue.put(("token", phase, tok))
    except Exception as e:
        out_queue.put(("token", phase, f"\n[该科分析暂不可用：{str(e)[:120]}]"))
    finally:
        out_queue.put(("done", phase, text))


def stream_consult(user_input: str, context: str = "", stop_event=None):
    """同步生成器，逐项 yield (event, data)。事件：
      urgent : {red_flags, advice}   红旗命中（尽早高亮，不问诊直接会诊）
      ask    : {questions:[{text,options}]}  首轮信息不足，先向患者追问关键症状
      token  : {phase, t}            打字机逐字
      final  : final_report dict     结构化结案（任何情况下都尽力送达，降级而非消失）
      error  : {error}               仅内部灾难性异常（理论不应出现）

    stop_event：可选的协同取消信号（threading.Event）。客户端断开时由 API 层置位，
    本生成器在每个关键节点检查它并尽快收尾返回，不再发起新的 LLM 调用。
    """
    def _stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    # context 为上一轮采集到的病史补充；拼接后作为完整"患者信息"用于安全预检与各科会诊。
    text = (user_input or "").strip()
    full_text = (" ".join(x for x in (context, text) if x)).strip() or text

    # ---- 规则层安全预检（快速、不依赖 LLM，红旗/禁忌硬保证） ----
    rf = check_red_flags(full_text)
    inter = find_interactions(extract_drugs(full_text))
    cand_internal = match_symptoms(full_text, "internal")
    cand_surgical = match_symptoms(full_text, "surgical")
    if rf:
        yield ("urgent", {"red_flags": [r["name"] for r in rf], "advice": rf[0]["advice"]})

    # ---- 逐轮问诊采集（多轮增量）：每轮用「主诉 + 已采集历史」判断还需问什么，直到采集智能体判定足够 ----
    # 是否足够不再靠粗糙的"有无时间词"预判，而是交给采集智能体的 need_clarify 决定；
    # 首轮主诉具体可跳过问诊（避免啰嗦），但一旦进入问诊循环，每轮都会重新评估。
    _has_time = re.search(r'(?:[0-9一二两三四五六七八九十几半多])\s*(?:天|日|周|月|年|小时|钟头|个月|半月|晚)', text or '')
    info_enough = bool(_has_time) and len(text) >= 8
    # 首轮、且非红旗、且主诉已够具体 → 直接会诊，不再追问。
    if not rf and not info_enough and not (context or "").strip():
        try:
            llm = LLM()
            # 喂入按症状命中的各科候选方向，让采集智能体把"该科需进一步确认项"也纳入追问。
            cand_hint = (
                f"内科候选方向：{_fmt(cand_internal) or '无'}\n"
                f"外科候选方向：{_fmt(cand_surgical) or '无'}"
            )
            cls = strict_json(
                llm,
                f"{_CLARIFY}\n\n患者主诉：{text}\n{cand_hint}\n"
                f"请判断当前信息是否足够；不足则输出当前最关键的一组追问。",
                output_model=ClarifyResult,
            )
            if cls.ok and cls.data.need_clarify and cls.data.questions:
                yield ("ask", {
                    "questions": [q.model_dump() for q in cls.data.questions],
                    "hint": "我会基于你的回答逐个追问，直到信息足够后进入会诊。",
                })
                return
        except Exception:
            pass

    # ---- 已有历史(context 非空)时：采集智能体重新评估还需问什么，不够就继续追问，够就放行会诊 ----
    # 每轮带 context 进来，_CLARIFY 会读到"已采集的答案"，据此判断是否还有缺失维度。
    # 追加上限：context 里已累计的问答轮数（数"答："）超过阈值即强制进入会诊，防止无效信息下无限追问。
    _asked = (context or "").count("答：")
    if not rf and (context or "").strip() and _asked < 8:
        try:
            llm = LLM()
            cand_hint = (
                f"内科候选方向：{_fmt(cand_internal) or '无'}\n"
                f"外科候选方向：{_fmt(cand_surgical) or '无'}"
            )
            cls = strict_json(
                llm,
                f"{_CLARIFY}\n\n患者主诉：{text}\n{cand_hint}\n"
                f"已采集到的回答：{context}。\n"
                f"请判断当前信息是否已足够；不足则输出当前最关键的一组追问（1-2 道）。",
                output_model=ClarifyResult,
            )
            if cls.ok and cls.data.need_clarify and cls.data.questions:
                yield ("ask", {
                    "questions": [q.model_dump() for q in cls.data.questions],
                    "hint": "请继续回答下面的问题；信息足够后我会进入会诊。",
                })
                return
        except Exception:
            pass

    # ---- 科室路由：临床科室按知识库命中决定，药剂/风险为常驻横切关卡 ----
    # 流式链路刻意不引入 LLM 分诊节点（实时体验优先），路由全部由规则层决定：
    # 知识库症状命中即跑该科；命中红旗则全科室；两者都没有则保守全跑。
    kb_depts = []
    if cand_internal:
        kb_depts.append("internal")
    if cand_surgical:
        kb_depts.append("surgical")
    depts = route_departments(has_red_flag=bool(rf), kb_depts=kb_depts)
    routing_reason = route_reason(depts, bool(rf), False)

    # ---- 各科提示词：统一用 full_text（含历史补充），保证各科看到完整患者信息 ----
    prompts = {}
    if "internal" in depts:
        prompts["internal"] = _STREAM_PROMPT.format(dept="内科", text=full_text, cand=_fmt(cand_internal))
    if "surgical" in depts:
        prompts["surgical"] = _STREAM_PROMPT.format(dept="外科/骨科", text=full_text, cand=_fmt(cand_surgical))
    # 药剂审方与红旗预警：任何病例都要过，不参与路由
    prompts["pharmacy"] = _STREAM_PROMPT.format(dept="药剂", text=full_text, cand=f"相互作用线索：{inter or '无'}")
    prompts["risk"] = _STREAM_PROMPT.format(dept="风险", text=full_text, cand=f"红旗线索：{rf or '无'}")

    # ---- 并发流式各科（线程池），主线程边收边转发。路数由路由决定，故按 len(prompts) 计数 ----
    q = queue.Queue()
    pool = ThreadPoolExecutor(max_workers=4)
    for ph, p in prompts.items():
        pool.submit(_stream_one, ph, p, q, stop_event)
    texts = {}
    done = 0
    try:
        while done < len(prompts):
            kind, phase, payload = q.get()
            if _stopped():
                break
            if kind == "token":
                yield ("token", {"phase": phase, "t": payload})
            elif kind == "done":
                texts[phase] = payload
                done += 1
    finally:
        pool.shutdown(wait=True)

    # 客户端已断开：各科线程已按 stop_event 收尾，此处不再发起仲裁调用，直接返回。
    if _stopped():
        return

    # ---- 仲裁：分层严格 JSON 产出结构化结案（可评估、绝不静默失败） ----
    brief = "\n".join(f"{k}：{texts.get(k, '')}" for k in prompts)
    # 未启用的科室显式告知仲裁：缺席是路由筛掉的，不等于该科无异常，避免被臆造结论。
    skipped = [NAME[d] for d in ALL_DEPTS if d not in depts]
    if skipped:
        brief += f"\n本轮路由：{routing_reason}（{'、'.join(skipped)}本次未启用，非该科意见）"
    urgent = bool(rf)  # 红旗由规则层硬决定（医疗安全核心，不依赖 LLM）
    fallback_disc = "辅助导诊建议，不做诊断，请以线下医生为准。"
    try:
        llm = LLM()
        res = strict_json(
            llm,
            f"{_ARBITER}\n\n患者主诉：{full_text}\n各方会诊要点：\n{brief}\n请给出会诊结论。",
            output_model=ArbitrationResult,
        )
        arb = res.data.model_dump() if res.ok else {}
    except Exception:
        arb = {}
    # 免责声明必带：LLM 显式空串也不允许覆盖默认文案。
    disclaimer = (arb.get("disclaimer") or "").strip() or fallback_disc
    report = {
        "summary": arb.get("primary", ""),
        "primary": arb.get("primary", ""),
        "confidence": arb.get("confidence", 0.0),
        "department": arb.get("department", ""),
        "conflicts": arb.get("conflicts", []),
        "check_items": arb.get("check_items", []),
        "disclaimers": [disclaimer],
        "degraded": not arb.get("primary", ""),
    }
    if urgent:
        report["urgent"] = True
        report["urgent_advice"] = rf[0]["advice"] if rf else "请立即就医。"
    # 附带审计用的完整过程数据（前端只消费 final 的展示字段，此键供落盘复盘）。
    report["_audit"] = {
        "text": full_text,
        "context": context or "",
        "red_flags": [r["name"] for r in rf],
        "candidates": {"internal": cand_internal, "surgical": cand_surgical},
        "interactions": inter or [],
        "dept_texts": texts,
        "routing": {"depts": depts, "reason": routing_reason},
        "clarify_done": bool(texts),  # 进入过会诊即视为采集完成（未走 ask 分支）
        "degraded": not arb.get("primary", ""),
    }
    yield ("final", report)
