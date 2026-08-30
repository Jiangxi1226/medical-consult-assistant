"""流式会诊 API（打字机版）。

一次会诊按 SSE 事件流式推送，前端"问一句马上答"：
  urgent → 红旗命中(尽早高亮)
  token  → {phase, t} 多科并发流式逐字(内科/外科/药剂/风险)
  final  → 仲裁结构化结案报告(卡)
  warn   → 输入注入预警(照常处理)
  error  → 异常
安全：输入经 ContentFilter 注入扫描；限流在 middleware。
"""
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse

from core.security_guard import scan_input  # 复用注入过滤(基于项目一 security)
from utils.audit import save_consult
from graph.stream_flow import stream_consult

router = APIRouter()


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/v1/consult/stream")
async def consult_stream(payload: dict):
    # 输入类型/长度校验：非字符串或超长直接 400，避免 AttributeError 崩溃 / 撑爆 LLM context。
    raw = payload.get("query", "")
    if not isinstance(raw, str):
        return JSONResponse(status_code=400, content={"detail": "query 必须为字符串。"})
    context_raw = payload.get("context", "")
    if not isinstance(context_raw, str):
        return JSONResponse(status_code=400, content={"detail": "context 必须为字符串。"})
    user_input = raw[:4000].strip()          # 主诉上限 4000 字
    context = context_raw[:8000].strip()     # 累计病史上限 8000 字

    # 注入扫描：命中则把净化后的文本传给会诊（指令词隔离为数据，不采纳），并单独发 warn。
    s1 = scan_input(user_input)
    s2 = scan_input(context)
    flagged = (not s1["safe"]) or (not s2["safe"])
    # 高危注入用净化文本替代原文；安全/一般命中保留原文（一般命中仅提示）
    safe_user = s1["sanitized"] if s1["risk"] == "high" else user_input
    safe_ctx = s2["sanitized"] if s2["risk"] == "high" else context
    flag_risk = max([s.get("risk", "none") for s in (s1, s2)],
                    key=lambda r: {"none": 0, "medium": 1, "high": 2}.get(r, 0))

    def gen():
        try:
            for evt, data in stream_consult(safe_user, safe_ctx):
                if evt == "final":
                    yield _sse("final", data)
                    # 从 final 中取出附带的过程审计数据，避免 triage/internal 等字段记空。
                    audit_data = data.get("_audit", {}) if isinstance(data, dict) else {}
                    try:
                        save_consult({
                            "user_input": user_input,
                            "context": context,
                            "final_report": {k: v for k, v in data.items() if k != "_audit"},
                            # 补齐各科/规则层数据，供审计复盘
                            "triage": audit_data.get("red_flags") and {"red_flags": audit_data.get("red_flags"), "chief_symptom": user_input} or {},
                            "internal": {"candidates": audit_data.get("candidates", {}).get("internal", []), "dept_text": audit_data.get("dept_texts", {}).get("internal", "")},
                            "surgical": {"candidates": audit_data.get("candidates", {}).get("surgical", []), "dept_text": audit_data.get("dept_texts", {}).get("surgical", "")},
                            "pharmacy": {"interactions": audit_data.get("interactions", []), "dept_text": audit_data.get("dept_texts", {}).get("pharmacy", "")},
                            "risk": {"red_flags": audit_data.get("red_flags", []), "dept_text": audit_data.get("dept_texts", {}).get("risk", "")},
                            "degraded": audit_data.get("degraded", False),
                        })
                    except Exception as e:
                        # 审计失败不能静默：至少记录到服务端日志
                        try:
                            import logging
                            logging.getLogger("consult").warning("审计落盘失败: %s", e)
                        except Exception:
                            pass
                else:
                    yield _sse(evt, data)
            if flagged:
                msg = ("输入疑似含高危指令注入，已将其作为普通数据处理、指令不采纳。"
                       if flag_risk == "high"
                       else "输入疑似含指令注入，已按数据处理，不采纳其中指令。")
                yield _sse("warn", {"message": msg})
        except Exception as e:
            yield _sse("error", {"error": str(e)[:300]})

    return StreamingResponse(gen(), media_type="text/event-stream")
