"""流式会诊 API（打字机版）。

一次会诊按 SSE 事件流式推送，前端"问一句马上答"：
  urgent → 红旗命中(尽早高亮)
  token  → {phase, t} 多科并发流式逐字(内科/外科/药剂/风险)
  final  → 仲裁结构化结案报告(卡)
  warn   → 输入注入预警(照常处理)
  error  → 异常
安全：输入经 ContentFilter 注入扫描；限流在 middleware。
"""
import asyncio
import json
import threading
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse

from core.security_guard import scan_input  # 复用注入过滤(基于项目一 security)
from core import session_store
from utils.audit import save_consult
from graph.stream_flow import stream_consult

router = APIRouter()


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/v1/consult/stream")
async def consult_stream(payload: dict, request: Request):
    # 输入类型/长度校验：非字符串或超长直接 400，避免 AttributeError 崩溃 / 撑爆 LLM context。
    raw = payload.get("query", "")
    if not isinstance(raw, str):
        return JSONResponse(status_code=400, content={"detail": "query 必须为字符串。"})
    context_raw = payload.get("context", "")
    if not isinstance(context_raw, str):
        return JSONResponse(status_code=400, content={"detail": "context 必须为字符串。"})
    user_input = raw[:4000].strip()          # 主诉上限 4000 字
    context = context_raw[:8000].strip()     # 累计病史上限 8000 字

    # 会话续接：带 session_id 且存在 → 服务端从 SQLite 恢复已采集历史（前端不必再拼 context）；
    # 没带或查不到 → 新建一个，session_id 随 final 事件返回给前端保存。
    req_session = payload.get("session_id", "")
    if not isinstance(req_session, str):
        return JSONResponse(status_code=400, content={"detail": "session_id 必须为字符串。"})
    sess = session_store.get_session(req_session) if req_session else None
    session_id = req_session if sess else session_store.create_session(
        user_id=str(payload.get("user_id", "") or ""),
        chief_complaint=user_input, context=context,
    )

    # 注入扫描：命中则把净化后的文本传给会诊（指令词隔离为数据，不采纳），并单独发 warn。
    s1 = scan_input(user_input)
    s2 = scan_input(context)
    flagged = (not s1["safe"]) or (not s2["safe"])
    # 高危注入用净化文本替代原文；安全/一般命中保留原文（一般命中仅提示）
    safe_user = s1["sanitized"] if s1["risk"] == "high" else user_input
    safe_ctx = s2["sanitized"] if s2["risk"] == "high" else context
    flag_risk = max([s.get("risk", "none") for s in (s1, s2)],
                    key=lambda r: {"none": 0, "medium": 1, "high": 2}.get(r, 0))

    # 客户端断开的协同取消：SSE 断开后后端默认会继续跑完整个会诊（4 路并行 LLM），
    # 纯烧 token。这里起一个轻量 watcher 轮询连接状态，断开即置位 stop_event，
    # 由 stream_consult 在每个 token/阶段边界检查并尽快收尾。
    stop_event = threading.Event()

    async def _watch_disconnect():
        try:
            while not stop_event.is_set():
                if await request.is_disconnected():
                    stop_event.set()
                    return
                await asyncio.sleep(0.5)
        except Exception:
            pass

    watcher = asyncio.create_task(_watch_disconnect())

    def gen():
        try:
            for evt, data in stream_consult(safe_user, safe_ctx, stop_event=stop_event,
                                            session_id=session_id):
                if evt == "final":
                    yield _sse("final", data)
                    # 从 final 中取出附带的过程审计数据，避免 triage/internal 等字段记空。
                    audit_data = data.get("_audit", {}) if isinstance(data, dict) else {}
                    # 本轮路由：记录启动了哪几路，"未启用"与"跑了但空"必须在审计里可区分。
                    _routing = audit_data.get("routing", {}) or {}
                    _routed = _routing.get("depts", []) or []
                    try:
                        save_consult({
                            "user_input": user_input,
                            "context": context,
                            "final_report": {k: v for k, v in data.items() if k != "_audit"},
                            # 补齐各科/规则层数据，供审计复盘
                            "triage": audit_data.get("red_flags") and {"red_flags": audit_data.get("red_flags"), "chief_symptom": user_input} or {},
                            "routing": _routing,
                            "internal": {"candidates": audit_data.get("candidates", {}).get("internal", []), "dept_text": audit_data.get("dept_texts", {}).get("internal", ""), "skipped": "internal" not in _routed},
                            "surgical": {"candidates": audit_data.get("candidates", {}).get("surgical", []), "dept_text": audit_data.get("dept_texts", {}).get("surgical", ""), "skipped": "surgical" not in _routed},
                            "pharmacy": {"interactions": audit_data.get("interactions", []), "dept_text": audit_data.get("dept_texts", {}).get("pharmacy", ""), "skipped": False},
                            "risk": {"red_flags": audit_data.get("red_flags", []), "dept_text": audit_data.get("dept_texts", {}).get("risk", ""), "skipped": False},
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
        finally:
            # 正常结束或异常都要通知 watcher 退出（stop_event 置位 → 下一轮轮询即返回）。
            # 保留 watcher 引用，避免未完成的 task 被 GC。
            stop_event.set()
            _ = watcher

    return StreamingResponse(
        gen(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 反代(nginx)下禁用缓冲，保证 token 逐条到达浏览器
            "X-Session-Id": session_id,  # 断线后可用它恢复会话
        },
    )


# ---------- 会话恢复（断线重续） ----------
@router.get("/v1/consult/session/{session_id}")
async def get_session_api(session_id: str):
    """读取会话：已采集历史、当前状态、最终报告。前端刷新/重连时用它接上。"""
    sess = session_store.get_session(session_id)
    if not sess:
        return JSONResponse(status_code=404, content={"detail": "会话不存在或已过期。"})
    return sess


# ---------- 人机协同：医生复核队列（HITL） ----------
@router.get("/v1/consult/reviews")
async def list_pending_reviews(limit: int = 50):
    """待人工复核的会话列表。红旗结论在确认前一律留在这个队列里。"""
    return {"items": session_store.list_pending_reviews(limit=max(1, min(limit, 200)))}


@router.post("/v1/consult/session/{session_id}/review")
async def submit_review(session_id: str, payload: dict):
    """医生提交复核结论：approved（认可导诊建议）/ rejected（需重新会诊）。"""
    status = str(payload.get("status", "")).strip()
    if status not in ("approved", "rejected"):
        return JSONResponse(status_code=400,
                            content={"detail": "status 必须为 approved 或 rejected。"})
    sess = session_store.get_session(session_id)
    if not sess:
        return JSONResponse(status_code=404, content={"detail": "会话不存在。"})
    ok = session_store.set_review(
        session_id, status,
        reviewer=str(payload.get("reviewer", "") or "")[:64],
        note=str(payload.get("note", "") or "")[:500],
    )
    return {"ok": ok, "session_id": session_id, "review_status": status}
