"""临床复核链路（LangGraph interrupt 版 HITL）。

与 `/v1/consult/stream` 的分工：
  - stream：患者端实时打字机，红旗即时高亮，结论落 pending 进复核队列（会话级 HITL）。
  - clinical：结构化会诊链路，结论在 LangGraph 的 `interrupt()` 处**真正挂起**，
    医生不确认就不结案。挂起期间完整 state 存在 SQLite checkpointer 里，
    服务重启也能续上——这是"高风险结论不由系统自己宣布结案"的硬保证。

状态机：
    POST /clinical/run            → 跑完 triage/各科/仲裁，在 review 处挂起
    GET  /clinical/{thread_id}    → 查是否挂起 + 待复核内容
    POST /clinical/{tid}/resume   → 医生提交结论，图从挂起处续跑至 END
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from core.security_guard import scan_input
from graph.consultant_graph import (
    get_pending_interrupt, resume_consult, run_consult_with_review,
)

router = APIRouter()


def _sanitize(text: str, limit: int) -> str:
    """截断 + 注入扫描：高危注入用净化文本替代原文（与流式链路一致）。"""
    text = (text or "")[:limit]
    res = scan_input(text)
    return res["sanitized"] if res.get("risk") == "high" else text


def _review_of(state: dict) -> dict | None:
    intr = (state or {}).get("__interrupt__")
    if not intr:
        return None
    first = intr[0]
    return {"value": getattr(first, "value", None), "id": getattr(first, "id", None)}


# 注意：这里用同步 def（不是 async def）——四路 LLM 并行是阻塞调用，
# FastAPI 会把同步端点丢到线程池，不会阻塞事件循环。
@router.post("/v1/consult/clinical/run")
def clinical_run(payload: dict):
    """跑一次结构化会诊，并在结案前挂起等医生复核。"""
    raw = payload.get("query", "")
    if not isinstance(raw, str) or not raw.strip():
        return JSONResponse(status_code=400, content={"detail": "query 不能为空。"})
    thread_id = str(payload.get("thread_id", "") or "") or uuid.uuid4().hex
    user_ctx = payload.get("user_context")
    if user_ctx is not None and not isinstance(user_ctx, dict):
        return JSONResponse(status_code=400, content={"detail": "user_context 必须为对象。"})

    state = run_consult_with_review(_sanitize(raw, 4000), thread_id, user_ctx)
    review = _review_of(state)
    body = {
        "thread_id": thread_id,
        "status": "pending_review" if review else "done",
        "review_request": review["value"] if review else None,
        "final_report": {} if review else state.get("final_report", {}),
        "audit": state.get("audit", []),
    }
    return body


@router.get("/v1/consult/clinical/{thread_id}")
def clinical_state(thread_id: str):
    """查询线程当前是否挂在复核点，以及待复核的内容。"""
    pending = get_pending_interrupt(thread_id)
    return {"thread_id": thread_id,
            "status": "pending_review" if pending else "unknown_or_done",
            "review_request": pending["value"] if pending else None}


@router.post("/v1/consult/clinical/{thread_id}/resume")
def clinical_resume(thread_id: str, payload: dict):
    """医生提交复核结论，恢复被挂起的会诊。rejected 的结论不外发给患者。"""
    status = str(payload.get("status", "")).strip()
    if status not in ("approved", "rejected"):
        return JSONResponse(status_code=400,
                            content={"detail": "status 必须为 approved 或 rejected。"})
    decision = {
        "status": status,
        "reviewer": str(payload.get("reviewer", "") or "")[:64],
        "note": str(payload.get("note", "") or "")[:500],
    }
    state = resume_consult(thread_id, decision)
    report = state.get("final_report", {}) or {}
    return {
        "thread_id": thread_id,
        "review": state.get("review", decision),
        "final_report": report,
        "withheld": bool(report.get("withheld")),
    }
