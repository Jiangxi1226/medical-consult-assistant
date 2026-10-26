"""会诊 LangGraph 编排。

triage(分诊) → parallel_consult(临床科室按分诊动态选路 + 药剂/风险常驻横切) → arbiter(仲裁) → END
每次调用用 new_state() 生成请求级隔离状态，并发请求互不影响。

两种编译形态：
  - `consultant`（默认，无 checkpointer）：评估/审计链路用，一次跑完不暂停。
  - `consult_with_review`（带 SQLite checkpointer + review 节点）：会诊结论在
    `interrupt()` 处挂起等医生复核，医生 `Command(resume=...)` 后才结案。
    这是"高风险结论不由系统自己宣布结案"的落地——挂起期间的 state 已落盘，
    进程重启也能续上。
"""
from __future__ import annotations

import os
import sqlite3

from langgraph.graph import StateGraph, END

from graph import nodes
from core.state import new_state

CHECKPOINT_DB = os.getenv("MEDICAL_CHECKPOINT_DB", "") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "checkpoints.db"
)


def _route_after_arbiter(state: dict) -> str:
    """需要复核 → 进 review（interrupt 挂起）；否则直接结束。"""
    return "review" if state.get("require_review") else "end"


def build_graph(checkpointer=None, with_review: bool = False):
    graph = StateGraph(dict)
    graph.add_node("triage", nodes.triage_node)
    graph.add_node("parallel_consult", nodes.parallel_consult)
    graph.add_node("arbiter", nodes.arbiter_node)
    graph.set_entry_point("triage")
    graph.add_edge("triage", "parallel_consult")
    graph.add_edge("parallel_consult", "arbiter")

    if with_review:
        graph.add_node("review", nodes.review_node)
        graph.add_conditional_edges(
            "arbiter", _route_after_arbiter, {"review": "review", "end": END}
        )
        graph.add_edge("review", END)
    else:
        graph.add_edge("arbiter", END)

    return graph.compile(checkpointer=checkpointer)


# 默认形态：与原行为完全一致（无 checkpointer、无中断）
consultant = build_graph()


# ---- 带人工复核的形态：懒加载，且依赖缺失时自动降级为内存 checkpointer ----
_review_graph = None
_ckpt_conn = None


def get_review_graph():
    """返回带 review 节点的图。首次调用时装配 checkpointer。

    SQLite checkpointer 不可用（未装 langgraph-checkpoint-sqlite）时退回
    MemorySaver——功能仍在，只是重启后挂起的会诊会丢。降级而非崩溃。
    """
    global _review_graph, _ckpt_conn
    if _review_graph is not None:
        return _review_graph

    saver = None
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        os.makedirs(os.path.dirname(CHECKPOINT_DB), exist_ok=True)
        # check_same_thread=False：SSE 在线程池里跑，同一连接会被多个线程使用
        _ckpt_conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
        saver = SqliteSaver(_ckpt_conn)
    except Exception:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            saver = MemorySaver()
        except Exception:
            saver = None

    _review_graph = build_graph(checkpointer=saver, with_review=True)
    return _review_graph


def run_consult(user_input: str, user_context: dict | None = None) -> dict:
    """执行一次会诊。返回完整 state(含 final_report/audit)。

    默认不中断——评估/回归链路依赖它一口气跑完。
    """
    state = new_state(user_input)
    if user_context:
        state["user_context"] = user_context
    final = consultant.invoke(state)
    return final


def run_consult_with_review(user_input: str, thread_id: str,
                            user_context: dict | None = None) -> dict:
    """执行一次会诊，并在结案前挂起等待医生复核。

    返回的 state 里若含 `__interrupt__`，说明图已挂起、等待
    `resume_consult(thread_id, {"status": "approved", ...})` 恢复。
    """
    graph = get_review_graph()
    state = new_state(user_input)
    state["require_review"] = True
    if user_context:
        state["user_context"] = user_context
    return graph.invoke(state, config={"configurable": {"thread_id": thread_id}})


def resume_consult(thread_id: str, decision: dict) -> dict:
    """医生提交复核结论后恢复被挂起的会诊。decision 形如
    {"status": "approved"|"rejected", "reviewer": "dr_x", "note": "..."}。"""
    from langgraph.types import Command

    graph = get_review_graph()
    return graph.invoke(Command(resume=decision),
                        config={"configurable": {"thread_id": thread_id}})


def get_pending_interrupt(thread_id: str) -> dict | None:
    """查询某线程是否正挂在复核点（供复核队列拉取待办）。"""
    graph = get_review_graph()
    try:
        snap = graph.get_state({"configurable": {"thread_id": thread_id}})
    except Exception:
        return None
    interrupts = getattr(snap, "interrupts", None) or ()
    if not interrupts:
        return None
    first = interrupts[0]
    return {"value": getattr(first, "value", None), "id": getattr(first, "id", None)}
