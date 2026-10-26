"""会诊会话持久化（SQLite）。

解决的问题：此前多轮问诊的"已采集病史"完全靠前端把 `context` 一路拼着回传，
服务端零状态。后果有三个——
  1. 用户在第三轮追问时断网/关页面，前面采到的信息**全丢**，只能重来；
  2. 服务端重启或前端刷新，会话无法恢复；
  3. 无法回答"这个患者上次问了什么"，复核与审计都接不上。

这里把会话落到 SQLite：`session_id` 为键，保存主诉、已采集上下文、轮次、
状态、最终报告与人工复核结论。

线程模型：SSE 生成器运行在线程池里，多个请求可能并发读写同一个库。
每次操作使用独立连接（短事务）+ 进程内锁串行化写入，避免 `database is locked`。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sessions.db"
)
DB_PATH = os.getenv("MEDICAL_SESSION_DB", "") or _DEFAULT_DB

_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS consult_sessions (
    session_id      TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT '',
    chief_complaint TEXT NOT NULL DEFAULT '',
    context         TEXT NOT NULL DEFAULT '',
    turns           INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'collecting',  -- collecting|consulting|done|reviewing
    report          TEXT NOT NULL DEFAULT '',
    review_status   TEXT NOT NULL DEFAULT '',            -- ''(无需复核)|pending|approved|rejected
    review_note     TEXT NOT NULL DEFAULT '',
    reviewer        TEXT NOT NULL DEFAULT '',
    risk_level      TEXT NOT NULL DEFAULT '',            -- normal|urgent
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON consult_sessions(updated_at);
CREATE INDEX IF NOT EXISTS idx_sessions_review  ON consult_sessions(review_status, updated_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """建表（幂等）。应用启动时调用一次即可。"""
    with _lock, _conn() as conn:
        conn.executescript(_SCHEMA)


def create_session(user_id: str = "", chief_complaint: str = "",
                   context: str = "") -> str:
    """新建会话，返回 session_id。"""
    sid = uuid.uuid4().hex
    ts = _now()
    with _lock, _conn() as conn:
        conn.execute(
            "INSERT INTO consult_sessions"
            "(session_id,user_id,chief_complaint,context,status,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (sid, user_id or "", chief_complaint or "", context or "", "collecting", ts, ts),
        )
    return sid


def get_session(session_id: str) -> dict | None:
    """读取会话；不存在返回 None。report 已反序列化为 dict。"""
    if not session_id:
        return None
    with _lock, _conn() as conn:
        row = conn.execute(
            "SELECT * FROM consult_sessions WHERE session_id=?", (session_id,)
        ).fetchone()
    if not row:
        return None
    data = dict(row)
    try:
        data["report"] = json.loads(data["report"]) if data["report"] else {}
    except Exception:
        data["report"] = {}
    return data


def update_session(session_id: str, **fields) -> bool:
    """更新指定字段。report 若是 dict 会序列化。返回是否命中记录。"""
    if not session_id or not fields:
        return False
    allowed = {"user_id", "chief_complaint", "context", "turns", "status",
               "report", "review_status", "review_note", "reviewer", "risk_level"}
    payload = {k: v for k, v in fields.items() if k in allowed}
    if not payload:
        return False
    if "report" in payload and isinstance(payload["report"], (dict, list)):
        payload["report"] = json.dumps(payload["report"], ensure_ascii=False)
    payload["updated_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in payload)
    with _lock, _conn() as conn:
        cur = conn.execute(
            f"UPDATE consult_sessions SET {sets} WHERE session_id=?",
            (*payload.values(), session_id),
        )
        return cur.rowcount > 0


def append_qa(session_id: str, question: str, answer: str) -> str:
    """追加一轮问答到 context，返回新的 context 字符串。

    context 沿用前端原有的 `问：…\\n答：…` 文本约定——stream_flow 里靠
    `context.count("答：")` 统计已采集轮次，格式不能改。
    """
    sess = get_session(session_id) or {}
    ctx = sess.get("context", "") or ""
    block = f"问：{question.strip()}\n答：{answer.strip()}"
    new_ctx = (ctx + "\n" + block).strip() if ctx else block
    turns = (sess.get("turns", 0) or 0) + 1
    update_session(session_id, context=new_ctx, turns=turns)
    return new_ctx


def append_question(session_id: str, question: str) -> str:
    """把本轮追问写进 context（只写"问："，答案等下一轮到达时补齐）。

    这样即便用户在这一步断线，重新连回来也知道"卡在第几问"。
    """
    sess = get_session(session_id) or {}
    ctx = (sess.get("context") or "").rstrip()
    new_ctx = f"{ctx}\n问：{question.strip()}" if ctx else f"问：{question.strip()}"
    update_session(session_id, context=new_ctx, status="collecting")
    return new_ctx


def list_pending_reviews(limit: int = 50) -> list[dict]:
    """待人工复核的会话（HITL 队列），按更新时间倒序。"""
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT session_id,user_id,chief_complaint,risk_level,created_at,updated_at"
            " FROM consult_sessions WHERE review_status='pending'"
            " ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def set_review(session_id: str, status: str, reviewer: str = "",
               note: str = "") -> bool:
    """写入人工复核结论。status: approved | rejected | pending"""
    if status not in ("approved", "rejected", "pending"):
        return False
    return update_session(session_id, review_status=status,
                          reviewer=reviewer, review_note=note)


# 模块加载即建表（幂等）。失败不阻断 import——真正用到时会再报错，
# 至少不会因为数据目录只读就把整个服务拖垮。
try:
    init_db()
except Exception:
    pass
