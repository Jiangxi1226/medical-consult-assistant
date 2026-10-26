"""轻量链路追踪（移植自项目一，按本项目多线程场景改写）。

与项目一版本的区别：项目一用 `threading.local` 自动挂父子 span，适合单线程顺序流程；
本项目各科会诊跑在线程池里，`threading.local` 会让子线程拿不到父 span、树直接断掉。
这里改成**显式传递父 span**（`trace.child("internal")`），跨线程也能正确成树。

记录什么（都来自实测，不含任何估算值）：
  - 每个阶段的起止耗时
  - 流式首字延迟（TTFT）——用户体感的关键指标
  - 输出片段数（chunk 数，可近似反映生成量，但不等同于 token，不拿来充当 token 数）

落盘：`data/traces/{trace_id}.json`，可直接喂给可视化或做回归对比。
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager

_TRACE_DIR = os.getenv("MEDICAL_TRACE_DIR", "") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "traces"
)


class Span:
    """一个追踪单元：有起止时间、属性、事件、子节点。"""

    __slots__ = ("span_id", "name", "start", "end", "duration_ms",
                 "attributes", "events", "children", "status", "_lock")

    def __init__(self, name: str):
        self.span_id = uuid.uuid4().hex[:12]
        self.name = name
        self.start = time.perf_counter()
        self.end = None
        self.duration_ms = 0.0
        self.attributes: dict = {}
        self.events: list[dict] = []
        self.children: list[Span] = []
        self.status = "running"
        self._lock = threading.Lock()

    def set(self, key: str, value):
        self.attributes[key] = value
        return self

    def add_event(self, name: str, **attrs):
        self.events.append({"name": name, "at_ms": self._elapsed(), **attrs})

    def _elapsed(self) -> float:
        return round((time.perf_counter() - self.start) * 1000, 1)

    def finish(self, status: str | None = None) -> "Span":
        if self.end is None:
            self.end = time.perf_counter()
            self.duration_ms = round((self.end - self.start) * 1000, 1)
        if status:
            self.status = status
        elif self.status == "running":
            self.status = "ok"
        return self

    def child(self, name: str) -> "Span":
        """新建子 span（线程安全，可在任意线程调用）。"""
        span = Span(name)
        with self._lock:
            self.children.append(span)
        return span

    def to_dict(self) -> dict:
        with self._lock:
            children = [c.to_dict() for c in self.children]
        return {
            "span_id": self.span_id,
            "name": self.name,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
            "children": children,
        }


class Trace:
    """一次会诊的完整追踪。root 为根 span。"""

    def __init__(self, name: str, **attrs):
        self.trace_id = uuid.uuid4().hex[:16]
        self.root = Span(name)
        for k, v in attrs.items():
            self.root.set(k, v)

    def child(self, name: str) -> Span:
        return self.root.child(name)

    @contextmanager
    def span(self, name: str, **attrs):
        """上下文管理器：进出自动计时。"""
        s = self.root.child(name)
        for k, v in attrs.items():
            s.set(k, v)
        try:
            yield s
        except Exception as e:
            s.add_event("exception", message=str(e)[:200])
            s.finish("error")
            raise
        else:
            s.finish("ok")

    def finish(self, status: str = "ok") -> dict:
        self.root.finish(status)
        return self.to_dict()

    def to_dict(self) -> dict:
        return {"trace_id": self.trace_id, "root": self.root.to_dict()}

    def save(self, directory: str | None = None) -> str:
        """写 JSON 落盘，返回文件路径。失败返回空串（追踪不能拖垮主流程）。"""
        d = directory or _TRACE_DIR
        try:
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{self.trace_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            return path
        except Exception:
            return ""


def flatten(summary: dict, limit: int = 20) -> list[dict]:
    """把追踪树拍平成 [name, duration_ms]，便于在 final 事件里带上关键耗时。"""
    out = []

    def walk(node: dict, depth: int = 0):
        out.append({"name": ("  " * depth) + node["name"],
                    "ms": node["duration_ms"], "status": node["status"]})
        for c in node.get("children", []):
            walk(c, depth + 1)

    walk(summary.get("root", {}))
    return out[:limit]
