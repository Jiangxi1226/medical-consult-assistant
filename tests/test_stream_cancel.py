"""流式会诊的「客户端断开协同取消」测试。

背景：SSE 断开后后端默认会继续跑完整个会诊（4 路并行 LLM），纯烧 token。
修复：API 层检测断开 → 置位 stop_event → stream_consult/_stream_one 在
token/阶段边界检查并尽快收尾。此测试用假 LLM 验证取消信号真的生效。
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import graph.stream_flow as sf


class _FakeLLM:
    """假 LLM：无限产出 token，用于验证 stop_event 能中断拉取。"""

    def __init__(self, *a, **k):
        pass

    def chat_stream(self, messages, temperature=0.6):
        for i in range(10000):
            yield f"t{i}"


def test_stream_one_breaks_when_event_set(monkeypatch):
    """stop_event 已置位时，单科应一个 token 都不产出，只回传 done。"""
    monkeypatch.setattr(sf, "LLM", _FakeLLM)
    ev = threading.Event()
    ev.set()
    q = sf.queue.Queue()
    sf._stream_one("internal", "prompt", q, ev)

    items = []
    while not q.empty():
        items.append(q.get())
    kinds = [k for k, _, _ in items]
    assert kinds == ["done"], f"应只回传 done，实际 {kinds}"


def test_stream_one_runs_without_event(monkeypatch):
    """不传 stop_event 时行为不变——仍会产出 token（向后兼容）。"""
    monkeypatch.setattr(sf, "LLM", _FakeLLM)
    q = sf.queue.Queue()
    # 直接跑会产出上万 token，这里只验证"确实产出了 token"即停。
    t = threading.Thread(target=sf._stream_one, args=("internal", "p", q), daemon=True)
    t.start()
    kind, _, _ = q.get(timeout=3)
    assert kind == "token"


def test_stream_consult_returns_early_on_stop(monkeypatch):
    """断开后 stream_consult 应提前返回，不再产出 final（不发起仲裁）。"""
    monkeypatch.setattr(sf, "LLM", _FakeLLM)
    ev = threading.Event()
    ev.set()
    # 含时间词 → 跳过追问采集，直接进入并行会诊分支
    events = list(sf.stream_consult("发烧三天了，怎么办", "", stop_event=ev))
    assert all(e[0] != "final" for e in events), "断开后不应再产出 final"
