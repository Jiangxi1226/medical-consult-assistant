"""分层严格 JSON 测试：MODEL→GATE1(解析/结构)→GATE2(业务)→ACCEPT。

用 fake LLM 注入固定回复，验证 GATE 各层判定与「业务失败绝不悄悄 {}」。
不依赖真实 LLM / 网络。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pydantic import BaseModel, Field

from utils.json_utils import (
    strict_json, JsonResult, JsonGate, extract_json,
)


class _Box(BaseModel):
    name: str = Field(...)
    count: int = Field(0, ge=0, le=100)


class _FakeLLM:
    """按顺序吐出预设回复；可设成调用抛异常。"""
    def __init__(self, replies=None, error=False):
        self.replies = list(replies or [])
        self.calls = []
        self.error = error

    def chat(self, messages, temperature=0.1):
        self.calls.append(messages)
        if self.error:
            raise RuntimeError("LLM 网络错误")
        if not self.replies:
            return "{}"
        return self.replies.pop(0)


def test_extract_json_brace_matching():
    """括号配对：字符串里含 } 也不截断错位。"""
    raw = '前面的话 {"name":"a}b"} 后面的话'
    assert extract_json('```json\n{"a":1}\n```') == '{"a":1}'
    assert extract_json('说明：\n{"name":"x"}') == '{"name":"x"}'
    # 字符串内 } 不影响配对
    assert '"name":"a}b"' in extract_json(raw)


def test_extract_json_no_brace():
    assert extract_json("没有任何json") == "没有任何json"


def test_strict_json_accept_valid():
    llm = _FakeLLM(['{"name":"ok","count":3}'])
    res = strict_json(llm, "test", output_model=_Box)
    assert res.ok and res.accepted
    assert res.gate == JsonGate.ACCEPT.value
    assert res.data.name == "ok"


def test_strict_json_retries_then_struct_fail():
    """结构失败可重试，超过上限显式失败(非 ACCEPT)。"""
    llm = _FakeLLM(['{"name":123}', '{"name":456}', '{"name":789}'])  # 全结构非法
    res = strict_json(llm, "test", output_model=_Box, max_retries=2)
    assert not res.ok
    assert res.gate == JsonGate.STRUCTURE.value
    assert res.degraded is True


def test_strict_json_business_fail_no_retry():
    """业务失败不重试、显式标记 degraded，绝不悄悄返回空 {}。"""
    calls = {"n": 0}

    def validator(d):
        calls["n"] += 1
        return False, "业务值域非法"

    llm = _FakeLLM(['{"name":"x","count":30}'])  # 结构合法，业务校验拦截
    res = strict_json(llm, "test", output_model=_Box, business_validator=validator)
    assert not res.ok
    assert res.gate == JsonGate.BUSINESS.value
    assert res.data.count == 30, "业务校验应按约定返回结果"
    assert calls["n"] == 1, "业务失败不应重试"
    assert res.degraded is True


def test_strict_json_llm_error_marks_fail():
    llm = _FakeLLM(error=True)
    res = strict_json(llm, "test", output_model=_Box)
    assert not res.ok
    assert res.gate == JsonGate.FAIL.value


def test_strict_json_never_silent_empty_on_model_fail():
    """降级时 degraded 必须为 True，暴露失败态而不当作成功。"""
    llm = _FakeLLM(error=True)
    res = strict_json(llm, "test", output_model=_Box)
    assert res.degraded is True
