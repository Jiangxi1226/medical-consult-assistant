"""审计落盘。

把每次会诊的 输入(脱敏)/各步耗时与成败/最终报告/降级标记 追加到 data/audit.jsonl。
可回归复盘、定位"哪科失败/是否降级/哪一步最慢"，拒绝做成不可解释的黑盒 demo。
"""
import json
import os
import time
import re

_AUDIT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "audit.jsonl",
)

# 保留上限：只保留最近 MAX_RECORDS 条审计记录，避免日志无限膨胀(数据保留策略/合规)。
MAX_RECORDS = 5000

# 预编译脱敏正则，避免每次逐条重编译。
_MOBILE_RE = re.compile(r"1[3-9]\d{9}")
_IDCARD_RE = re.compile(r"\d{17}[\dXx]")


def _mask_text(text: str) -> str:
    """单条字符串脱敏：去除手机号/身份证等敏感模式，仅用于审计展示。

    顺序必须"先身份证、后手机号"：身份证号(18位)总包含一段可被手机号正则
    `1[3-9]\\d{9}` 命中的 11 位子串，若先跑手机号正则会把身份证内部截断成
    [手机号] 残片，反而遮不干净。先替换更长更特定的身份证，再处理手机号。
    """
    text = _IDCARD_RE.sub("[身份证]", str(text))
    text = _MOBILE_RE.sub("[手机号]", text)
    return text[:200]


def _mask(obj):
    """递归脱敏：把任意结构(含嵌套 dict/list)里的所有字符串字段统一脱敏。

    原先只对顶层 input 脱敏，但各科 slot 的 evidence / chief_symptom / advice、
    仲裁的 conflicts 等字段都可能携带患者原文里的手机号/身份证，必须全覆盖，
    否则审计文件本身就会泄露敏感信息。
    """
    if isinstance(obj, str):
        return _mask_text(obj)
    if isinstance(obj, dict):
        return {k: _mask(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mask(v) for v in obj]
    return obj


def _trim_old() -> None:
    """保留上限：若已超 MAX_RECORDS 则裁掉最早的记录，只留最近 N 条。"""
    try:
        with open(_AUDIT_PATH, encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        if len(lines) <= MAX_RECORDS:
            return
        with open(_AUDIT_PATH, "w", encoding="utf-8") as f:
            f.write("".join(lines[-MAX_RECORDS:]))
    except FileNotFoundError:
        pass
    except Exception:
        pass


def save_consult(state: dict) -> str:
    """落盘一次会诊。返回记录 id。"""
    rec_id = f"{int(time.time() * 1000)}"
    record = {
        "id": rec_id,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input": _mask_text(state.get("user_input", "")),
        "triage": _mask(state.get("triage", {})),
        "internal": _mask(state.get("internal", {})),
        "surgical": _mask(state.get("surgical", {})),
        "pharmacy": _mask(state.get("pharmacy", {})),
        "risk": _mask(state.get("risk", {})),
        "arbitration": _mask(state.get("arbitration", {})),
        "report": _mask(state.get("final_report", {})),
        "degraded": bool(state.get("final_report", {}).get("degraded", False)),
        "audit": _mask(state.get("audit", [])),
    }
    try:
        os.makedirs(os.path.dirname(_AUDIT_PATH), exist_ok=True)
        _trim_old()
        with open(_AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return rec_id
