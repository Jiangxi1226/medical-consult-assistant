"""医疗知识库加载与规则检索。

各会诊 Agent 共享一套"依据源"：红旗征象(red_flags)、用药禁忌(contraindications)、
症状→分科映射(symptoms)。规则检索先于 LLM 判决，保证"有依据才输出、无依据不硬猜"。
"""
import json
import os
from functools import lru_cache

_KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kb")


@lru_cache(maxsize=1)
def _load(name: str):
    path = os.path.join(_KB_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def red_flags() -> list[dict]:
    return _load("red_flags.json")["urgent"]


def contraindications() -> dict:
    return _load("contraindications.json")


def symptoms() -> dict:
    return _load("symptoms.json")


# ---------------- 规则检索 ----------------

def check_red_flags(text: str) -> list[dict]:
    """扫描主诉文本命中哪些红旗征象。命中即提示立即就医。"""
    hits = []
    for rf in red_flags():
        matched = [s for s in rf["signals"] if s in text]
        if matched:
            hits.append({"id": rf["id"], "name": rf["name"],
                         "matched": matched, "advice": rf["advice"],
                         "urgency": rf["urgency"]})
    # 按紧急程度排序：emergent 优先
    return sorted(hits, key=lambda r: 0 if r["urgency"] == "emergent" else 1)


# ---------------- 药物识别(统一词表，供 nodes / stream_flow 共用) ----------------

# 候选用药关键词：用于从主诉/用药文本里挑出可能药物，再交给 find_interactions 查禁忌。
# 集中在此一处，避免散布在 nodes/stream_flow 造成两份词表漂移(此处修改即全局生效)。
DRUG_KEYS = ["布洛芬", "阿司匹林", "华法林", "头孢", "甲硝唑", "二甲双胍", "普利",
             "沙坦", "利尿剂", "他汀", "贝特", "SSRI", "氟西汀", "帕罗西汀", "地高辛",
             "抗生素", "感冒药", "阿莫西林", "美托洛尔", "缬沙坦", "氯吡格雷"]


def extract_drugs(text: str) -> list[str]:
    """轻量药物名识别(规则)：命中即返回，供 find_interactions 做禁忌/联用预检。"""
    return [k for k in DRUG_KEYS if k in (text or "")]


def find_interactions(drugs: list[str]) -> list[dict]:
    """在候选用药列表里找已知禁忌/相互作用。返回命中项。"""
    if not drugs:
        return []
    res = []
    for it in contraindications().get("interactions", []):
        hit = [d for d in it["drugs"] if any(d in dr for dr in drugs)]
        # 需要至少两个不同药物类别的关键词命中才构成"联用"
        if len(hit) >= 2:
            res.append({"id": it["id"], "risk": it["risk"],
                        "action": it["action"], "severity": it["severity"], "matched": hit})
    return res


def match_symptoms(text: str, dept: str) -> list[dict]:
    """按科室匹配症状→疾病候选，返回带 evidence/advice 的候选项。"""
    table = symptoms().get(dept, [])
    scored = []
    for ent in table:
        matched = [s for s in ent["symptoms"] if s in text]
        if matched:
            scored.append({"id": ent["id"], "name": ent["name"],
                           "matched": matched, "evidence": ent["evidence"],
                           "advice": ent["advice"], "department": ent["department"],
                           "priority": ent["priority"], "score": len(matched)})
    return sorted(scored, key=lambda x: x["score"], reverse=True)
