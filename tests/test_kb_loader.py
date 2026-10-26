"""知识库规则层测试：红旗召回 / 联用检出 / 症状匹配。

纯规则层（不依赖 LLM），验证「红旗硬保证、联用硬检出」，
与 eval/cases.json 的用例口径一致。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from utils.kb_loader import (
    check_red_flags, extract_drugs, find_interactions, match_symptoms,
)


def _cases(split: str | None = None):
    """读取用例。split=None 返回全部；'core' / 'adversarial' 只返回该组。

    分组的意义：核心集是**契约**（规则层必须做对），对抗集是**已知边界**
    （专门构造来让子串匹配出错的，不该当作回归红线的硬断言）。
    """
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "eval", "cases.json"), encoding="utf-8") as f:
        data = json.load(f)
    return data if split is None else [c for c in data if c.get("split", "core") == split]


def test_red_flag_recall():
    """所有期望急症的红旗用例都必须命中（漏召回 = 致命）。"""
    for c in _cases("core"):
        if not c.get("expect_urgent"):
            continue
        rf = check_red_flags(c["query"])
        assert rf, f"期望急症但未命中红旗: {c['id']} {c['query']}"


def test_red_flag_no_false_positive():
    """核心集里的非急症用例不得误命中红旗（过度就医）。"""
    for c in _cases("core"):
        if c.get("expect_urgent"):
            continue
        rf = check_red_flags(c["query"])
        assert not rf, f"非急症却误命中红旗: {c['id']} {c['query']}"


@pytest.mark.xfail(strict=False, reason=(
    "规则层为子串匹配，否定/第三人称/引用/既往史语境会误命中红旗——这是已知边界，"
    "不是回归。将来若加入否定识别，此测试会 XPASS，届时应去掉 xfail 转成正式契约。"))
def test_adversarial_red_flag_known_limitation():
    """对抗集：记录规则层当前的已知误命中。

    失败的用例就是系统的真实短板——把它写成测试而不是删掉，
    是为了让这条边界**始终可见**，而不是藏起来假装不存在。
    """
    hits = [f"{c['id']}:{c['query']}" for c in _cases("adversarial")
            if not c.get("expect_urgent") and check_red_flags(c["query"])]
    assert not hits, f"对抗集误命中红旗 {len(hits)} 例: {hits}"


def test_red_flag_urgency_sorted_emergent_first():
    """emergent 红旗应排在最前。"""
    rf = check_red_flags("突发胸痛伴冷汗，喘不上气；口角歪一边手抬不起来")
    assert rf and rf[0]["urgency"] == "emergent"


def test_interaction_anticoag_nsaid():
    """c06：抗凝 + NSAID 联用必须检出出血风险。"""
    drugs = extract_drugs("正在吃阿司匹林和华法林，今天又加了布洛芬")
    inter = find_interactions(drugs)
    ids = [i["id"] for i in inter]
    assert "INT_ANTICOAG_NSAID" in ids, "抗凝+NSAID 联用未检出"


def test_interaction_requires_two_distinct_drugs():
    """单药不构成联用。"""
    drugs = extract_drugs("一直在吃华法林")
    inter = find_interactions(drugs)
    assert inter == [], "仅单个药物不应判为联用"


def test_drug_extraction():
    drugs = extract_drugs("早上吃了布洛芬")
    assert "布洛芬" in drugs


def test_match_symptoms_internal():
    """c05：上腹痛反酸 → 消化内科候选。"""
    cand = match_symptoms("上腹痛反酸烧心，吃了东西更胀", "internal")
    assert any("溃疡" in c["name"] or "胃炎" in c["name"] for c in cand)


def test_match_symptoms_surgical():
    """c04：膝盖疼晨僵 → 骨关节炎候选。"""
    cand = match_symptoms("膝盖疼，早上起来僵硬", "surgical")
    assert any("骨关节" in c["name"] for c in cand)


def test_red_flags_kb_exists():
    """知识库文件必须存在，否则规则层直接崩。"""
    from utils import kb_loader
    assert kb_loader.red_flags(), "red_flags.json 为空或缺失"
    assert kb_loader.contraindications(), "contraindications.json 为空或缺失"
    assert kb_loader.symptoms(), "symptoms.json 为空或缺失"
