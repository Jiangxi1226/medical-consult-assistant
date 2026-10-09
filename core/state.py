"""请求级状态隔离。

根治了 football_agent 的"串卡/答案错位"：绝不用 module 级全局可变状态，
每次会诊都 new_state() 生成一个全新 dict，LangGraph 每次 invoke 传独立副本。
"""
from __future__ import annotations


def new_state(user_input: str) -> dict:
    """创建一次会诊的全新、请求级隔离的状态字典。

    与会话无关：同一个用户的不同提问、不同用户的并发请求，各自独立。
    """
    return {
        # 输入
        "user_input": user_input,
        "user_context": {},            # 年龄/过敏史/既往史/用药等（输入脱敏后）
        # 分诊结果
        "triage": {"red_flags": [], "chief_symptom": "", "department_hint": ""},
        # 本轮路由决策：启用了哪些临床科室、为什么(供审计与仲裁解释"某科为何缺席")
        "routing": {"depts": [], "reason": "", "skipped": []},
        # 各科会诊（并行写入，互不覆盖）
        # skipped=True 表示该路由由分诊筛掉、本次未启动(非失败，仲裁勿当作"该科无意见")
        "internal": {"done": False, "candidates": [], "evidence": [], "conflict": "",
                     "skipped": False},
        "surgical": {"done": False, "candidates": [], "evidence": [], "conflict": "",
                     "skipped": False},
        "pharmacy": {"done": False, "interactions": [], "risk_notes": []},
        "risk": {"done": False, "red_flags": [], "advice": ""},
        # 仲裁结果
        "arbitration": {"primary": "", "alternatives": [], "conflicts": [],
                        "confidence": 0.0, "department": "", "check_items": []},
        "final_report": {},
        "error": "",
        "audit": [],                  # 每步 trace(agent/耗时/错误)，审计落盘
    }


STATE_META = {
    "user_input": "患者主诉文本",
    "triage": "分诊台结果(红旗/主诉/科室提示)",
    "routing": "本轮路由(启用了哪些科室及依据)",
    "internal": "内科会诊",
    "surgical": "外科/骨科会诊",
    "pharmacy": "药剂核对(用药/相互作用)",
    "risk": "风险核对(红旗征象)",
    "arbitration": "仲裁与最终建议",
}
