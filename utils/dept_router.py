"""临床科室路由：决定本次会诊要启动哪几路。

设计要点（对齐真实多学科会诊 MDT 结构）：
  - 临床科室（内科 / 外科·骨科）是【变量】：由分诊/规则命中决定跑哪几路。
    分诊的意义正在于把不相关的科室筛掉——若四路无条件全跑，分诊就只是装饰。
  - 药剂审方、红旗预警是【常量】：任何病例都要过，属流程安全关卡，不参与路由。

判定顺序（安全优先于省调用）：
  1) 规则层命中红旗        → 全科室（急重症不省路，且不由 LLM 决定）
  2) 知识库症状命中（确定性）→ 只跑命中的科室
  3) LLM 分诊的科室倾向文本 → 关键词映射
  4) 倾向不明确            → 全科室（宁可多算一路，不漏诊）
  仅"判为简单主诉且无任何科室指向"时才真的不启动专科会诊。
"""
from __future__ import annotations

# 科室倾向文本 → 槽位的关键词表。故意做得宽：命中即跑该路，
# 宁可多跑（如"神经外科"同时命中"神经"与"外科"，两路都启动也不漏）。
_INTERNAL_KEYS = (
    "内科", "消化", "呼吸", "心血管", "心内", "心脏", "神经", "内分泌",
    "肾内", "肾脏", "血液", "感染", "传染", "风湿", "免疫", "发热",
    "胃肠", "肝病", "中毒",
)

_SURGICAL_KEYS = (
    "外科", "骨科", "创伤", "骨折", "扭伤", "关节", "脊柱", "颈椎", "腰椎",
    "皮肤", "伤口", "肿物", "疝", "痔", "乳腺", "泌尿外", "神外", "胸外",
    "普外", "软组织", "肌腱", "韧带", "烧伤",
)

ALL_DEPTS = ("internal", "surgical")
NAME = {"internal": "内科", "surgical": "外科/骨科"}


def route_departments(department_hint: str = "", chief_symptom: str = "",
                      is_simple: bool = False, has_red_flag: bool = False,
                      kb_depts: list[str] | None = None) -> list[str]:
    """返回本次要启动的临床科室槽位列表（可能为空）。

    department_hint：分诊给出的科室倾向自由文本（如"心血管内科、消化内科"）；
    chief_symptom 一并参与匹配以降低单字段误判；
    kb_depts：知识库症状匹配已命中的确定性科室，优先级高于 LLM 文本。
    返回 [] 表示不启动专科会诊，仅保留药剂/风险两道横切关卡。
    """
    # 1) 规则层红旗优先：急重症一律全科室过一遍，不省路、也不交给 LLM 决定。
    if has_red_flag:
        return list(ALL_DEPTS)

    # 2) 知识库症状命中是确定性的，优先于 LLM 分诊文本。
    if kb_depts:
        return [d for d in ALL_DEPTS if d in kb_depts]

    # 3) 回落到分诊文本的关键词映射。
    text = f"{department_hint or ''}{chief_symptom or ''}"
    depts = []
    if any(k in text for k in _INTERNAL_KEYS):
        depts.append("internal")
    if any(k in text for k in _SURGICAL_KEYS):
        depts.append("surgical")
    if depts:
        return depts

    # 4) 指向不明确：简单主诉可免专科会诊，否则保守全跑。
    if is_simple:
        return []
    return list(ALL_DEPTS)


def route_reason(depts: list[str], has_red_flag: bool, is_simple: bool) -> str:
    """给审计/前端一句可读的路由依据，便于解释"这次为什么只跑这几路"。"""
    if has_red_flag:
        return "命中红旗：急重症优先，全科室会诊"
    if not depts:
        return "判为简单主诉且无科室指向：仅走药剂/风险关卡"
    names = "、".join(NAME[d] for d in depts)
    if len(depts) == len(ALL_DEPTS):
        return "科室指向不明确：保守全科室会诊"
    return f"指向单一科室：仅启动{names}"
