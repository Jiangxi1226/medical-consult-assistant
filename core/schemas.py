"""会诊各 Agent 的结构化输出 schema(Pydantic)。

传给 strict_json 的 output_model：MODEL 边界注入约束 + GATE1 结构校验。
业务规则(禁忌/红旗)在 nodes 里用 kb_loader 先做 GATE2 性质的硬校验，
冲突即显式降级，绝不静默输出。
"""
from pydantic import BaseModel, Field


class TriageResult(BaseModel):
    chief_symptom: str = Field(..., description="从主诉中提炼的核心症状短语")
    department_hint: str = Field("", description="初步倾向的科室(可空)")
    is_simple: bool = Field(False, description="是否无需深度会诊的简单主诉")


class DepartmentVerdict(BaseModel):
    """内科 / 外科共用：该科基于症状给出的可能方向。"""
    candidates: list[str] = Field(..., description="该科可能的诊断方向(疾病名)")
    evidence: list[str] = Field(..., description="每条候选对应的症状依据")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="本会诊置信度(0-1)")
    priority: str = Field("low", description="low/medium/high")
    caveat: str = Field("", description="注意事项/需排除项；证据不足时如实说明")
    unsure: bool = Field(False, description="true 表示该科无法确定，需进一步检查")


class PharmacyVerdict(BaseModel):
    """药剂核对：交叉检查候选用药的禁忌/相互作用。"""
    interactions: list[str] = Field([], description="检出的药物相互作用/禁忌描述")
    risk_notes: list[str] = Field([], description="用药风险提示")
    severity: str = Field("none", description="none/low/medium/high")
    requires_doctor: bool = Field(False, description="true 表示必须由医生确认，勿自行用药")


class RiskVerdict(BaseModel):
    """风险核对：红旗征象与就医优先级。"""
    red_flags: list[str] = Field([], description="命中的红旗征象")
    urgent: bool = Field(False, description="true 表示需立即急诊")
    advice: str = Field("", description="就医指导")


class ArbitrationResult(BaseModel):
    """仲裁：综合多科会诊，收敛主方向/分歧/建议。"""
    primary: str = Field("", description="最可能方向(综合多方)")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    conflicts: list[str] = Field([], description="各科之间的分歧，显式暴露")
    department: str = Field("", description="建议挂号科室")
    check_items: list[str] = Field([], description="建议的检查/必查项")
    disclaimer: str = Field("辅助导诊建议，不做诊断，请以线下医生为准。", description="免责声明")


class FinalReport(BaseModel):
    """最终面向患者的报告(流式前端最终收敛为结构化卡)。"""
    summary: str = Field(...)
    primary: str
    confidence: float = 0.0
    department: str = ""
    conflicts: list[str] = []
    check_items: list[str] = []
    disclaimers: list[str] = []


class ClarifyQuestion(BaseModel):
    """单个追问：一个关键问题 + 候选选项(供患者点选,也可自行描述)。"""
    text: str = Field(..., description="要追问的简明问题(一句话)")
    options: list[str] = Field(default_factory=list, description="3-4 个候选选项;可为空表示自由回答")


class ClarifyResult(BaseModel):
    """问诊采集结果：判断是否需要先向用户澄清关键信息,才能安全导诊。"""
    need_clarify: bool = Field(False, description="true 表示信息不足以安全判断,需先追问")
    questions: list[ClarifyQuestion] = Field(default_factory=list, description="最多 2 个最关键追问")
