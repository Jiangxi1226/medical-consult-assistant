"""会诊 LangGraph 编排。

triage(分诊) → parallel_consult(临床科室按分诊动态选路 + 药剂/风险常驻横切) → arbiter(仲裁) → END
每次调用用 new_state() 生成请求级隔离状态，并发请求互不影响。
"""
from langgraph.graph import StateGraph, END

from graph import nodes
from core.state import new_state


def build_graph():
    graph = StateGraph(dict)
    graph.add_node("triage", nodes.triage_node)
    graph.add_node("parallel_consult", nodes.parallel_consult)
    graph.add_node("arbiter", nodes.arbiter_node)
    graph.set_entry_point("triage")
    graph.add_edge("triage", "parallel_consult")
    graph.add_edge("parallel_consult", "arbiter")
    graph.add_edge("arbiter", END)
    return graph.compile()


consultant = build_graph()


def run_consult(user_input: str, user_context: dict | None = None) -> dict:
    """执行一次会诊。返回完整 state(含 final_report/audit)。"""
    state = new_state(user_input)
    if user_context:
        state["user_context"] = user_context
    final = consultant.invoke(state)
    return final
