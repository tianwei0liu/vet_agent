from state.diagnosis import DiagnosisState
from state.orchestrator import AgentStatus
from langgraph.graph import END

def diagnosis_router(state: DiagnosisState):
    """
    决定下一步去哪
    """
    agent_status = state.get("agent_status", "")
    if agent_status == AgentStatus.INQUIRY:
        return "inquiry_subgraph"
    elif agent_status == AgentStatus.DIAGNOSIS:
        return "diagnosis_subgraph"
    return END
