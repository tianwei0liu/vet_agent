from state.orchestrator import OrchestratorState
from state.agent_status import AgentStatus
from langgraph.graph import END

def orchestrator_router(state: OrchestratorState):
    """
    决定下一步去哪
    """
    agent_status = state.get("agent_status", "")
    if agent_status == AgentStatus.INQUIRY:
        return "inquiry_subgraph"
    elif agent_status == AgentStatus.TREATMENT:
        return END
    elif agent_status == AgentStatus.KNOWLEDGE:
        return END
    return END
