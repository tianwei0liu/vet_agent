from langgraph.graph import StateGraph, END
from state.orchestrator import OrchestratorState
from state.orchestrator import IntentType
from agents.orchestrator import orchestrator_node
from routers.orchestrator import orchestrator_router
from workflows.inquiry import InquiryWorkflow
from workflows.diagnosis import DiagnosisWorkflow
from routers.diagnosis import diagnosis_router

class OrchestratorWorkflow:
    def __init__(self):
        self.graph = None
    
    
    def build(self, memory):
        orchestrator_workflow = StateGraph(OrchestratorState)
        # 1. 添加 Router 节点
        orchestrator_workflow.add_node("orchestrator_node", orchestrator_node)
        inquiry_workflow_instance = InquiryWorkflow()
        inquiry_subgraph = inquiry_workflow_instance.get_runnable()
        orchestrator_workflow.add_node("inquiry_subgraph", inquiry_subgraph)
        diagnosis_workflow_instance = DiagnosisWorkflow()
        diagnosis_subgraph = diagnosis_workflow_instance.get_runnable()
        orchestrator_workflow.add_node("diagnosis_subgraph", diagnosis_subgraph)

        # 3. 设置连线
        orchestrator_workflow.set_entry_point("orchestrator_node")

        orchestrator_workflow.add_conditional_edges(
            "orchestrator_node",
            orchestrator_router,
            {
                "inquiry_subgraph": "inquiry_subgraph",
                IntentType.INTENT_TREATMENT: END,
                IntentType.INTENT_KNOWLEDGE: END,
                IntentType.INTENT_FAILURE: END,
                END: END
            }
        )

        # 子图运行结束后，返回主图的 END (或者指向主图的下一个节点)
        orchestrator_workflow.add_conditional_edges(
            "inquiry_subgraph",
            diagnosis_router,
            {
                "inquiry_subgraph": END,
                "diagnosis_subgraph": "diagnosis_subgraph",
                END: END
            }
        )
        orchestrator_workflow.add_edge("diagnosis_subgraph", END)

        self.graph = orchestrator_workflow.compile(checkpointer=memory)
        return self.graph
    
    def get_runnable(self, memory):
        if not self.graph:
            self.build(memory)
        return self.graph