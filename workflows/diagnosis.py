from langgraph.graph import StateGraph, START, END
from state.diagnosis import DiagnosisState

# RAG Diagnosis Phase Nodes
from agents.diagnosis_query_generator import diagnostic_query_generator_node
from agents.diagnosis_retriever import diagnosis_retriever_node
from agents.diagnosis_actor import diagnosis_actor_node
from agents.diagnosis_critic import diagnosis_critic_node

class DiagnosisWorkflow:
    def __init__(self):
        self.graph = None
    
    def build(self):
        # 初始化 StateGraph
        workflow = StateGraph(DiagnosisState)

        # --- 1. Add Nodes (仅包含 RAG 诊断链路的节点) ---
        workflow.add_node("diagnostic_query_generator_node", diagnostic_query_generator_node)
        workflow.add_node("diagnostic_retrieve_node", diagnosis_retriever_node)
        workflow.add_node("diagnosis_actor_node", diagnosis_actor_node)
        workflow.add_node("diagnosis_critic_node", diagnosis_critic_node)

        # --- 2. Define Edges (纯线性流程) ---

        # Entry Point: 直接开始生成查询
        # Orchestrator 传递进来的 State 应该已经包含了完整的 pet_profile
        workflow.add_edge(START, "diagnostic_query_generator_node")

        # Step 1 -> Step 2: 生成 Query 后去检索
        workflow.add_edge("diagnostic_query_generator_node", "diagnostic_retrieve_node")

        # Step 2 -> Step 3: 检索完成后给 Actor 诊断
        workflow.add_edge("diagnostic_retrieve_node", "diagnosis_actor_node")

        # Step 3 -> Step 4: 诊断完给 Critic 审核
        workflow.add_edge("diagnosis_actor_node", "diagnosis_critic_node")
        
        # Step 4 -> End: Critic 输出最终回复，流程结束
        # 此时控制权交还给 Orchestrator
        workflow.add_edge("diagnosis_critic_node", END)

        self.graph =  workflow.compile()
        return self.graph
    
    def get_runnable(self):
        if not self.graph:
            self.build()
        return self.graph
