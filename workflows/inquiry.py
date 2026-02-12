from langgraph.graph import StateGraph, END
from state.diagnosis import DiagnosisState
from agents.extractor import extractor_node
from agents.inquiry import inquiry_node

class InquiryWorkflow:
    def __init__(self):
        self.graph = None
    
    
    def build(self):
        inquiry_workflow = StateGraph(DiagnosisState)
        inquiry_workflow.add_node("extractor_node", extractor_node)
        inquiry_workflow.add_node("inquiry_node", inquiry_node)

        inquiry_workflow.set_entry_point("extractor_node")
        
        inquiry_workflow.add_edge("extractor_node", "inquiry_node")
        inquiry_workflow.add_edge("inquiry_node", END)
        
        self.graph = inquiry_workflow.compile()
        return self.graph
    
    def get_runnable(self):
        if not self.graph:
            self.build()
        return self.graph