from typing import List, Optional, Dict, Any
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from state.agent_status import AgentStatus
from state.pet_profile import PetProfile

class DiagnosisState(MessagesState):
    agent_status: Optional[AgentStatus]
    
    # --- multi-turn inquiry ---
    pet_profile: Optional[PetProfile]
    inquiry_turns: int = 0
    additional_inquiry_turns: int = 0
    diagnosis_turns: int = 0
    
    # --- RAG ---
    search_queries: List[str] = []
    retrieved_docs: List[str] = []
    query_rewrite_count: int = 0
    
    diagnosis_draft: Optional[Dict[str, Any]] = None


class MultiViewSearchQueries(BaseModel):
    """
    Generate 2 distinct search queries based on the PetProfile to maximize retrieval coverage.
    """
    # 视图 1: 模拟原始观察 (针对 Dense Vector)
    # 目标：生成像 "My cat keeps throwing up" 这样的句子，利用向量的语义相似性去撞击数据库里的 Observation
    simulated_observation: str = Field(
        description="A first-person, colloquial description simulating what the owner might have originally posted. E.g., 'My cat keeps throwing up.'"
    )
    
    # 视图 2: 医学/同义词扩展 (针对 Sparse/Keyword Search)
    # 目标：生成像 "Feline emesis, lethargy" 这样的专业词汇，利用 BM25 的精确匹配去撞击数据库里的专业描述
    medical_expansion: str = Field(
        description="A query using medical terminology and synonyms to match professional records. E.g., 'Feline emesis, lethargy, gastrointestinal distress.'"
    )

class DiagnosisActorOutput(BaseModel):
    """
    Actor 节点的输出：包含思维链（CoT）和唯一的诊断结论。
    """
    key_symptoms_analysis: str = Field(
        description="Brief analysis of the patient's symptoms based on the profile."
    )
    matched_doc_ids: List[str] = Field(
        description="List of Doc IDs from the retrieved context that support this diagnosis."
    )
    # 核心约束：只输出 1 个最可能的诊断
    most_likely_condition: str = Field(
        description="The SINGLE most probable medical condition identified. Do not list multiple."
    )
    reasoning: str = Field(
        description="Why is this the most likely condition compared to others? Cite evidence."
    )
    advice_for_owner: str = Field(
        description="Actionable advice for the pet owner (e.g., immediate ER visit, diet change, etc.)."
    )

class DiagnosisCriticOutput(BaseModel):
    """
    Critic 节点的输出：审核结果。
    """
    is_approved: bool = Field(
        description="True if the diagnosis is supported by evidence and safe; False if hallucinated or irrelevant."
    )
    critique: str = Field(
        description="Reason for approval or rejection."
    )
    final_response_to_user: str = Field(
        description="The final message to show to the user. If approved, polish the Actor's advice. If rejected, write a safe fallback message."
    )
