import logging
from langchain_core.messages import SystemMessage, AIMessage # <--- 确保导入 AIMessage
from state.diagnosis import DiagnosisState
from model.orchestrator import orchestrator_model
from state.diagnosis import DiagnosisCriticOutput, DiagnosisActorOutput # 确保路径正确

logger = logging.getLogger(__name__)

def diagnosis_critic_node(state: DiagnosisState):
    """
    Critic Node: 审核 Actor 的诊断是否存在幻觉或风险。
    """
    logger.info("--- Entering Diagnosis Critic Node ---")
    
    # 1. 获取输入
    profile = state.get("pet_profile")
    retrieved_docs = state.get("retrieved_docs", [])
    draft = state.get("diagnosis_draft")
    
    # 守卫逻辑：如果 Actor 挂了没有生成草稿，直接兜底
    if not draft:
        fallback_msg = "I'm sorry, I couldn't process the medical records to provide a diagnosis at this time. Please consult a veterinarian."
        # [修改] 使用 AIMessage，代表这是 AI 说的话
        return {"messages": [AIMessage(content=fallback_msg, name="VeterinaryAgent")]}

    # 2. 还原 Actor 对象
    # [安全访问] 使用 getattr 或者默认值，防止旧数据报错
    user_lang = getattr(profile, "language", "English")
    
    actor_output = DiagnosisActorOutput(**draft)

    # 3. 构造 Prompt
    critic_prompt = f"""
    # Role
    You are a Veterinary Clinical Supervisor. Your job is to validate the diagnosis made by a junior doctor (Actor).

    # Context
    - Patient Symptoms: {profile.symptoms} (Species: {getattr(profile.species, "value", str(profile.species))}; Breed: {getattr(profile.breed, "value", str(profile.breed))})
    - Retrieved Evidence:
    {chr(10).join(retrieved_docs)} 

    # Actor's Proposed Diagnosis
    - Condition: {actor_output.most_likely_condition}
    - Reasoning: {actor_output.reasoning}
    - Proposed Advice: {actor_output.advice_for_owner}

    # Validation Criteria (Pass/Fail)
    1. **Hallucination Check**: Does the 'Condition' actually appear in or is strongly inferred from the 'Retrieved Evidence'? If the evidence is irrelevant, you must REJECT.
    2. **Symptom Match**: Is the diagnosis logically consistent with the Patient Symptoms? 
    3. **Safety**: Does the advice include necessary disclaimers (e.g., "See a vet")?

    # Task
    - If APPROVED: Rewrite the 'Proposed Advice' into a warm, professional response for the user in {user_lang}. Include the Condition name and the reasoning.
    - IF REJECTED: Write a polite refusal message in {user_lang} stating that the system cannot determine the cause based on current data, and recommend seeing a vet.
    """

    # 4. 调用 LLM
    try:
        critic_model = orchestrator_model.with_structured_output(DiagnosisCriticOutput)
        
        critic_response: DiagnosisCriticOutput = critic_model.invoke([
            SystemMessage(content=critic_prompt)
        ])
        
        logger.info(f"Critic Decision: Approved={critic_response.is_approved} | Reason: {critic_response.critique}")
        
        # 5. 生成最终消息
        final_msg = critic_response.final_response_to_user
        
        # 6. 更新 State
        return {
            "messages": [AIMessage(content=final_msg, name="VeterinaryAgent")]
        }

    except Exception as e:
        logger.error(f"Critic Logic Failed: {e}")
        return {
            "messages": [AIMessage(content="System Error during validation. Please consult a vet.", name="System")]
        }