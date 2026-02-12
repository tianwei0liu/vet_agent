import logging
from langchain_core.messages import SystemMessage
from state.diagnosis import DiagnosisState
from model.orchestrator import orchestrator_model
from state.diagnosis import DiagnosisActorOutput

logger = logging.getLogger(__name__)

def diagnosis_actor_node(state: DiagnosisState):
    """
    Actor Node: 根据 Profile 和 Retrieved Docs 生成初步诊断。
    """
    logger.info("--- Entering Diagnosis Actor Node ---")
    
    # 1. 获取输入
    profile = state.get("pet_profile")
    retrieved_docs = state.get("retrieved_docs", [])
    
    # 守卫逻辑：如果没有检索到文档，Actor 无法工作
    if not retrieved_docs:
        logger.warning("No docs retrieved. Actor cannot diagnose.")
        return {
            "diagnosis_draft": None
        }

    user_lang = profile.language if profile.language else "English"
    
    # 2. 构造 Prompt
    # 注意：retrieved_docs 已经是格式化好的包含 Diagnosis 字段的字符串列表
    actor_prompt = f"""
    # Role
    You are a Senior Veterinary Diagnostician. Your job is to identify the SINGLE most likely medical condition based strictly on the provided Evidence.

    # Patient Profile
    - Species: {getattr(profile.species, "value", str(profile.species))}
    - Breed: {profile.breed}
    - Age: {profile.age}
    - Reported Symptoms: {profile.symptoms}

    # Evidence (Retrieved Medical Records)
    {chr(10).join(retrieved_docs)}  # chr(10) -> \n
    
    # Task
    1. Analyze the patient's symptoms against the Evidence.
    2. Identify the ONE condition that best matches the symptom pattern.
    3. Even if multiple conditions are possible, pick the one with the strongest evidence overlap.
    4. Formulate professional advice.

    # Constraints
    - IF the evidence contradicts the symptoms (e.g., evidence says "Cat" but patient is "Dog"), ignore that evidence.
    - IF no sufficient evidence exists to make a match, admit it in the reasoning.
    
    # Output Requirements
    - reasoning: You can write this in English or {user_lang}, whichever is more precise.
    - advice_for_owner: **MUST BE WRITTEN IN {user_lang}.** (This is crucial for the user to understand).
    """

    # 3. 调用 LLM
    try:
        # 绑定结构化输出
        actor_model = orchestrator_model.with_structured_output(DiagnosisActorOutput)
        
        actor_response: DiagnosisActorOutput = actor_model.invoke([
            SystemMessage(content=actor_prompt)
        ])
        
        logger.info(f"Actor Diagnosis: {actor_response.most_likely_condition}")
        
        # 4. 更新 State
        # 将结果存入 'diagnosis_draft' 传给 Critic，暂时不更新 messages
        return {
            "diagnosis_draft": actor_response.model_dump()
        }

    except Exception as e:
        logger.error(f"Actor Logic Failed: {e}")
        return {"diagnosis_draft": None}
