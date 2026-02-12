import logging
from langchain_core.messages import SystemMessage

# 注意：这里导入了刚才新建的 schema
from state.diagnosis import MultiViewSearchQueries
from state.diagnosis import DiagnosisState

# 复用 orchestrator_model (DeepSeek-Chat, Temp=0)，因为它指令遵循能力强且便宜
from model.orchestrator import orchestrator_model


logger = logging.getLogger(__name__)


def diagnostic_query_generator_node(state: DiagnosisState):
    logger.info("--- Entering Query Generator Node (Dual-View) ---")

    # 1. 获取上下文
    profile = state.get("pet_profile")
    if not profile:
        logger.error("PetProfile is empty. Cannot generate queries.")
        return {"search_queries": []}

    # 2. 准备 Prompt 数据
    # 处理 Enum (如果 species 是 Enum 对象则取 value，否则转字符串)
    species_val = getattr(profile.species, "value", str(profile.species))

    # 3. 构造 System Prompt (精简版：只保留 Strategies 1 & 2)
    system_prompt = f"""
    # Role
    You are a Search Query Specialist for a Veterinary RAG System.
    
    # Input Data (Pet Profile)
    - Species: {species_val}
    - Breed: {profile.breed}
    - Age: {profile.age}
    - Symptoms: {profile.symptoms}
    
    # Task
    Generate 2 specific types of search queries to ensure we find relevant medical records.
    
    ## Strategy 1: Simulated Observation (For Vector Semantic Match)
    - **Goal**: Match the "Owner Observation" field in our database (which is often casual and descriptive).
    - **Instruction**: Reconstruct a likely user complaint based on the profile. Use "I" or "My pet". Include the symptoms naturally.
    - **Language**: English (Must Translate if input is Chinese).
    
    ## Strategy 2: Medical Expansion (For Keyword/BM25 Match)
    - **Goal**: Match professional veterinary case descriptions and cover synonyms.
    - **Instruction**: Convert the casual symptoms into standard veterinary terms.
    - **Example**: "Throwing up" -> "Emesis" or "Vomiting"; "No energy" -> "Lethargy".
    - **Constraint**: Keep the Species constraint (e.g., Feline, Canine) but DO NOT remove the Breed unless it's unknown.
    - **Language**: English (Must Translate if input is Chinese).
    
    # Output Format
    Strictly output the JSON object defined by the schema.
    """

    # 4. 绑定结构化输出
    # 这一步会自动强制 LLM 输出符合 MultiViewSearchQueries 定义的 JSON
    query_generator = orchestrator_model.with_structured_output(MultiViewSearchQueries)

    try:
        # 5. 调用 LLM
        result: MultiViewSearchQueries = query_generator.invoke(
            [SystemMessage(content=system_prompt)]
        )

        # 6. 提取结果
        generated_queries = [result.simulated_observation, result.medical_expansion]

        logger.info(f"Generated Dual-View Queries: {generated_queries}")

        # 7. 更新 State
        return {"search_queries": generated_queries, "query_rewrite_count": 0}

    except Exception as e:
        logger.error(f"Query Generation Failed: {e}")
        # 兜底策略：简单拼接
        fallback_query = f"{species_val} {', '.join(profile.symptoms)}"
        return {"search_queries": [fallback_query], "query_rewrite_count": 0}
