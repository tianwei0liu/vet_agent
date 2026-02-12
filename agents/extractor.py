import json
from typing import Optional
from pydantic import BaseModel, field_validator, ValidationError
from state.diagnosis import DiagnosisState, PetProfile
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from model.extractor import extractor_model

import logging

logger = logging.getLogger(__name__)


def merge_profiles(current: PetProfile, new_delta: PetProfile) -> PetProfile:
    """
    负责将新提取的信息 (Delta) 合并进现有档案 (Current)
    适配 Pydantic V2 写法
    """
    updated_data = current.model_dump()
    delta_data = new_delta.model_dump(exclude_none=True)

    for key, new_value in delta_data.items():
        # 特殊处理 Symptoms: 采用追加模式
        if key == "symptoms":
            if updated_data.get(key) is None:
                updated_data[key] = []

            # [防御] 确保 new_value 是列表，防止 None 进来遍历报错
            if isinstance(new_value, list):
                for item in new_value:
                    if not isinstance(item, str):
                        continue

                    # [清洗] 去除空格，保证匹配准确
                    clean_item = item.strip()

                    # [去重] 只有不存在时才追加
                    if clean_item and clean_item not in updated_data[key]:
                        updated_data[key].append(clean_item)

        # 其他字段: 覆盖模式
        else:
            updated_data[key] = new_value

    return PetProfile(**updated_data)


def extractor_node(state: DiagnosisState):
    """
    Inquiry Node: Delta Extraction + Python Merging
    """
    logger.info("entering extractor_node")
    messages = state["messages"]
    current_profile = state.get("pet_profile", PetProfile())
    turns = state.get("inquiry_turns", 0)

    last_user_msg = messages[-1].content
    last_bot_question = "Please describe your pet's health issue or situation."
    if len(messages) > 1:
        last_bot_question = messages[-2].content
    
    current_state_json = current_profile.model_dump_json(indent=2)
    # --- Step 1: 提取增量信息 (Delta Extraction) ---
    # Prompt 强调只提取“本次”对话中的新信息
    extraction_prompt = f"""
    ### Role & Objective
    You are a **Veterinary Data Extractor**.

    ### Global Language Protocol (CRITICAL)
    1. **General Rule:** You must translate **ALL** extracted values (Species, Breed, Symptoms, age, sex, weight) into **ENGLISH**, even if the user speaks Chinese, Spanish, etc.
    2. **The EXCEPTION:** The **Pet's Name** must remain in its **ORIGINAL LANGUAGE**.
       - *Example:* "我的狗叫大黄" -> Name: "大黄" (Keep), Species: "dog" (Translate).

    ### Input Data Streams
    
    <context_state>
        <last_agent_question>{last_bot_question}</last_agent_question>
        <current_profile_snapshot>
        {current_state_json}
        </current_profile_snapshot>
    </context_state>

    <user_latest_input>
        "{last_user_msg}"
    </user_latest_input>

    ### Extraction Logic & Scope
    
    #### 1. The "Delta" Rule (CRITICAL)
    * **Source of Truth:** You must extract information **EXCLUSIVELY** from the `<user_latest_input>` tag.
    * **Purpose of Context:** Use `<context_state>` **ONLY** to resolve ambiguity (e.g., coreference resolution).
        * *Example:* If Input is "Yes", and Last Question is "Is he vomiting?", extract "vomiting".
        * *Example:* If Input is "No", and Last Question is "Is he vomiting?", do NOT extract.
    * **Prohibition:** NEVER re-extract data that exists *only* in `<context_state>`. If the user didn't say it *now*, do not output it.

    #### 2. Name Analysis
    * **Constraint:** Distinguish Proper Names from Species/Breeds.
    * **Action:** If `<user_latest_input>` contains a proper name (e.g., "Charlie"), extract it. If it only contains "My dog", set `name` to `null`.
    
    #### 3. Species & Breed Logic
    * **Allowed Species:** ["dog", "cat", "rabbit", "hamster", "guinea_pig", "bird", "ferret", "unknown"].
    * **Inference:** If user implies species (e.g., "My puppy"), infer "dog". 
    * **Consistency:** If species is already in `<current_profile_snapshot>`, do not change it unless the user explicitly corrects it.

    * **Breed Strategy (Precision Handling):**
       - **Case 1: Specific Mention**
         - IF user mentions a breed (e.g., "Golden Retriever") -> **Extract it**.

       - **Case 2: Explicit "Unknown" / "Mixed" (Stop the Loop)**
         - IF user explicitly says "I don't know", "It's a stray", "Mixed", "Mutts", or "Just a normal cat" -> **Action:** Fill `breed` with the value of `species` (e.g., "dog" or "cat").
         - *Reasoning:* This signals the system that the "breed question" is answered and should not be asked again.

       - **Case 3: Not Mentioned (Keep Asking)**
         - IF user has NOT mentioned the breed at all (e.g., "My dog is vomiting") -> **Action:** Leave `breed` as `null`.
         - *Reasoning:* This allows the Inquiry Agent to ask "What breed is your dog?" in the next turn.

    #### 4. Symptom Normalization
    * Translate descriptions to English.
    * Extract the user's raw description of symptoms/behaviors from `<user_latest_input>`.

    ### Output Format
    Return a strictly valid JSON object matching the `PetProfile` schema.
    """

    extractor = extractor_model.with_structured_output(PetProfile)
    # 这里我们只把 Prompt 发给 LLM，不发之前的 Profile，防止它幻觉
    delta_profile = extractor.invoke([SystemMessage(content=extraction_prompt)])
    # --- Step 2: 在 Python 侧进行合并 (Safe Merging) ---
    if delta_profile is None:
        logger.warning("Extractor returned None. Skipping update to prevent crash.")
        return {"pet_profile": current_profile}
    updated_profile = merge_profiles(current_profile, delta_profile)
    logger.info(f"updated_profile summarize:\n{updated_profile.summarization}\n")
    return {"pet_profile": updated_profile}
