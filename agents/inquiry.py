import json
from typing import Optional, List
from pydantic import BaseModel, field_validator, ValidationError
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

from state.diagnosis import DiagnosisState
from state.agent_status import AgentStatus

# 确保引入了你移动后的 SpeciesEnum
from common.species_enum import SpeciesEnum

# 如果 state.pet_profile 里还没有引入 SpeciesEnum，请确保先完成上一步的 State 修改
from state.pet_profile import PetProfile
from model.inquiry import inquiry_model

import logging

logger = logging.getLogger(__name__)


def inquiry_node(state: DiagnosisState):
    logger.info("entering inquiry_node")
    agent_status = "agent_status"
    updated_profile = state.get("pet_profile", PetProfile())
    turns = state.get("inquiry_turns", 0)
    additional_turns = state.get("additional_inquiry_turns", 0)

    # --- 修改点 1: 将 species 加入必填项 ---
    mandatory_fields = ["species", "name", "breed", "symptoms"]
    missing_mandatory = []

    for field in mandatory_fields:
        value = getattr(updated_profile, field)

        # --- 修改点 2: 特殊处理 Species 的 Unknown 状态 ---
        # 如果是 None，或者显式为 "unknown"，都视为缺失
        if field == "species":
            if not value or value == SpeciesEnum.UNKNOWN:
                missing_mandatory.append(field)
                continue

        # 其他字段判空逻辑
        if not value:
            missing_mandatory.append(field)

    # 可选字段逻辑保持不变
    optional_fields = ["age", "sex", "weight"]
    missing_optional = []
    for field in optional_fields:
        value = getattr(updated_profile, field)
        if not value:
            missing_optional.append(field)

    # --- 结束条件判断 ---
    if not missing_mandatory:
        if not missing_optional or additional_turns > 0 or turns >= 3:

            summary = updated_profile.summarization
            return {
                agent_status: AgentStatus.DIAGNOSIS,
                "pet_profile": updated_profile,
                "inquiry_turns": turns + 1,
                "messages": [
                    AIMessage(
                        content=f"Got it. Gathering info complete.\n\n{summary}\nProceeding to diagnosis..."
                    )
                ],
            }

    if turns >= 3:
        return {
            agent_status: AgentStatus.END,
            "pet_profile": updated_profile,
            "inquiry_turns": turns + 1,
            "messages": [],
        }

    schema_json = json.dumps(
        PetProfile.model_json_schema(), indent=2, ensure_ascii=False
    )
    current_state_json = updated_profile.model_dump_json(indent=2)

    # 1. Format History (The Context)
    # Convert message objects to a readable string format
    # Limit to last 6 messages to ensure we fit in context (though 5 turns is small enough)
    role_map = {"human": "User", "ai": "Assistant"}

    if state["messages"]:
        history_str = "\n".join([
            f"{role_map.get(msg.type, msg.type)}: {msg.content}" 
            for msg in state["messages"][-6:]
        ])
    else:
        history_str = "(No conversation history)"

    # 2. Explicitly List Targets (The Objective)
    # Why make the LLM "Gap Analyze" again? You already calculated missing_mandatory/optional above!
    target_fields = missing_mandatory if missing_mandatory else missing_optional
    target_fields_str = ", ".join(target_fields)

    # --- 修改点 3: 升级 System Prompt ---
    # 明确指示 LLM 关注 'unknown' 的 species
    ask_prompt = f"""
    ### Role
    You are an empathetic and professional Veterinary Triage Assistant. 
    Your goal is to collect specific missing information to complete a health profile.

    ### Context Data
    
    <conversation_history>
    {history_str}
    </conversation_history>

    <current_profile_snapshot>
    {current_state_json}
    </current_profile_snapshot>

    <task_objective>
    The user has NOT provided the following required fields: [{target_fields_str}]
    </task_objective>

    ### Instructions

    1. **Goal**: Ask the user for the information listed in `<task_objective>`.
        - If multiple fields are missing, prioritize: **Symptoms > Species > Name > Breed**.
        - Combine questions naturally (e.g., "What is your dog's name and breed?" instead of two separate questions), but do not ask more than 2 things at once.

    2. **Symptoms must be specific, not generic**:
        - Treat vague phrases such as "my dog is sick", "not feeling well", "unwell", "something is wrong" as **signals to ask follow‑up questions**, **NOT** as symptoms to record.
        - When symptoms are missing or too vague, ask targeted follow‑up questions to clarify:
          - **What** is happening? (e.g., vomiting, diarrhea, not eating, very tired)
          - **Any changes in eating, drinking, urination, or behavior?**

    3. **Smart Skip (Anti-Looping)**: 
        - Check `<conversation_history>`. If the user has *already* explicitly stated they do not know the answer to a missing field (e.g., "I don't know the breed"), **DO NOT ASK AGAIN**.
        - Instead, move to the next missing field or simply acknowledge it.

    4. **Tone & Empathy**: 
        - If the user described symptoms in the history, briefly acknowledge them (e.g., "I'm sorry to hear your cat is vomiting...") before asking your question.
        - Be concise.

    ### Critical Language Rule
    **Language Mirroring**: Respond in the **exact same language** as the User's last message in `<conversation_history>`.
    """
    question_response = inquiry_model.invoke([SystemMessage(content=ask_prompt)])

    if missing_mandatory:
        return {
            "messages": [question_response],
            "pet_profile": updated_profile,
            "inquiry_turns": turns + 1,
            "additional_inquiry_turns": additional_turns,
        }
    else:
        # ask one additional round for optional fields
        return {
            "messages": [question_response],
            "pet_profile": updated_profile,
            "inquiry_turns": turns + 1,
            "additional_inquiry_turns": additional_turns + 1,
        }
