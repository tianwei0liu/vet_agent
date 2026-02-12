from state.orchestrator import OrchestratorState
from state.agent_status import AgentStatus
from state.user_intent import IntentType, UserIntentOutput
from model.orchestrator import orchestrator_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

import logging
logger = logging.getLogger(__name__)

def orchestrator_node(state: OrchestratorState):
    """
    main agent：intent recognition
    """
    agent_status = "agent_status"
    user_intent = "user_intent"
    
    status = state.get(agent_status, "")
    if status == AgentStatus.INQUIRY:
        return {}
    elif status == AgentStatus.DIAGNOSIS:
        return {}
    elif status == AgentStatus.TREATMENT:
        return {}
    elif status == AgentStatus.KNOWLEDGE:
        return {}
    
    messages = state["messages"]

    # intent recognition
    last_user_message = messages[-1].content
    
    # structured LLM response for intent recognition
    structured_llm = orchestrator_model.with_structured_output(UserIntentOutput)
    system_prompt = """
    # Role Definition
    You are the **Master Orchestrator** for a professional Online Pet Agent. 
    Your SOLE responsibility is to analyze the user's input and route them to the correct sub-workflow.
    You function as a deterministic API router. Do NOT answer the user's question directly.

    # Context
    The user input will be in **Chinese (Simplified/Traditional)** or **English**, or a mix of both. 
    Regardless of the input language, your logic reasoning and output format must remain in **English**.

    # Intent Categories (Routing Table)
    Classify the user's input into exactly one of the following categories. 
    **CRITICAL:** If the user input is a generic greeting (e.g., "Hi", "你好") or irrelevant to pets, use the fallback intents defined below.

    1. **`{IntentType.INTENT_DIAGNOSIS}`**
    - **Definition:** User describes specific symptoms, behavioral changes, or abnormal conditions. The goal is to figure out "What is wrong?".
    - **Key Signals:** "vomiting", "blood", "limping", "not eating", "sick", "hair loss", "吐了", "拉稀", "没精神".
    - **Priority Rule:** If the user mentions symptoms AND asks for medicine/treatment simultaneously (e.g., "My cat is vomiting, what medicine should I buy?"), classify as **DIAGNOSIS** first, as we need to diagnose before treating.

    2. **`{IntentType.INTENT_TREATMENT}`**
    - **Definition:** User explicitly asks about medication, post-surgery care, or management of a *known/diagnosed* condition. They are asking "How to fix/manage it?".
    - **Key Signals:** "dosage", "side effects", "post-op care", "vaccine schedule", "insulin", "药量", "疫苗", "术后护理".
    - **Exclusion:** If the user mentions new, undiagnosed symptoms, fallback to DIAGNOSIS.

    3. **`{IntentType.INTENT_KNOWLEDGE}`**
    - **Definition:** General educational questions about breeds, diet, habits, or raising pets. No urgent medical crisis is implied.
    - **Key Signals:** "best food for puppy", "how to train", "breed characteristics", "lifespan", "什么猫粮好", "训练", "品种".

    4. **`{IntentType.CHIT_CHAT}`** (Fallback)
    - **Definition:** Pure greetings, phatic expressions, or checking if the bot is alive.
    - **Key Signals:** "Hi", "Hello", "Are you there?", "你好", "在吗".

    5. **`{IntentType.OUT_OF_SCOPE}`** (Fallback)
    - **Definition:** Queries unrelated to pets or veterinary medicine (e.g., coding, politics, weather, human medical advice).

    # Decision Logic (Chain of Thought)
    Before determining the final intent, you must perform the following analysis step-by-step:
    1. **Detect Language:** Identify the user's input language.
    2. **Entity Analysis:** Identify if a pet or a specific symptom is mentioned.
    3. **Intent Disambiguation:** - Distinguish between "Buying food" (Customer Service/Out of scope) vs. "Eating food causes vomiting" (Diagnosis).
    - Distinguish between "My dog has fleas" (Diagnosis) vs. "What is the best flea medicine?" (Treatment).
    4. **Final Routing:** Select the most appropriate Category ID.

    # Output Format
    You must output a valid **JSON object** wrapped in markdown code blocks. NO other text is allowed.

    **JSON Schema:**
    ```json
    {
    "thought_process": "Brief explanation of why this intent was chosen (in English). Be specific about the detected signals.",
    "detected_language": "CN" or "EN",
    "confidence_score": float (0.0 to 1.0),
    "intent": "String (Must be one of the Category IDs above)"
    }
    """
    
    response = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_user_message)
    ])
    
    intent = response.intent
    
    # 如果意图是诊断，我们初始化追问流程
    if intent == IntentType.INTENT_DIAGNOSIS:
        return {
            agent_status: AgentStatus.INQUIRY,
            user_intent: IntentType.INTENT_DIAGNOSIS,
        }
    # 其他意图，简单回复并结束 (根据你的需求，这里暂时 End)
    elif intent == IntentType.INTENT_TREATMENT:
        return {agent_status: AgentStatus.TREATMENT, user_intent:IntentType.INTENT_TREATMENT, "messages": [AIMessage(content="[Orchestrator] I see you are asking for treatment. Let's move to the treatment module (Placeholder).")]}
    elif intent == IntentType.INTENT_KNOWLEDGE:
        return {agent_status: AgentStatus.KNOWLEDGE, user_intent:IntentType.INTENT_KNOWLEDGE, "messages": [AIMessage(content="[Orchestrator] I can help answer general questions about pets.")]}
    else:
        return {agent_status: AgentStatus.END, user_intent:IntentType.INTENT_FAILURE}
