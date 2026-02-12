from enum import Enum
from state.agent_status import AgentStatus
from state.pet_profile import PetProfile
from state.user_intent import IntentType
from typing import Annotated, Optional
from langgraph.graph import MessagesState

class OrchestratorState(MessagesState):
    agent_status: Optional[AgentStatus]
    user_intent: Optional[IntentType]
    pet_profile: Optional[PetProfile]