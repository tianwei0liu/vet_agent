from enum import Enum
from pydantic import BaseModel, Field, field_validator, ValidationError

class IntentType(str, Enum):
    INTENT_DIAGNOSIS = "intent_diagnosis"
    INTENT_TREATMENT = "intent_treatment"
    INTENT_KNOWLEDGE = "intent_knowledge"
    INTENT_FAILURE = "intent_failure"
    CHIT_CHAT = "chit_chat"
    OUT_OF_SCOPE = "out_of_scope"
        
    
class UserIntentOutput(BaseModel):
    intent: IntentType
    confidence: float = Field(description="0.0 to 1.0 confidence score")