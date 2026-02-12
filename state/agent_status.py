from enum import Enum

class AgentStatus(str, Enum):
    INITIALIZED = "initialized"
    INQUIRY = "inquiry"
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    KNOWLEDGE = "knowledge"
    END = "end"