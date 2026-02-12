import logging
from typing import Any, Dict
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
        logger.info(f"\nbefore_model: Sending {len(state['messages'])} messages to the model. last user message {state['messages'][-1]}\n")
        return None
    
    def after_model(self, state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
        logger.info(f"\nafter_model: Model responded: {state['messages'][-1]}\n")
        return None
