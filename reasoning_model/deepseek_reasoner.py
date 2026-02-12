import os
from typing import List, Dict, Any, Optional, Union
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
import logging
logger = logging.getLogger(__name__)

class ChatDeepSeekReasoner(ChatDeepSeek):
    """
    A custom ChatModel for DeepSeek R1 (Reasoner).
    
    Problem:
      1. DeepSeek R1 returns a 'reasoning_content' field.
      2. Standard ChatOpenAI discards this field during response parsing.
      3. DeepSeek API throws 400 if you reply to a message but omit its original reasoning.
    
    Solution:
      1. _create_chat_result: Intercept raw response, extract 'reasoning_content', save to additional_kwargs.
      2. _create_message_dicts: Intercept payload creation, reinject 'reasoning_content' from additional_kwargs.
    """

    def _create_chat_result(self, response: Union[dict, Any]) -> ChatResult:
        """
        INGRESS: Capture the 'reasoning_content' from the raw API response 
        before LangChain sanitizes it.
        """
        # 1. Let standard ChatOpenAI process the standard fields (content, tool_calls)
        result = super()._create_chat_result(response)

        # 2. Access the raw choices from the response object (handling both Pydantic and Dict)
        #    The 'response' is the raw object returned by the openai python client.
        if isinstance(response, dict):
            choices = response.get("choices", [])
        else:
            choices = response.choices

        # 3. Iterate through choices and inject reasoning into the LangChain Message
        for i, choice in enumerate(choices):
            # Extract reasoning from the raw choice
            reasoning = None
            if isinstance(choice, dict):
                message = choice.get("message", {})
                reasoning = message.get("reasoning_content")
            else:
                message = choice.message
                # The openai client might hide extra fields, but DeepSeek sends it here
                reasoning = getattr(message, "reasoning_content", None)

            logger.info(f"\n_create_chat_result\nreasoning: {reasoning}\n\n")
            # If found, staple it to the AIMessage so it persists in memory
            if reasoning and i < len(result.generations):
                msg = result.generations[i].message
                if isinstance(msg, AIMessage):
                    msg.additional_kwargs["reasoning_content"] = reasoning

        return result

    def _create_message_dicts(self, messages: List[BaseMessage], stop: Optional[List[str]]) -> List[Dict[str, Any]]:
        """
        EGRESS: Look for 'reasoning_content' in our message history and 
        add it back to the API payload.
        """
        # 1. Let standard ChatOpenAI create the standard payload
        dicts = super()._create_message_dicts(messages, stop)

        # 2. Iterate and reinject the reasoning field if it exists in our history
        for msg, payload in zip(messages, dicts):
            if isinstance(msg, AIMessage):
                reasoning = msg.additional_kwargs.get("reasoning_content")
                logger.info(f"\n_create_message_dicts\nreasoning: {reasoning}\n\n")
                if reasoning:
                    payload["reasoning_content"] = reasoning
        
        return dicts