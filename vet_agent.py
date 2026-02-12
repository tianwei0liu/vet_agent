import os
import asyncio
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from workflows.orchestrator import OrchestratorWorkflow

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
loaded = load_dotenv()
print(f"1. .env Loaded Success? {loaded}")

# 2. 打印关键变量
print(f"2. API Key Prefix: {os.environ.get('LANGCHAIN_API_KEY', '')[:8]}...")
print(f"3. Tracing Enabled: {os.environ.get('LANGCHAIN_TRACING_V2')}")
print(f"4. Project Name: {os.environ.get('LANGCHAIN_PROJECT')}")

async def run_interaction_loop_async():
    # 这是一个唯一的会话ID
    thread_id = "user_session_001"
    config = {"configurable": {"thread_id": thread_id}}
    memory = MemorySaver()
    graph = OrchestratorWorkflow().get_runnable(memory)
    
    logger.info("----- Vet Agent -----")
    
    while True:
        loop = asyncio.get_running_loop()
        try:
            user_input = await loop.run_in_executor(None, input, "\nUser >: ")
        except EOFError:
            break

        if user_input.lower() in ["quit", "exit"]:
            break
        
        input_message = HumanMessage(content=user_input)
        
        try:
            async for event in graph.astream(
                {"messages": [input_message]}, 
                config, 
                stream_mode="values"
            ):
                if "messages" in event:
                    last_msg = event["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        logger.info(f"Agent: {last_msg.content}")
        except Exception as e:
            logger.info(f"Error: {e}")
            break

if __name__ == "__main__":
    asyncio.run(run_interaction_loop_async())
