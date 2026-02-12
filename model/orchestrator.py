import os
from langchain_deepseek import ChatDeepSeek

orchestrator_model = ChatDeepSeek(
    model="deepseek-chat", 
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0
)