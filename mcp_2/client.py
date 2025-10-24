
import asyncio
from langchain_groq import ChatGroq
from mcp_use import MCPAgent,MCPClient
import os

async def run_memory_chat():
    load_dotenv
    os.environ
    config_file="server/weather.json"
    client=MCPClient.from_config_file(config_file)
    llm=ChatGroq(model="qwen-qwq-32b")
    agent=MCPAgent(llm=llm,)