

from langchain_core.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import asyncio
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langgraph.prebuilts import *
from langchain.agents import AgentType
from langgraph.graph.messsage import add_message
from langchain.schema import BaseMessage,HumanMessage
from langchain.agents import tool
from dotenv import load_dotenv
from langgraph.graph import StateGraph,Start
from typing import TypedDict,Annotated
load_dotenv()

llm=ChatOpenAI(temperature=0,model_name="gpt-5")

@tool
def calculator(first_num,second_num,operation):
  try:
    if operation=="add":
      return first_num+second_num
    elif operation=="subtract":
      return first_num-second
    elif operation=="multiply":
      return first_num*second_num
    elif operation=="divide":
      return first_num/second_num
  except Exception as e:
    return str(e)

tools=[calculator]
llm_with_tools=llm.bind(functions=tools)
agent=initialize_agent(tools,llm,agent=AgentType.OPENAI_FUNCTIONS,verbose=True)

class ChatState(TypedDict):
  message: Annotated[list[BaseMessage],add_message]

## node
def chat_node(state):
  message=state["message"]
  response=llm_with_tools.invoke(message)
  return {"message":[response]}

tool_node=ToolNode(tools)
graph=StateGraph(tool_node)
graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)
graph.add_edge(start,"chat_node")
graph.add_edge("chat_node","tools")
graph.add_edge("tools","chat_node")
chatbot=graph.compile()
## running the graph
result=chatbot.invoke({"message":[HumanMessage(content='find the modulous of 12345')]})
print(result['messages'][-1].content)

## async code

from langchain_core.tools import DuckDuckGoSearchRun
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langgraph.prebuilts import *
from langchain.agents import AgentType
from langgraph.graph.messsage import add_message
from langchain.schema import BaseMessage,HumanMessage
from langchain.agents import tool
from dotenv import load_dotenv
from langgraph.graph import StateGraph,Start
from typing import TypedDict,Annotated
load_dotenv()

llm=ChatOpenAI(temperature=0,model_name="gpt-5")

@tool
def calculator(first_num,second_num,operation):
  try:
    if operation=="add":
      return first_num+second_num
    elif operation=="subtract":
      return first_num-second
    elif operation=="multiply":
      return first_num*second_num
    elif operation=="divide":
      return first_num/second_num
  except Exception as e:
    return str(e)

class ChatState(TypedDict):
  message: Annotated[list[BaseMessage],add_message]

def build_graph():
tool_node=ToolNode(tools)
graph=StateGraph(tool_node)
graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)
graph.add_edge(start,"chat_node")
graph.add_edge("chat_node","tools")
graph.add_edge("tools","chat_node")
chatbot=graph.compile()
return chatbot
## running the graph
# result=chatbot.invoke({"message":[HumanMessage(content='find the modulous of 12345')]})
# print(result['messages'][-1].content)



async def main():
  chatbot=build_graph()
  result=await chatbot.ainvoke({"message":[HumanMessage(content='find the modulous of 12345')]})
  print(result['messages'][-1].content)

if __name__=="__main__":
  asyncio.run(main())

## mcp Client

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import DuckDuckGoSearchRun
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langgraph.prebuilts import *
from langchain.agents import AgentType
from langgraph.graph.messsage import add_message
from langchain.schema import BaseMessage,HumanMessage
from langchain.agents import tool
from dotenv import load_dotenv
from langchain_core.tools import tool,BaseTool
from langgraph.graph import StateGraph,Start
from typing import TypedDict,Annotated
import threading
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
load_dotenv()
llm=ChatOpenAI(temperature=0,model_name="gpt-5")
## MCP Client for local FAST MCP SERVER

client=MultiServerMCPClient({"arith":{"transport":"stdio",command:"python3","args":["/Users/Sai/Desktop/mcp-math-server/main.py"},"expense":{"transport":"streamable_http","url":"https://splendid-gold-dingo.fastmcp.app/mcp"})

class ChatState(TypedDict):
  message: Annotated[list[BaseMessage],add_message]
async def build_graph():
  tools=await client.get_tools()
  print(tools)
  llm_with_tools=llm.bind_tools(tools)
  ##nodes
  async def chat_node(state):
    message=state["message"]
    response=await llm_with_tools.ainvoke(message)
    return {'message':[response]}
tool_node=ToolNode(tools)
graph=StateGraph(tool_node)
graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)
graph.add_edge(start,"chat_node")
graph.add_edge("chat_node","tools")
graph.add_edge("tools","chat_node")
chatbot=graph.compile()
return chatbot
async def main():
  chatbot=build_graph()
  result=await chatbot.ainvoke({"message":[HumanMessage(content='Add an expense  for the expense')]})
  print(result['messages'][-1].content)

if __name__=="__main__":
  asyncio.run(main())

## langgraph backend

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import DuckDuckGoSearchRun
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langgraph.prebuilts import *
from langchain.agents import AgentType
from langgraph.graph.messsage import add_message
from langchain.schema import BaseMessage,HumanMessage
from langchain.agents import tool
from dotenv import load_dotenv
from langchain_core.tools import tool,BaseTool
from langgraph.graph import StateGraph,Start
from typing import TypedDict,Annotated
import threading
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
load_dotenv()
import json
import requests
llm=ChatOpenAI(temperature=0,model_name="gpt-5")
## MCP Client for local FAST MCP SERVER

search_tool=  DuckDuckGoSearchRun()

@tool
def get_stock_price(symbol):
  url=f"https://www.alphavantage.co/query?function=GLOBAL_QUOTES&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
  r=requests.get(url)
  return r.json()

client=MultiServerMCPClient({"arith":{"transport":"stdio",command:"python3","args":["/Users/Sai/Desktop/mcp-math-server/main.py"},"expense":{"transport":"streamable_http","url":"https://splendid-gold-dingo.fastmcp.app/mcp"})

def load_mcp_tools():
  try:
    return run_async(client.get_tools())
  except Exception
  return []

mcp_tools=load_mcp_tools()
tools=[search_tool,get_stock_price,*mcp_tools]
llm_with_tools=llm.bind_tools(tools) if tools else llm

## state
class ChatState(TypedDict):
  message: Annotated[list[BaseMessage],add_message]

## Nodes

async def chat_node(state):
  """LLM Node that may answer or request a tool call"""
  message=state["message"]
  response=await llm_with_tools.ainvoke(message)
  return {"message":[response]}
tool_node=ToolNode(tools) if tools else llm

## checkpointer

async def _init_checkpointer():
  conn=await aiosqlite.connect("langgraph.db")
  return AsyncSqliteSaver(conn)

## Graph

graph=StateGraph(ChatState,tool_node)
graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)
if tool_node:
  graph.add_node("tools",tool_node)
  graph.add_edge(start,"chat_node")
  graph.add_edge("chat_node","tools")
  graph.add_edge("tools","chat_node")
chatbot=graph.compile()

## Helper

async def _alist_threads():
  all_thread=set()
  async for checkpoint in checkpointer.alist(None):
    all_thread.add(checkpoint.config["configurable"]["thread_id"])
  return all_thread

def retrieve_all_thread():
  return run_async(_alist_threads())

## front end pending

def generate_thread_id():
  return str(uuid.uuid4())
def reset_chat():
  return {"message":[]}
def load_conversation(thread_id):
  state=chatbot.get_state(thread_id)
  return state.values.get("message",[])
def
with st.chat_message("assistant"):
  status_holder={"box":None}
  def ai_only_stream():
    event_queue:queue.Queue=queue.Queue()
    async def run_stream():
      try:
        async for message_chunck,meta_data in chatbot.astream()
      except Exception as e:
        event_queue.put(e)
      finally:
        event_queue.put(None)
    submit_async_task(run_stream())