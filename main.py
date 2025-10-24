
from fastmcp import FastMCP
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
load_dotenv()

mcp=FastMCP(name="Notes App")
@mcp.tool()
def get_my_notes()->str:
    return "no notes"
@mcp.tool()
def add_note(content:str)->str:
    return f"added note: {content}"


if __name__=="__main__":
    mcp.run(transport="http",host="127.0.0.1",port=8000,Middleware=Middleware(CORSMiddleware,allow_orgins=["*"],allow_credentials=True,allow_method=["*"],allow_headers=["*"]))


## uv run main.py in the terminal