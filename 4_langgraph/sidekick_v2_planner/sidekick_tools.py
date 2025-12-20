from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from dotenv import load_dotenv
import os
import requests
from langchain.agents import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper



load_dotenv(override=True)
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"
serper = GoogleSerperAPIWrapper()

async def playwright_tools():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools(), browser, playwright


def push(text: str):
    """Send a push notification to the user"""
    requests.post(pushover_url, data = {"token": pushover_token, "user": pushover_user, "message": text})
    return "success"

"""
Q. explain what FileManagementToolkit toolkit does?
FileManagementToolkit is a LangChain agent toolkit that gives an LLM (agent) the ability to interact with the local file system in a controlled way in a specific directory.
It bundles together several file-related tools so an agent can read, write, list, move, and delete files while reasoning about them.
"""
def get_file_tools():
    toolkit = FileManagementToolkit(root_dir="sandbox")
    return toolkit.get_tools()


"""
PythonREPLTool is a LangChain tool that lets an agent execute Python code at runtime and see the results. 
It effectively gives the LLM access to a Python REPL (Read–Eval–Print Loop) so it can calculate, transform data, inspect objects, and prototype logic on the fly.

PythonREPLTool allows an agent to:
- Write Python code
- Execute it immediately: Receives a string containing Python code and Executes it using Python’s built-in execution mechanisms (exec)
- Capture the printed output or returned values
- Use the results to inform the next reasoning step
"""
async def other_tools():
    push_tool = Tool(name="send_push_notification", func=push, description="Use this tool when you want to send a push notification")
    file_tools = get_file_tools()

    tool_search =Tool(
        name="search",
        func=serper.run,
        description="Use this tool when you want to get the results of an online web search"
    )

    wikipedia = WikipediaAPIWrapper()
    wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)

    python_repl = PythonREPLTool()
    
    return file_tools + [push_tool, tool_search, python_repl,  wiki_tool]

