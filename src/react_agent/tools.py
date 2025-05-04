"""This module provides tools for the agent supervisor.

It includes:
- Web Search: For general web results using Tavily.
- Python REPL: For executing Python code (Use with caution!).
"""

from typing import Annotated, List, Any, Callable, Optional, cast

# Core Tools & Utilities
from langchain_core.tools import tool

# Experimental Tools (Use with caution)
from langchain_experimental.utilities import PythonREPL

# Use TavilySearchResults from langchain_community like in the notebook
from langchain_community.tools.tavily_search import TavilySearchResults
from react_agent.configuration import Configuration


# Create Tavily tool using configuration from context (more consistent approach)
def create_tavily_tool():
    """Create the Tavily search tool with configuration from context.

    Returns:
        Configured TavilySearchResults tool
    """
    configuration = Configuration.from_context()
    return TavilySearchResults(max_results=configuration.max_search_results)

# Initialize the tool
tavily_tool = create_tavily_tool()


# --- Python REPL Tool ---
# WARNING: Executes arbitrary Python code locally. Be extremely careful
#          about exposing this tool, especially in production environments.
repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute. Use print(...) to see output."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    # Filter out potentially sensitive REPL implementation details
    result_str = f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    return result_str


# --- Tool List ---

# The list of tools available to the agent supervisor.
TOOLS: List[Callable[..., Any]] = [tavily_tool, python_repl_tool]
