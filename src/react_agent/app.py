"""Web application for the Agent Supervisor.

This module provides a simple web interface for interacting with the Agent Supervisor.
It uses LangGraph's built-in development server.
"""

import asyncio
import os
from typing import Dict, List

from langchain_core.messages import HumanMessage
from langgraph.dev import server
from langgraph.checkpoint.sqlite import SqliteSaver

from react_agent.graph import graph


async def main():
    """Start the development server for the agent supervisor."""
    # Configure an asynchronous checkpointer
    checkpointer = SqliteSaver.acreate(conn_string=":memory:")
    
    # Configure the server
    configurable = {
        # You can override configuration settings here
        # "model": "google_genai/gemini-2.0-flash",
        # "max_search_results": 5,
    }
    
    config = {"configurable": configurable}
    
    # Define an input handler for processing user queries
    async def handle_input(state, input_message):
        """Convert user input to the correct state format."""
        # Create a new message to add to the state
        new_message = HumanMessage(content=input_message)
        
        # Return the new state with the message added
        return {"messages": [new_message]}
    
    # Start the server with the graph and the async checkpointer
    await server.run(
        graph,
        config,
        checkpointer=checkpointer,
        handle_input=handle_input,
        name="Agent Supervisor",
        description="A supervisor that delegates tasks to specialized agents",
    )


if __name__ == "__main__":
    # No need for blocking flags when using an async checkpointer
    # os.environ["BG_JOB_ISOLATED_LOOPS"] = "true"
    
    asyncio.run(main())


# Example usage from Python (if not running server):
# async def run_graph():
#     async for output in graph.astream(
#         {"messages": [HumanMessage(content="Write a brief research report on the North American beaver.")]},
#         config=config,
#     ):
#         # stream() yields detailed logs of intermediate steps
#         # For now, print final messages
#         if "messages" in output:
#             print(output["messages"][-1])

# if __name__ == "__main__":
#     asyncio.run(run_graph())
