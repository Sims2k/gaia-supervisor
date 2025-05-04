"""Web application for the Agent Supervisor.

This module provides a simple web interface for interacting with the Agent Supervisor.
It uses LangGraph's built-in development server.
"""

import asyncio
import os
from typing import Dict, List
import json
import uuid

from langchain_core.messages import HumanMessage
from langgraph.dev import server
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from react_agent.graph import create_agent_supervisor_graph, graph


async def main():
    """Start the development server for the agent supervisor."""
    # Use a persistent file path instead of in-memory for better debugging
    db_path = "agent_supervisor.sqlite"
    
    # Configure an asynchronous checkpointer with the persistent path
    checkpointer = SqliteSaver.acreate(conn_string=db_path)
    
    # Configure the server
    configurable = {
        # You can override configuration settings here
        "model": "google_genai/gemini-2.0-flash",
        "max_search_results": 5,
    }
    
    config = {"configurable": configurable}
    
    # Define an input handler for processing user queries
    async def handle_input(state, input_message):
        """Convert user input to the correct state format."""
        # Create a thread_id if it doesn't exist
        thread_id = state.get("configurable", {}).get("thread_id", str(uuid.uuid4()))
        
        # Create a new message to add to the state
        new_message = HumanMessage(content=input_message)
        
        # Return the new state with the message added and thread_id for persistence
        return {
            "messages": [new_message],
            "configurable": {"thread_id": thread_id}
        }
    
    # Create a compiled graph with checkpointer
    compiled_graph = create_agent_supervisor_graph()
    compiled_graph = compiled_graph.compile(checkpointer=checkpointer, name="Agent Supervisor")
    
    # Start the server with the compiled graph
    await server.run(
        compiled_graph,
        config,
        handle_input=handle_input,
        name="Agent Supervisor",
        description="A supervisor that delegates tasks to specialized agents",
    )


if __name__ == "__main__":
    # Enable blocking operations in LangGraph
    os.environ["BG_JOB_ISOLATED_LOOPS"] = "true"
    
    asyncio.run(main())


# Example usage from Python (if not running server):
# async def run_graph():
#     # Initialize with a unique thread_id to enable persistence across runs
#     thread_id = "your-unique-thread-id"  # Use a consistent ID to maintain state
#     
#     # Load the checkpointer
#     checkpointer = SqliteSaver.create(conn_string="agent_supervisor.sqlite")
#     
#     # Create a compiled graph with the checkpointer
#     compiled_graph = create_agent_supervisor_graph()
#     compiled_graph = compiled_graph.compile(checkpointer=checkpointer)
#     
#     # Configure the agent with thread_id for persistence
#     config = {"configurable": {"thread_id": thread_id}}
#     
#     # Run the graph with persistence
#     async for output in compiled_graph.astream(
#         {"messages": [HumanMessage(content="Write a brief research report on the North American beaver.")]},
#         config=config,
#     ):
#         if "messages" in output:
#             print(output["messages"][-1])
