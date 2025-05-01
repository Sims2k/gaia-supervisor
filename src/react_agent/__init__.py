"""Agent Supervisor.

This module defines a supervisor agent that delegates tasks to specialized worker agents:
- Researcher: Specialized in web search and finding information
- Coder: Specialized in writing and executing Python code
"""

from react_agent.graph import graph

__all__ = ["graph"]
