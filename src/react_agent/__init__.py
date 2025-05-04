"""Structured Reasoning Loop Agent Supervisor.

This module defines a structured workflow for complex multi-step tasks:
- Supervisor: Coordinates the overall workflow
- Planner: Creates a structured plan with specific steps
- Worker Agents:
  - Researcher: Specialized in web search and finding information
  - Coder: Specialized in writing and executing Python code
- Critic: Evaluates the final answer for completeness and correctness

The architecture follows a Supervisor → Planner → Workers → Critic loop,
with retry capabilities for more robust and accurate responses.
"""

from react_agent.graph import graph

__all__ = ["graph"]
