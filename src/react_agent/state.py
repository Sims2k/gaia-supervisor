"""Define the state structures for the agent supervisor."""

from __future__ import annotations

from typing import List, Literal, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState, add_messages
from typing_extensions import TypedDict, Annotated


# --- Constants and shared definitions ---------------------------------------

# Define team members - these are the specialized workers our supervisor manages
MEMBERS = ["researcher", "coder"]

# All routing options (members + finishing)
OPTIONS = MEMBERS + ["FINISH"]


# --- Router for supervisor decisions ---------------------------------------

class Router(TypedDict):
    """Determines which worker to route to next or if the task is complete.
    
    The supervisor returns this structure to navigate the workflow.
    Valid values are defined in the OPTIONS list: researcher, coder, FINISH
    """
    next: Literal[*OPTIONS]


# --- State for the agent supervisor ----------------------------------------

class State(MessagesState):
    """State for the agent supervisor workflow.
    
    Extends MessagesState which provides message history tracking.
    Adds 'next' to track routing information between steps.
    """
    next: str
