"""Define the state structures for the agent supervisor."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState, add_messages
from typing_extensions import TypedDict, Annotated


# --- Constants and shared definitions ---------------------------------------

# Define worker types (specialized agents that perform tasks)
WORKERS = ["researcher", "coder"]

# Define all member types (including control nodes)
MEMBERS = WORKERS + ["planner", "critic", "supervisor"]

# Define status/routing options
VERDICTS = ["CORRECT", "RETRY"]
ROUTING = ["FINISH"] + WORKERS
OPTIONS = ROUTING + VERDICTS


# --- Router for supervisor decisions ---------------------------------------

class Router(TypedDict):
    """Determines which worker to route to next or if the task is complete.
    
    The supervisor returns this structure to navigate the workflow.
    Valid values are defined in the ROUTING list.
    """
    next: Literal[*ROUTING]


# --- Plan structure for the Planner node -----------------------------------

class PlanStep(TypedDict):
    """A single step in the plan created by the Planner."""
    worker: Literal[*WORKERS]
    instruction: str


class Plan(TypedDict):
    """The complete plan produced by the Planner node."""
    steps: List[PlanStep]


# --- Critic verdict structure ----------------------------------------------

class CriticVerdict(TypedDict):
    """The verdict from the Critic on whether the answer is satisfactory."""
    verdict: Literal[*VERDICTS]
    reason: Optional[str]


# --- State for the agent supervisor ----------------------------------------

class State(MessagesState):
    """State for the agent supervisor workflow.
    
    Extends MessagesState which provides message history tracking.
    Adds fields to track routing information, plan, and critic verdict.
    """
    next: str
    plan: Optional[Plan] = None
    current_step_index: Optional[int] = None
    draft_answer: Optional[str] = None
    critic_verdict: Optional[CriticVerdict] = None
    context: Dict[str, Any] = {}  # Shared context accessible to all agents
    worker_results: Dict[str, List[str]] = {}  # Store results from each worker
