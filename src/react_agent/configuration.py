"""Define the configurable parameters for the agent supervisor system."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated

from langchain_core.runnables import ensure_config
from langgraph.config import get_config

from react_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent supervisor system."""

    # Supervisor configuration
    supervisor_prompt: str = field(
        default=prompts.SUPERVISOR_PROMPT,
        metadata={
            "description": "The system prompt for the supervisor agent. "
            "This prompt guides how the supervisor delegates tasks to worker agents."
        },
    )
    
    # Planner configuration
    planner_prompt: str = field(
        default=prompts.PLANNER_PROMPT,
        metadata={
            "description": "The system prompt for the planner agent. "
            "This prompt guides how the planner creates structured plans."
        },
    )
    
    # Critic configuration
    critic_prompt: str = field(
        default=prompts.CRITIC_PROMPT,
        metadata={
            "description": "The system prompt for the critic agent. "
            "This prompt guides how the critic evaluates answers."
        },
    )
    
    # Worker agents configuration
    researcher_prompt: str = field(
        default=prompts.RESEARCHER_PROMPT,
        metadata={
            "description": "The system prompt for the researcher agent. "
            "This prompt defines the researcher's capabilities and limitations."
        },
    )
    
    coder_prompt: str = field(
        default=prompts.CODER_PROMPT,
        metadata={
            "description": "The system prompt for the coder agent. "
            "This prompt defines the coder's capabilities and approach to programming tasks."
        },
    )
    
    # Shared configuration
    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "Legacy system prompt for backward compatibility. "
            "This prompt is used when running the agent in non-supervisor mode."
        },
    )

    # LLM Configuration
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o-mini", # Switched to OpenAI for now
        # default="google_genai/gemini-2.0-flash", # Keep Gemini as an option
        metadata={
            "description": "The large language model used by the agents (provider/model_name)."
        },
    )
    
    # Planner model (lightweight reasoning model)
    planner_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o-mini", # Can be replaced with a cheaper model
        metadata={
            "description": "The lightweight reasoning model used by the planner (provider/model_name)."
        },
    )

    # Tool Configuration
    max_search_results: int = field(
        default=5,
        metadata={
            "description": "The maximum number of search results to return."
        },
    )

    @classmethod
    def from_context(cls) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        try:
            config = get_config()
        except RuntimeError:
            config = None
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
