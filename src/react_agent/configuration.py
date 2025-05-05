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

    # LLM Configuration - Default model for backward compatibility
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o-mini",  # Original config
        # default="anthropic/claude-3-7-sonnet-20250219",  # Not available
        # default="anthropic/claude-3-5-sonnet-20240620",  # Revert to 3.5 as it's available
        metadata={
            "description": "The default large language model used by the agents (provider/model_name)."
        },
    )
    
    # Model for the researcher (information gathering) - use powerful model
    researcher_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        # default="openai/gpt-4o-mini",  # Original config
        # default="anthropic/claude-3-7-sonnet-20250219",  # Not available
        default="anthropic/claude-3-5-sonnet-20240620",  # Revert to 3.5 as it's available
        metadata={
            "description": "The model used by the researcher agent for gathering information (provider/model_name)."
        },
    )
    
    # Model for the coder (code execution) - use Claude Sonnet
    coder_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        # default="anthropic/claude-3-7-sonnet-20250219",  # Keep using 3.7 as it's available
        default="anthropic/claude-3-5-sonnet-20240620", 
        metadata={
            "description": "The model used by the coder agent for programming tasks (provider/model_name)."
        },
    )
    
    # Model for lightweight reasoning tasks (planner, supervisor, critic)
    planner_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="google_genai/gemini-1.5-flash",  # Revert to Flash to avoid quota issues
        # default="google_genai/gemini-1.5-pro",  # Hitting rate limits
        metadata={
            "description": "The lightweight reasoning model used by the planner, supervisor, and critic (provider/model_name)."
        },
    )
    
    # Same model used for supervisor and critic (points to planner_model)
    supervisor_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="google_genai/gemini-1.5-flash",  # Revert to Flash to avoid quota issues
        # default="google_genai/gemini-1.5-pro",  # Hitting rate limits
        metadata={
            "description": "The model used by the supervisor for routing (provider/model_name)."
        },
    )
    
    critic_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="google_genai/gemini-1.5-flash",  # Revert to Flash to avoid quota issues
        # default="google_genai/gemini-1.5-pro",  # Hitting rate limits
        metadata={
            "description": "The model used by the critic for evaluation (provider/model_name)."
        },
    )
    
    # Model for final answer generation - using Claude for precise formatting
    final_answer_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o-mini",  # Use GPT-4o-mini instead of Claude to avoid overload issues
        # default="anthropic/claude-3-5-sonnet-20240620",  # Keep using 3.5 as it's available
        # default="anthropic/claude-3-7-sonnet-20250219",  # Not available
        metadata={
            "description": "The model used for generating the final answers in GAIA benchmark format (provider/model_name)."
        },
    )

    # Tool Configuration
    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return."
        },
    )
    
    max_wikipedia_results: int = field(
        default=3,
        metadata={
            "description": "The maximum number of Wikipedia results to return."
        },
    )
    
    max_arxiv_results: int = field(
        default=5,
        metadata={
            "description": "The maximum number of ArXiv paper results to return."
        },
    )
    
    max_youtube_results: int = field(
        default=3,
        metadata={
            "description": "The maximum number of YouTube video results to return."
        },
    )
    
    # Execution Configuration
    recursion_limit: int = field(
        default=50,
        metadata={
            "description": "Maximum number of recursion steps allowed in the LangGraph execution."
        },
    )
    
    max_iterations: int = field(
        default=12,
        metadata={
            "description": "Maximum number of iterations allowed to prevent infinite loops."
        },
    )
    
    allow_agent_to_extract_answers: bool = field(
        default=True,
        metadata={
            "description": "Whether to allow the agent to extract answers from context when formatting fails."
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
