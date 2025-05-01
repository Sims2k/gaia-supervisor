"""Utility & helper functions."""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
import asyncio
from datetime import UTC, datetime
from react_agent.state import MEMBERS


# Load environment variables from .env file
load_dotenv()


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def format_system_prompt(prompt_template: str) -> str:
    """Format a system prompt template with current system time and members.
    
    Args:
        prompt_template: The prompt template to format
        
    Returns:
        The formatted prompt with system time and members
    """
    return prompt_template.format(
        system_time=datetime.now(tz=UTC).isoformat(),
        members=MEMBERS
    )


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    
    # Special handling for Google Genai models to ensure they're configured for async
    if provider == "google_genai":
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Make sure we have the API key
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable is required for google_genai models")
        
        return ChatGoogleGenerativeAI(model=model)
    else:
        return init_chat_model(model, model_provider=provider)
