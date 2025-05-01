# GAIA Supervisor

An Agent Supervisor architecture built with LangGraph and LangChain that routes tasks to specialized worker agents.

## Overview

This project implements an Agent Supervisor architecture that:

- Uses a supervisor LLM to analyze user queries and route them to specialized agents
- Includes specialized worker agents:
  - **Researcher**: Uses web search to find information online
  - **Coder**: Executes Python code to solve programming tasks
- Built with LangGraph for agent orchestration
- Compatible with multiple LLM providers (OpenAI, Google Gemini, etc.)

## Features

- 🧠 Smart task routing based on query type
- 🔄 Asynchronous state management
- 🔍 Web search capabilities
- 💻 Python code execution
- 📝 Persistent conversation history

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/gaia-supervisor.git
cd gaia-supervisor

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Usage

```python
from react_agent import create_agent

# Create the agent
agent = create_agent()

# Run the agent
result = agent.invoke({"messages": [{"role": "user", "content": "YOUR_QUERY"}]})
```

## Architecture

The system uses a graph-based architecture:
- Supervisor node: Routes tasks to appropriate workers
- Researcher node: Handles web search tasks
- Coder node: Handles Python code execution

## License

MIT