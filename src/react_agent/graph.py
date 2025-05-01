"""Define an Agent Supervisor graph with specialized worker agents.

The supervisor routes tasks to specialized agents based on the query type.
"""

from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from react_agent.configuration import Configuration
from react_agent.state import MEMBERS, OPTIONS, State, Router
from react_agent.tools import TOOLS, tavily_tool, python_repl_tool
from react_agent.utils import load_chat_model, format_system_prompt
from react_agent import prompts


# --- Supervisor node ------------------------------------------------------

def supervisor_node(state: State) -> Command[Literal[*MEMBERS, "__end__"]]:
    """Supervising LLM that decides which specialized agent should act next.

    Args:
        state: The current state with messages

    Returns:
        Command with routing information
    """
    configuration = Configuration.from_context()
    supervisor_llm = load_chat_model(configuration.model)
    
    # Build messages with raw dict format (matching notebook)
    messages = [
        {"role": "system", "content": format_system_prompt(prompts.SUPERVISOR_PROMPT)},
    ] + state["messages"]
    
    # Get structured output from the supervisor model
    response = supervisor_llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    
    # Convert FINISH to END for graph routing
    if goto == "FINISH":
        goto = END
    
    return Command(goto=goto, update={"next": goto})


# --- Worker agents --------------------------------------------------------

# Create the research agent (matches notebook pattern)
def create_researcher_node():
    """Creates the researcher node function that uses web search.

    Returns:
        A function that processes research requests
    """
    configuration = Configuration.from_context()
    llm = load_chat_model(configuration.model)
    
    # Create the agent exactly like in the notebook
    research_agent = create_react_agent(
        llm, 
        tools=[tavily_tool], 
        prompt=format_system_prompt(prompts.RESEARCHER_PROMPT)
    )
    
    # Define node function (matches notebook pattern)
    def researcher_node(state: State) -> Command[Literal["supervisor"]]:
        """Process research queries using web search.
        
        Args:
            state: The current conversation state
            
        Returns:
            Command to return to supervisor with results
        """
        result = research_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="researcher")
                ]
            },
            goto="supervisor",
        )
    
    return researcher_node


# Create the coder agent (matches notebook pattern)
def create_coder_node():
    """Creates the coder node function that executes Python code.
    
    Returns:
        A function that processes coding requests
    """
    configuration = Configuration.from_context()
    llm = load_chat_model(configuration.model)
    
    # Create the agent exactly like in the notebook
    code_agent = create_react_agent(
        llm, 
        tools=[python_repl_tool],
        prompt=format_system_prompt(prompts.CODER_PROMPT)
    )
    
    # Define node function (matches notebook pattern)
    def coder_node(state: State) -> Command[Literal["supervisor"]]:
        """Process coding tasks using Python REPL.
        
        Args:
            state: The current conversation state
            
        Returns:
            Command to return to supervisor with results
        """
        result = code_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="coder")
                ]
            },
            goto="supervisor",
        )
    
    return coder_node


# --- Graph assembly -------------------------------------------------------

def create_agent_supervisor_graph() -> StateGraph:
    """Create the agent supervisor graph with all nodes and edges.

    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize the graph with our State type
    builder = StateGraph(State)
    
    # Start the graph at the supervisor (like the notebook)
    builder.add_edge(START, "supervisor")
    
    # Add the supervisor node
    builder.add_node("supervisor", supervisor_node)
    
    # Add worker nodes (matching notebook pattern)
    builder.add_node("researcher", create_researcher_node())
    builder.add_node("coder", create_coder_node())
    
    # Compile the graph
    return builder.compile(name="Agent Supervisor")


# Initialize the graph (makes it available for import)
graph = create_agent_supervisor_graph()
