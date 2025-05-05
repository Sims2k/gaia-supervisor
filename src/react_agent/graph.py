"""Define an Agent Supervisor graph with specialized worker agents.

The supervisor routes tasks to specialized agents based on the query type.
"""

from typing import Dict, List, Literal, Optional, Union, Type, cast

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
# Import adjusted for compatibility
from langgraph.prebuilt import create_react_agent  # Try original import path first
from langgraph.types import Command

from react_agent.configuration import Configuration
from react_agent.state import WORKERS, MEMBERS, ROUTING, VERDICTS, State, Router, Plan, PlanStep, CriticVerdict
from react_agent.tools import TOOLS, tavily_tool, python_repl_tool, wikipedia_tool, arxiv_tool, youtube_tool, youtube_transcript_tool, wolfram_alpha_tool
from react_agent.utils import load_chat_model, format_system_prompt, get_message_text
from react_agent import prompts
from react_agent.supervisor_node import supervisor_node


# Compile-time type definitions
SupervisorDestinations = Literal["planner", "critic", "researcher", "coder", "final_answer", "__end__"]
WorkerDestination = Literal["supervisor"]


# Helper function to check if a message is from a user
def is_user_message(message):
    """Check if a message is from a user regardless of message format."""
    if isinstance(message, dict):
        return message.get("role") == "user"
    elif isinstance(message, HumanMessage):
        return True
    return False


# Helper function to get message content
def get_message_content(message):
    """Extract content from a message regardless of format."""
    if isinstance(message, dict):
        return message.get("content", "")
    elif hasattr(message, "content"):
        return message.content
    return ""


# --- Planner node ---------------------------------------------------------

def planner_node(state: State) -> Command[WorkerDestination]:
    """Planning LLM that creates a step-by-step execution plan.

    Args:
        state: The current state with messages

    Returns:
        Command to update the state with a plan
    """
    configuration = Configuration.from_context()
    # Use the specific planner model
    planner_llm = load_chat_model(configuration.planner_model)
    
    # Track steps
    steps_taken = state.get("steps_taken", 0)
    steps_taken += 1
    
    # Get the original user question (the latest user message)
    user_messages = [m for m in state["messages"] if is_user_message(m)]
    original_question = get_message_content(user_messages[-1]) if user_messages else "Help me"
    
    # Create a chat prompt template with proper formatting
    planner_prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompts.PLANNER_PROMPT),
        ("user", "{question}")
    ])
    
    # Format the prompt with the necessary variables
    formatted_messages = planner_prompt_template.format_messages(
        question=original_question,
        system_time=format_system_prompt("{system_time}"),
        workers=", ".join(WORKERS),
        worker_options=", ".join([f'"{w}"' for w in WORKERS]),
        example_worker_1=WORKERS[0] if WORKERS else "researcher",
        example_worker_2=WORKERS[1] if len(WORKERS) > 1 else "coder"
    )

    # Get structured output from the planner model
    plan = planner_llm.with_structured_output(Plan).invoke(formatted_messages)
    
    # Return with updated state
    return Command(
        goto="supervisor",
        update={
            "plan": plan,
            "current_step_index": 0,
            # Add a message to show the plan was created
            "messages": [
                HumanMessage(
                    content=f"Created plan with {len(plan['steps'])} steps",
                    name="planner"
                )
            ],
            "steps_taken": steps_taken
        }
    )


# --- Final Answer node -----------------------------------------------------

def final_answer_node(state: State) -> Command[Literal["__end__"]]:
    """Generate a final answer based on gathered information.

    Args:
        state: The current state with messages and context

    Returns:
        Command with final answer
    """
    configuration = Configuration.from_context()

    # Track steps
    steps_taken = state.get("steps_taken", 0)
    steps_taken += 1
    
    # Check if we've exhausted retries and already have a draft answer
    retry_exhausted = state.get("retry_exhausted", False)
    draft_answer = state.get("draft_answer")
    
    # Variable to store the final answer
    gaia_answer = ""
    
    if retry_exhausted and draft_answer and draft_answer.startswith("FINAL ANSWER:"):
        # If supervisor already provided a properly formatted answer after exhausting retries,
        # use it directly without calling the model again
        import re
        final_answer_match = re.search(r"FINAL ANSWER:\s*(.*?)(?:\n|$)", draft_answer, re.IGNORECASE)
        if final_answer_match:
            gaia_answer = final_answer_match.group(1).strip()
        else:
            gaia_answer = "unknown"
    else:
        # Use the specific final answer model
        final_llm = load_chat_model(configuration.final_answer_model)
        
        # Get the original user question (the latest user message)
        user_messages = [m for m in state["messages"] if is_user_message(m)]
        original_question = get_message_content(user_messages[-1]) if user_messages else "Help me"
        
        # Check if we already have a draft answer from supervisor
        if draft_answer and draft_answer.startswith("FINAL ANSWER:"):
            # If supervisor already provided a properly formatted answer, use it directly
            raw_answer = draft_answer
        else:
            # Get the context and worker results
            context = state.get("context", {})
            worker_results = state.get("worker_results", {})

            # Compose a prompt for the final answer using the GAIA-specific format
            final_prompt = ChatPromptTemplate.from_messages([
                ("system", prompts.FINAL_ANSWER_PROMPT),
                ("user", prompts.FINAL_ANSWER_USER_PROMPT)
            ])
            
            # Format the context information more effectively
            context_list = []
            # First include researcher context as it provides background
            if "researcher" in context:
                context_list.append(f"Research information: {context['researcher']}")
            
            # Then include coder results which are typically calculations
            if "coder" in context:
                context_list.append(f"Calculation results: {context['coder']}")
            
            # Add any other workers
            for worker, content in context.items():
                if worker not in ["researcher", "coder"]:
                    context_list.append(f"{worker.capitalize()}: {content}")
            
            # Get the final answer
            formatted_messages = final_prompt.format_messages(
                question=original_question,
                context="\n\n".join(context_list)
            )
            
            raw_answer = final_llm.invoke(formatted_messages).content
        
        # Extract the answer in GAIA format: "FINAL ANSWER: [x]"
        import re
        gaia_answer = raw_answer
        final_answer_match = re.search(r"FINAL ANSWER:\s*(.*?)(?:\n|$)", raw_answer, re.IGNORECASE)
        if final_answer_match:
            gaia_answer = final_answer_match.group(1).strip()
        
        # Ensure answer is properly formatted - if we don't have a valid answer
        # but have sufficient context, try to extract directly
        if configuration.allow_agent_to_extract_answers and (not gaia_answer or gaia_answer.lower() in ["unknown", "insufficient information"]):
            context = state.get("context", {})
            from react_agent.supervisor_node import extract_best_answer_from_context
            extracted_answer = extract_best_answer_from_context(context)
            if extracted_answer != "unknown":
                gaia_answer = extracted_answer
    
    # Set status to "final_answer_generated" to indicate we're done
    return Command(
        goto=END,
        update={
            "messages": [
                AIMessage(
                    content=f"FINAL ANSWER: {gaia_answer}",
                    name="supervisor"
                )
            ],
            "next": "FINISH",  # Update next to indicate we're done
            "gaia_answer": gaia_answer,  # Store answer in GAIA-compatible format
            "submitted_answer": gaia_answer,  # Store as submitted_answer for GAIA benchmark
            "status": "final_answer_generated",  # Add status to indicate we're complete
            "steps_taken": steps_taken
        }
    )


# --- Critic node ----------------------------------------------------------

def critic_node(state: State) -> Command[Union[WorkerDestination, SupervisorDestinations]]:
    """Critic that evaluates if the answer fully satisfies the request.
    
    Args:
        state: The current state with messages and draft answer
        
    Returns:
        Command with evaluation verdict
    """
    configuration = Configuration.from_context()
    # Use the specific critic model
    critic_llm = load_chat_model(configuration.critic_model)
    
    # Track steps
    steps_taken = state.get("steps_taken", 0)
    steps_taken += 1
    
    # Get the original user question (the latest user message)
    user_messages = [m for m in state["messages"] if is_user_message(m)]
    original_question = get_message_content(user_messages[-1]) if user_messages else "Help me"
    
    # Get the draft answer
    draft_answer = state.get("draft_answer", "No answer provided.")
    
    # Create a chat prompt template with proper formatting
    critic_prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompts.CRITIC_PROMPT),
        ("user", prompts.CRITIC_USER_PROMPT)
    ])
    
    # Format the prompt with the necessary variables
    formatted_messages = critic_prompt_template.format_messages(
        question=original_question,
        answer=draft_answer,
        system_time=format_system_prompt("{system_time}"),
        correct_verdict=VERDICTS[0] if VERDICTS else "CORRECT",
        retry_verdict=VERDICTS[1] if len(VERDICTS) > 1 else "RETRY"
    )

    # Get structured output from the critic model
    verdict = critic_llm.with_structured_output(CriticVerdict).invoke(formatted_messages)
    
    # Add a message about the verdict
    if verdict["verdict"] == VERDICTS[0]:  # CORRECT
        verdict_message = "Answer is complete, accurate, and properly formatted for GAIA."
        goto = "final_answer"  # Go to final answer node if correct
    else:
        verdict_message = f"Answer needs improvement. Reason: {verdict.get('reason', 'Unknown')}"
        goto = "supervisor"
    
    # Return with updated state
    return Command(
        goto=goto,
        update={
            "critic_verdict": verdict,
            "messages": [
                HumanMessage(
                    content=verdict_message,
                    name="critic"
                )
            ],
            "steps_taken": steps_taken
        }
    )


# --- Worker agent factory -------------------------------------------------

def create_worker_node(worker_type: str):
    """Factory function to create a worker node of the specified type.

    Args:
        worker_type: The type of worker to create (must be in WORKERS)

    Returns:
        A function that processes requests for the specified worker type
    """
    if worker_type not in WORKERS:
        raise ValueError(f"Unknown worker type: {worker_type}")
    
    configuration = Configuration.from_context()
    
    # Select the appropriate model for each worker type
    if worker_type == "researcher":
        llm = load_chat_model(configuration.researcher_model)
        worker_prompt = prompts.RESEARCHER_PROMPT
        worker_tools = [tavily_tool, wikipedia_tool, arxiv_tool, youtube_tool, youtube_transcript_tool]
    elif worker_type == "coder":
        llm = load_chat_model(configuration.coder_model)
        worker_prompt = prompts.CODER_PROMPT
        worker_tools = [python_repl_tool, wolfram_alpha_tool]
    else:
        # Default case
        llm = load_chat_model(configuration.model)
        worker_prompt = getattr(prompts, f"{worker_type.upper()}_PROMPT", prompts.SYSTEM_PROMPT)
        worker_tools = TOOLS
    
    # Create the agent
    worker_agent = create_react_agent(
        llm, 
        tools=worker_tools,
        prompt=format_system_prompt(worker_prompt)
    )
    
    # Define node function
    def worker_node(state: State) -> Command[WorkerDestination]:
        """Process requests using the specified worker.
        
        Args:
            state: The current conversation state
            
        Returns:
            Command to return to supervisor with results
        """
        # Track steps
        steps_taken = state.get("steps_taken", 0)
        steps_taken += 1
        
        # Get the last message from the supervisor, which contains our task
        task_message = None
        if state.get("messages"):
            for msg in reversed(state["messages"]):
                if hasattr(msg, "name") and msg.name == "supervisor":
                    task_message = msg
                    break
        
        if not task_message:
            return Command(
                goto="supervisor",
                update={
                    "messages": [
                        HumanMessage(
                            content=f"Error: No task message found for {worker_type}",
                            name=worker_type
                        )
                    ],
                    "steps_taken": steps_taken
                }
            )
        
        # Create a new state with just the relevant messages for this worker
        # This prevents confusion from unrelated parts of the conversation
        agent_input = {
            "messages": [
                # Include the first user message for context
                state["messages"][0] if state["messages"] else HumanMessage(content="Help me"),
                # Include the task message
                task_message
            ]
        }
        
        # Invoke the agent with the clean input
        result = worker_agent.invoke(agent_input)
        
        # Extract the result from the agent response
        result_content = extract_worker_result(worker_type, result, state)
        
        # Store the worker's result in shared context
        context_update = state.get("context", {}).copy()
        context_update[worker_type] = result_content
        
        # Store in worker_results history
        worker_results = state.get("worker_results", {}).copy()
        if worker_type not in worker_results:
            worker_results[worker_type] = []
        worker_results[worker_type].append(result_content)
        
        # Increment the step index after worker completes
        current_step_index = state.get("current_step_index", 0)
        
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result_content, name=worker_type)
                ],
                "current_step_index": current_step_index + 1,
                "context": context_update,
                "worker_results": worker_results,
                "steps_taken": steps_taken
            },
            goto="supervisor",
        )
    
    return worker_node


def extract_worker_result(worker_type: str, result: dict, state: State) -> str:
    """Extract a clean, useful result from the worker's output.
    
    This handles different response formats from different worker types.

    Args:
        worker_type: The type of worker (researcher or coder)
        result: The raw result from the worker agent
        state: The current state for context

    Returns:
        A cleaned string with the relevant result information
    """
    # Handle empty results
    if not result or "messages" not in result or not result["messages"]:
        return f"No output from {worker_type}"
    
    # Get the last message from the agent
    last_message = result["messages"][-1]
    
    # Default to extracting content directly
    if hasattr(last_message, "content") and last_message.content:
        result_content = last_message.content
    else:
        result_content = f"No content from {worker_type}"
    
    # Special handling based on worker type
    if worker_type == "coder":
        # For coder outputs, extract the actual result values from code execution
        if "```" in result_content:
            # Try to extract stdout from code execution
            import re
            stdout_match = re.search(r"Stdout:\s*(.*?)(?:\n\n|$)", result_content, re.DOTALL)
            if stdout_match:
                # Extract the actual execution output, not just the code
                execution_result = stdout_match.group(1).strip()
                if execution_result:
                    # Check if this is just a simple number result
                    if re.match(r"^\d+(\.\d+)?$", execution_result):
                        return execution_result
                    else:
                        return f"Code executed with result: {execution_result}"
            
            # If we couldn't find stdout, try to extract output in a different way
            # Look for "Result:" or similar indicators
            result_match = re.search(r"(?:Result|Output|Answer):\s*(.*?)(?:\n\n|$)", result_content, re.DOTALL)
            if result_match:
                return result_match.group(1).strip()
    
    elif worker_type == "researcher":
        # For researcher outputs, keep the full detailed response
        # but ensure it's well-formatted
        if len(result_content) > 800:
            # If too long, try to extract key sections
            # Look for summary or conclusion sections
            import re
            summary_match = re.search(r"(?:Summary|Conclusion|To summarize|In summary):(.*?)(?:\n\n|$)", 
                                      result_content, re.IGNORECASE | re.DOTALL)
            if summary_match:
                return summary_match.group(1).strip()
    
    # If no special handling was triggered, return the content as is
    return result_content


# --- Graph assembly -------------------------------------------------------

def create_agent_supervisor_graph() -> StateGraph:
    """Create the agent supervisor graph with all nodes and edges.

    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize the graph with our State type
    builder = StateGraph(State)
    
    # Add control nodes
    builder.add_node("planner", planner_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("critic", critic_node)
    builder.add_node("final_answer", final_answer_node)

    # Add worker nodes dynamically based on WORKERS list
    for worker_type in WORKERS:
        builder.add_node(worker_type, create_worker_node(worker_type))
    
    # Define the workflow
    builder.add_edge(START, "supervisor")
    builder.add_edge("planner", "supervisor")
    builder.add_edge("critic", "supervisor")
    builder.add_edge("critic", "final_answer")  # Add edge from critic to final_answer
    builder.add_edge("final_answer", END)  # Final answer node goes to END
    builder.add_edge("supervisor", END)  # Allow the supervisor to end the workflow
    
    # Connect all workers to supervisor
    for worker_type in WORKERS:
        builder.add_edge(worker_type, "supervisor")
    
    # Return the builder, not a compiled graph
    # This allows the caller to compile with a checkpointer
    return builder


# --- Graph instantiation (with flexible checkpointing) -----------------------------

def get_compiled_graph(checkpointer=None):
    """Get a compiled graph with optional checkpointer.

    Args:
        checkpointer: Optional checkpointer for persistence

    Returns:
        Compiled StateGraph ready for execution
    """
    # Get configuration
    configuration = Configuration.from_context()
    
    builder = create_agent_supervisor_graph()
    
    # Define termination condition function to prevent loops
    def should_end(state):
        """Determine if the graph should terminate."""
        # End if status is set to final_answer_generated
        if state.get("status") == "final_answer_generated":
            return True
            
        # End if retry_exhausted flag is set and we've gone through final_answer
        if state.get("retry_exhausted") and state.get("gaia_answer"):
            return True
            
        # End if we've hit maximum recursion limit defined by LangGraph
        steps_taken = state.get("steps_taken", 0)
        if steps_taken >= configuration.recursion_limit - 5:  # Leave buffer
            return True
            
        return False
    
    # Define step counter for tracking step count
    def count_steps(state):
        """Count steps to prevent infinite loops."""
        steps_taken = state.get("steps_taken", 0)
        return {"steps_taken": steps_taken + 1}
    
    # Compile the graph (don't use add_state_transform which isn't available)
    if checkpointer:
        graph = builder.compile(
            checkpointer=checkpointer, 
            name="Structured Reasoning Loop"
        )
    else:
        graph = builder.compile(
            name="Structured Reasoning Loop"
        )
    
    # Configure the graph with recursion limit and max iterations
    graph = graph.with_config({
        "recursion_limit": configuration.recursion_limit,
        "max_iterations": configuration.max_iterations
    })
    
    return graph


# Initialize a default non-checkpointed graph (for backward compatibility)
graph = get_compiled_graph()
