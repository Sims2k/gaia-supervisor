"""Supervisor node implementation for the agent supervisor system."""

from typing import Dict, List, Literal, Optional, Union, Type, cast

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from react_agent.configuration import Configuration
from react_agent.state import WORKERS, MEMBERS, ROUTING, VERDICTS, State, Router
from react_agent.utils import load_chat_model, format_system_prompt, get_message_text
from react_agent import prompts


# Compile-time type definitions
SupervisorDestinations = Literal["planner", "critic", "researcher", "coder", "final_answer", "__end__"]


def supervisor_node(state: State) -> Command[SupervisorDestinations]:
    """Supervising LLM that decides which specialized agent should act next.

    Args:
        state: The current state with messages

    Returns:
        Command with routing information
    """
    # Check if we need a plan
    if not state.get("plan"):
        return Command(goto="planner")
    
    # Check if we have a critic verdict that requires replanning
    critic_verdict = state.get("critic_verdict")
    if critic_verdict:
        if critic_verdict.get("verdict") == VERDICTS[0]:  # CORRECT
            # Final answer is approved, navigate to the final_answer node
            # This will generate a polished response before ending
            return Command(
                goto="final_answer",
                update={
                    "messages": [
                        HumanMessage(
                            content="Answer approved by critic. Generating final response.",
                            name="supervisor"
                        )
                    ]
                }
            )
        elif critic_verdict.get("verdict") == VERDICTS[1]:  # RETRY
            # Reset the plan but KEEP the context from previous iterations
            context = state.get("context", {})
            worker_results = state.get("worker_results", {})
            
            return Command(
                goto="planner", 
                update={
                    "plan": None, 
                    "current_step_index": None,
                    "draft_answer": None,
                    "critic_verdict": None,
                    # Keep the context and worker_results
                    "context": context,
                    "worker_results": worker_results,
                    # Add a message about the retry
                    "messages": [
                        HumanMessage(
                            content=f"Retrying with new plan. Reason: {critic_verdict.get('reason', 'Incomplete answer')}",
                            name="supervisor"
                        )
                    ]
                }
            )
    
    # Get the current step from the plan
    plan = state["plan"]
    current_step_index = state.get("current_step_index", 0)
    
    # Check if we've completed all steps
    if current_step_index >= len(plan["steps"]):
        # Use context to compile the draft answer
        context = state.get("context", {})
        
        # Combine the most recent worker outputs as the draft answer
        worker_results = []
        for worker in WORKERS:
            if worker in context:
                worker_results.append(f"**{worker.title()}**: {context[worker]}")
        
        # Compile the draft answer from all worker outputs
        draft_content = "\n\n".join(worker_results)
        
        # Send to the critic for evaluation
        return Command(
            goto="critic",
            update={
                "draft_answer": draft_content,
                # Add a message about moving to evaluation
                "messages": [
                    HumanMessage(
                        content="All steps completed. Evaluating the answer.",
                        name="supervisor"
                    )
                ]
            }
        )
    
    # Get the current step
    current_step = plan["steps"][current_step_index]
    worker = current_step["worker"]
    instruction = current_step["instruction"]
    
    # Extract only the most relevant context for the current worker and task
    context_info = ""
    if state.get("context"):
        # Filter context by relevance to the current task
        relevant_context = {}
        
        # For the coder, extract numerical data and parameters from researcher
        if worker == "coder" and "researcher" in state["context"]:
            relevant_context["researcher"] = state["context"]["researcher"]
        
        # For the researcher, previous coder calculations might be relevant
        if worker == "researcher" and "coder" in state["context"]:
            # Only include numerical results from coder, not code snippets
            coder_content = state["context"]["coder"]
            if len(coder_content) < 100:  # Only short results are likely just numbers
                relevant_context["coder"] = coder_content
        
        # Format the relevant context items
        context_items = []
        for key, value in relevant_context.items():
            # Summarize if value is too long
            if len(value) > 200:
                # Find first sentence or up to 200 chars
                summary = value[:200]
                if '.' in summary:
                    summary = summary.split('.')[0] + '.'
                context_items.append(f"Previous {key} found: {summary}...")
            else:
                context_items.append(f"Previous {key} found: {value}")
        
        if context_items:
            context_info = "\n\nRelevant context: " + "\n".join(context_items)
    
    # Enhance the instruction with context
    enhanced_instruction = f"{instruction}{context_info}"
    
    # Add guidance based on worker type
    if worker == "coder":
        enhanced_instruction += "\nProvide both your calculation method AND the final result value."
    elif worker == "researcher":
        enhanced_instruction += "\nFocus on gathering factual information related to the task."
    
    # Add the instruction to the messages
    messages_update = [
        HumanMessage(
            content=f"Step {current_step_index + 1}: {enhanced_instruction}",
            name="supervisor"
        )
    ]
    
    # Cast worker to appropriate type to satisfy type checking
    worker_destination = cast(SupervisorDestinations, worker)
    
    # Move to the appropriate worker
    return Command(
        goto=worker_destination,
        update={
            "messages": messages_update,
            "next": worker  # For backward compatibility
        }
    ) 