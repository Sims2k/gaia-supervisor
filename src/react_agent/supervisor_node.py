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
    # Get configuration to use supervisor_model
    configuration = Configuration.from_context()
    
    # Track steps to prevent infinite loops
    steps_taken = state.get("steps_taken", 0)
    steps_taken += 1
    state_updates = {"steps_taken": steps_taken}
    
    # Check if we've hit our step limit
    if steps_taken >= configuration.recursion_limit - 5:  # Buffer of 5 steps
        # Extract the best answer we have from context if possible
        context = state.get("context", {})
        answer = extract_best_answer_from_context(context)
        
        return Command(
            goto="final_answer",
            update={
                "messages": [
                    HumanMessage(
                        content=f"Maximum steps ({steps_taken}) reached. Extracting best answer from available information.",
                        name="supervisor"
                    )
                ],
                "draft_answer": f"FINAL ANSWER: {answer}",
                "retry_exhausted": True,  # Flag to indicate we've exhausted retries
                "steps_taken": steps_taken
            }
        )
    
    # Safety check - prevent infinite loops by forcing termination after too many retry steps
    retry_count = state.get("retry_count", 0)
    max_retries = 2  # Maximum number of allowed retries
    
    if retry_count > max_retries:
        # Extract the best answer we have from context if possible
        context = state.get("context", {})
        answer = extract_best_answer_from_context(context)
        
        return Command(
            goto="final_answer",
            update={
                "messages": [
                    HumanMessage(
                        content=f"Maximum retries ({max_retries}) reached. Extracting best answer from available information.",
                        name="supervisor"
                    )
                ],
                "draft_answer": f"FINAL ANSWER: {answer}",
                "retry_exhausted": True,  # Flag to indicate we've exhausted retries
                "steps_taken": steps_taken
            }
        )
        
    # Check if we need a plan
    if not state.get("plan"):
        return Command(
            goto="planner",
            update={
                **state_updates
            }
        )
    
    # Validate that the plan has at least one step
    plan = state.get("plan")
    if not plan.get("steps") or len(plan.get("steps", [])) == 0:
        # Plan has no steps, go back to planner with explicit instructions
        return Command(
            goto="planner",
            update={
                "messages": [
                    HumanMessage(
                        content="Previous plan had 0 steps. Please create a plan with at least 1 step to solve the user's question.",
                        name="supervisor"
                    )
                ],
                "plan": None,
                **state_updates
            }
        )
    
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
            # IMPORTANT: Get the current retry count BEFORE incrementing
            current_retry_count = state.get("retry_count", 0)
            
            # Check if we're at the maximum allowed retries
            if current_retry_count >= max_retries:
                # Extract best answer and go to final_answer
                context = state.get("context", {})
                answer = extract_best_answer_from_context(context)
                
                return Command(
                    goto="final_answer",
                    update={
                        "messages": [
                            HumanMessage(
                                content=f"Maximum retries ({max_retries}) reached. Proceeding with best available answer.",
                                name="supervisor"
                            )
                        ],
                        "draft_answer": f"FINAL ANSWER: {answer}",
                        "retry_exhausted": True  # Flag to indicate we've exhausted retries
                    }
                )
            
            # Reset the plan but KEEP the context from previous iterations
            context = state.get("context", {})
            worker_results = state.get("worker_results", {})
            
            # Get the critic's reason for rejection, if any
            reason = critic_verdict.get("reason", "")
            if not reason or reason.strip() == "\"":
                reason = "Answer did not meet format requirements"
                
            # Check if this is a formatting issue
            format_issues = [
                "format", "concise", "explanation", "not formatted", 
                "instead of just", "contains explanations", "FINAL ANSWER"
            ]
            is_format_issue = any(issue in reason.lower() for issue in format_issues)
            
            # If we have enough information but the format is wrong, go directly to final answer
            has_sufficient_info = has_sufficient_information(state)
            
            if is_format_issue and has_sufficient_info and current_retry_count >= 0:
                # We have information but formatting is wrong - skip planning and go to final answer
                return Command(
                    goto="final_answer",
                    update={
                        "messages": [
                            HumanMessage(
                                content="We have sufficient information but formatting issues. Generating properly formatted answer.",
                                name="supervisor"
                            )
                        ],
                        "retry_count": current_retry_count + 1  # Still increment retry count
                    }
                )
            
            # Increment the retry counter
            next_retry_count = current_retry_count + 1
            
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
                    # Track retries - IMPORTANT: store the incremented count
                    "retry_count": next_retry_count,
                    # Add a message about the retry (using the INCREMENTED count)
                    "messages": [
                        HumanMessage(
                            content=f"Retrying with new plan (retry #{next_retry_count}). Reason: {reason}",
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
            "next": worker,  # For backward compatibility
            **state_updates
        }
    )

def extract_best_answer_from_context(context):
    """Extract the best available answer from context.
    
    This is a generic function to extract answers from any type of question context.
    It progressively tries different strategies to find a suitable answer.
    
    Args:
        context: The state context containing worker outputs
        
    Returns:
        Best answer found or "unknown" if nothing suitable is found
    """
    answer = "unknown"
    
    # First check if the coder already provided a properly formatted answer
    if "coder" in context:
        coder_content = context["coder"]
        
        # Look for "FINAL ANSWER: X" pattern in the coder output
        import re
        answer_match = re.search(r"FINAL ANSWER:\s*(.*?)(?:\n|$)", coder_content, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
    
    # If no answer in coder output, check researcher content
    if "researcher" in context:
        researcher_content = context["researcher"]
        
        # Look for lists in the researcher content (common pattern)
        import re
        
        # Look for bulleted list items
        list_items = re.findall(r"[-•*]\s+([^:\n]+)", researcher_content)
        if list_items:
            # Format as comma-separated list
            answer = ",".join(item.strip() for item in list_items)
            return answer
            
        # Look for emphasized/bold items which might be key information
        bold_items = re.findall(r"\*\*([^*]+)\*\*", researcher_content)
        if bold_items:
            # Join the important items as a comma-separated list
            processed_items = []
            for item in bold_items:
                # Remove common filler words and clean up the item
                clean_item = re.sub(r'(^|\s)(a|an|the|is|are|was|were|be|been)(\s|$)', ' ', item)
                clean_item = clean_item.strip()
                if clean_item and len(clean_item) < 30:  # Only include reasonably short items
                    processed_items.append(clean_item)
            
            if processed_items:
                answer = ",".join(processed_items)
                return answer
    
    # If we still don't have an answer, try to extract common entities
    combined_content = ""
    for worker_type, content in context.items():
        combined_content += " " + content
    
    # Look for numbers in the content
    import re
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', combined_content)
    if numbers:
        answer = numbers[0]  # Use the first number found
    
    return answer

def has_sufficient_information(state):
    """Determine if we have enough information to generate a final answer.
    
    Args:
        state: The current conversation state
        
    Returns:
        Boolean indicating if we have sufficient information
    """
    context = state.get("context", {})
    
    # If we have both researcher and coder outputs, we likely have enough info
    if "researcher" in context and "coder" in context:
        return True
        
    # If we have a substantial researcher output, that might be enough
    if "researcher" in context and len(context["researcher"]) > 150:
        return True
        
    # If we have any worker output that contains lists or formatted data
    for worker, content in context.items():
        if content and (
            "- " in content or  # Bullet point
            "•" in content or   # Bullet point
            "*" in content or   # Emphasis or bullet
            ":" in content      # Definition or explanation
        ):
            return True
    
    return False 