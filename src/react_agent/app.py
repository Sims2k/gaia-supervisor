"""Web application for the Agent Supervisor with GAIA benchmark integration.

This module provides a Gradio web interface for interacting with the Agent Supervisor
and evaluating it against the GAIA benchmark.
"""

import os
import json
import uuid
import asyncio
import requests
import pandas as pd
import gradio as gr

from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from react_agent.graph import create_agent_supervisor_graph, get_compiled_graph

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

class GaiaAgent:
    """Agent implementation for the GAIA benchmark using the LangGraph supervisor."""
    
    def __init__(self, model_name=None, checkpointer=None):
        """Initialize the GAIA agent with LangGraph architecture.
        
        Args:
            model_name: Optional model name to override the default
            checkpointer: Optional checkpointer for persistence
        """
        print("Initializing GaiaAgent...")
        
        # Import Configuration class
        from react_agent.configuration import Configuration
        
        # Get configuration
        config = Configuration.from_context()
        default_model = config.model
        
        # If no checkpointer provided, create a default one - using MemorySaver to avoid SQLite thread issues
        if checkpointer is None:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
            print("Using in-memory checkpointer to avoid thread safety issues")

        # Create and compile the graph
        self.graph = get_compiled_graph(checkpointer=checkpointer)
        
        # Configure the agent using values from Configuration
        self.config = {
            "configurable": {
                # Use configuration model or override if provided
                "model": model_name if model_name else default_model,
                # Import specific models for each role from Configuration
                "researcher_model": config.researcher_model,
                "coder_model": config.coder_model,
                "planner_model": config.planner_model,
                "supervisor_model": config.supervisor_model, 
                "critic_model": config.critic_model,
                "final_answer_model": config.final_answer_model,
                # Other settings from Configuration
                "max_search_results": config.max_search_results,
                "recursion_limit": config.recursion_limit,
                "max_iterations": config.max_iterations,
                "allow_agent_to_extract_answers": config.allow_agent_to_extract_answers
            }
        }
        
        print(f"GaiaAgent initialized successfully with model: {self.config['configurable']['model']}")
        
    def __call__(self, question: str) -> str:
        """Process a question and return an answer formatted for GAIA benchmark.
        
        Args:
            question: The GAIA benchmark question
            
        Returns:
            Answer formatted for GAIA benchmark evaluation
        """
        print(f"Agent received question: {question[:100]}...")
        
        # Create a thread_id for this interaction
        thread_id = str(uuid.uuid4())
        self.config["configurable"]["thread_id"] = thread_id
        
        # Import configuration
        from react_agent.configuration import Configuration
        config = Configuration.from_context()
        
        # Add a system prompt to ensure proper GAIA format
        system_prompt = """You are a general AI assistant. Answer the question concisely. 
        YOUR ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. 
        If asked for a number, don't use commas or units like $ or % unless specified. 
        If asked for a string, don't use articles or abbreviations (e.g. for cities), and write digits as plain text unless specified otherwise.
        Focus on brevity and correctness."""
        
        # Create input state with the human message and system prompt
        input_state = {
            "messages": [HumanMessage(content=question)],
            "configurable": {
                "thread_id": thread_id,
                "system_prompt": system_prompt,
                "model": config.model  # Ensure model is also set in the state
            }
        }
        
        # Process the question with our graph
        try:
            # Execute the graph and get the final state
            # Use invoke instead of stream to limit operations
            try:
                final_state = self.graph.invoke(input_state, config=self.config)
            except Exception as e:
                # If we hit recursion error, try again with higher limit
                print(f"Initial invocation failed: {str(e)}")
                # Use double the recursion limit as fallback
                self.config["configurable"]["recursion_limit"] = config.recursion_limit * 2
                final_state = self.graph.invoke(input_state, config=self.config)
            
            # Extract the answer - either from gaia_answer or from the last message
            if "gaia_answer" in final_state:
                answer = final_state["gaia_answer"]
            else:
                messages = final_state.get("messages", [])
                answer = messages[-1].content if messages else "No answer generated."
            
            # Clean the answer to ensure proper GAIA format (remove any FINAL ANSWER prefix)
            if "FINAL ANSWER:" in answer:
                answer = answer.split("FINAL ANSWER:")[1].strip()
                
            print(f"Agent returning answer: {answer[:100]}...")
            return answer
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(error_msg)
            return error_msg


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """Fetches all questions, runs the GaiaAgent on them, submits answers, and displays the results."""
    
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID")  # Get the SPACE_ID for sending link to the code

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent
    try:
        agent = GaiaAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
        
    # In the case of an app running as a hugging Face space, this link points toward your codebase
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from questions endpoint: {e}")
        print(f"Response text: {response.text[:500]}")
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run the Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            answer = agent(question_text)
            # Format answers according to API requirements - use submitted_answer as required
            answers_payload.append({
                "task_id": task_id, 
                "submitted_answer": answer
            })
            results_log.append({
                "Task ID": task_id, 
                "Question": question_text, 
                "Answer": answer,
                "Correct Answer": "Pending submission..."  # Placeholder until we get results
            })
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({
                "Task ID": task_id, 
                "Question": question_text, 
                "Answer": f"AGENT ERROR: {e}",
                "Correct Answer": "Error - not submitted"
            })

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        
        # Update results with correct answers if available
        if "results" in result_data:
            # Map task_id to correct answer
            correct_answers = {}
            for result in result_data.get("results", []):
                task_id = result.get("task_id")
                correct_answer = result.get("correct_answer", "Not provided")
                is_correct = result.get("is_correct", False)
                submitted = result.get("submitted_answer", "")
                
                if task_id:
                    status = "✅ Correct" if is_correct else "❌ Incorrect"
                    correct_answers[task_id] = f"{correct_answer} ({status})"
            
            # Update results_log with correct answers
            for result in results_log:
                task_id = result.get("Task ID")
                if task_id in correct_answers:
                    result["Correct Answer"] = correct_answers[task_id]
        
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# Function to test a single random question
def test_random_question():
    """Fetch a random question from the API and run the agent on it."""
    api_url = DEFAULT_API_URL
    random_question_url = f"{api_url}/random-question"
    
    try:
        # Fetch a random question
        response = requests.get(random_question_url, timeout=15)
        response.raise_for_status()
        question_data = response.json()
        
        if not question_data:
            return "Error: Received empty response from random question endpoint.", None
            
        task_id = question_data.get("task_id")
        question_text = question_data.get("question")
        
        if not task_id or not question_text:
            return "Error: Invalid question format received.", None
            
        # Initialize agent and get answer
        agent = GaiaAgent()
        answer = agent(question_text)
        
        # Attempt to get the correct answer using a test submission
        correct_answer = "Unknown (submit all questions to see correct answers)"
        try:
            test_submit_response = requests.post(
                f"{api_url}/submit", 
                json={
                    "username": "test_user",
                    "agent_code": "test_code",
                    "answers": [{"task_id": task_id, "submitted_answer": answer}]
                },
                timeout=15
            )
            if test_submit_response.status_code == 200:
                submit_data = test_submit_response.json()
                if "results" in submit_data and submit_data["results"]:
                    result = submit_data["results"][0]
                    correct_answer = result.get("correct_answer", "Not provided")
                    is_correct = result.get("is_correct", False)
                    status = "✅ Correct" if is_correct else "❌ Incorrect"
                    correct_answer = f"{correct_answer} ({status})"
        except Exception as e:
            print(f"Error getting correct answer: {e}")
        
        # Return results
        result = {
            "Task ID": task_id,
            "Question": question_text,
            "Answer": answer,
            "Correct Answer": correct_answer
        }
        
        return "Test completed successfully.", result
        
    except Exception as e:
        return f"Error testing random question: {str(e)}", None


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Benchmark Agent Evaluation")
    gr.Markdown(
        """
        **Instructions:**

        1. Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        2. Click 'Run Evaluation & Submit All Answers' to fetch questions, run the agent, submit answers, and see the score.
        3. Alternatively, click 'Test on Random Question' to test the agent on a single random question.

        ---
        **Note:** Running the agent on all questions may take some time. Please be patient while the agent processes all the questions.
        """
    )

    gr.LoginButton()

    with gr.Tabs():
        with gr.TabItem("Full Evaluation"):
            run_button = gr.Button("Run Evaluation & Submit All Answers")
            status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
            results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)
            
            run_button.click(
                fn=run_and_submit_all,
                outputs=[status_output, results_table]
            )
            
        with gr.TabItem("Test Single Question"):
            test_button = gr.Button("Test on Random Question")
            test_status = gr.Textbox(label="Test Status", lines=2, interactive=False)
            test_result = gr.JSON(label="Question and Answer")
            
            test_button.click(
                fn=test_random_question,
                outputs=[test_status, test_result]
            )


if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")  # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:  # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for GAIA Agent Evaluation...")
    demo.launch(debug=True, share=False)
