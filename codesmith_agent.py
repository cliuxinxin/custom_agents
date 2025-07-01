import argparse
import asyncio
import os
import subprocess
import sys
from typing import List, Dict # Explicitly import Dict
from agents import Agent, Runner, enable_verbose_stdout_logging, handoff, function_tool, RunContextWrapper
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents import set_tracing_disabled # Correct import for disabling tracing
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX # Recommended for handoffs

import logging
# json and pypinyin were imported but not used in the provided snippet, removing for simplicity.
# import json
# import pypinyin

# --- Configuration and Setup ---
# Disable tracing as requested
set_tracing_disabled(True)
enable_verbose_stdout_logging()
logging.basicConfig(level=logging.DEBUG) # Set logging level to DEBUG for more details

# Load .env file
load_dotenv()

# Configure OpenRouter
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_base_url = "https://openrouter.ai/api/v1"
# Use a suitable model from OpenRouter
# Check https://openrouter.ai/docs#models for available models and their pricing/capabilities
# deepseek/deepseek-chat-v3-0324 might be free but has limitations, consider alternatives like
# "openai/gpt-4o-mini" or "mistralai/mistral-large-latest" or "google/gemini-pro"
# For this example, we'll use a placeholder, replace with your chosen model:
openrouter_model_name = "deepseek/deepseek-chat-v3-0324:free" # Recommended starting point

if not openrouter_api_key:
    logging.error("OPENROUTER_API_KEY not found in environment variables.")
    sys.exit(1)

# Create OpenAI client, using OpenRouter's info
openai_client = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url=openrouter_base_url
)

# Create OpenAIChatCompletionsModel instance
model = OpenAIChatCompletionsModel(
    model=openrouter_model_name,
    openai_client=openai_client
)

logging.info(f"Using OpenRouter model: {openrouter_model_name}")

# --- Context ---
from dataclasses import dataclass, field

@dataclass
class CodeContext:
    """Context for the code generation workflow."""
    working_dir: str = "./generated_code_run" # Code will be saved here
    # Add conversation history if needed for more complex interactions, but
    # for this simple chain, the handoff input manages the key info.

# --- Tools ---
@function_tool
def write_file(filename: str, content: str, context: CodeContext) -> str:
    """Writes content to a file in the working directory."""
    filepath = os.path.join(context.working_dir, filename)
    try:
        os.makedirs(context.working_dir, exist_ok=True)
        with open(filepath, "w", encoding='utf-8') as f: # Specify encoding
            f.write(content)
        logging.info(f"Successfully wrote code to {filepath}")
        return f"Successfully wrote code to {filepath}"
    except Exception as e:
        logging.error(f"Error writing file {filepath}: {e}")
        return f"Error writing file {filepath}: {e}"

@function_tool
def execute_python_file(filename: str, context: CodeContext) -> str:
    """Executes a Python file in the working directory and returns output/errors."""
    filepath = os.path.join(context.working_dir, filename)
    if not os.path.exists(filepath):
        logging.warning(f"Attempted to execute non-existent file: {filepath}")
        return f"Error: File not found at {filepath}"

    logging.info(f"Executing file: {filepath}")
    try:
        # Simple execution, may need more robust handling for complex scripts or environments
        # Using shell=False is safer if command components are user-provided, but here filename is agent-controlled.
        # cwd sets the current working directory for the subprocess.
        result = subprocess.run(
            [sys.executable, filepath], # Use sys.executable to find the correct python interpreter
            capture_output=True,
            text=True,
            cwd=context.working_dir,
            check=True, # Raise CalledProcessError for non-zero exit code
            encoding='utf-8' # Specify encoding for text capture
        )
        logging.info(f"Execution Successful for {filename}")
        return f"Execution Successful:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except subprocess.CalledProcessError as e:
        logging.error(f"Execution Failed for {filename}: Return Code {e.returncode}")
        return f"Execution Failed:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}\nReturn Code: {e.returncode}"
    except FileNotFoundError:
         logging.error("Python interpreter not found.")
         return f"Execution Failed: Python interpreter not found."
    except Exception as e:
         logging.error(f"Execution Failed with unexpected error for {filename}: {e}")
         return f"Execution Failed with unexpected error: {e}"

# --- Agents ---

# Define Agents in reverse order of dependency for handoffs
# 3. Code Tester Agent
# This agent receives the filename from the Coder and executes it.
code_tester_agent = Agent[CodeContext]( # Specify Context type
    name="Code Tester",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You are a code execution environment for Python scripts.
You receive a single filename as input, which is the Python script to execute.
Your task is to execute the specified Python file using the `execute_python_file` tool.
After execution, report the exact output (STDOUT and STDERR) from the `execute_python_file` tool call to the user.
If execution fails, report the error details provided by the tool.
Do NOT write or modify code. Only execute the provided file.
""",
    tools=[execute_python_file], # Can use execute_python_file tool
    model=model
    # No handoffs - this is the end of the workflow chain
)

# 2. Code Writer Agent
# This agent receives the coding plan from the Planner and writes the code file.
# It then hands off the filename to the Tester.
code_writer_agent = Agent[CodeContext]( # Specify Context type
    name="Code Writer",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You are a Python code writing expert.
You receive a detailed plan or description of a Python script to write.
Your task is to write a complete, correct, and well-commented Python script based *exactly* on the plan you receive.
Name the file `script.py`.
Use the `write_file` tool to save the generated code to `script.py` in the user's working directory.
After successfully writing the file, you MUST handoff to the 'Code Tester' agent, providing the filename `script.py` as the input to the handoff tool.
Do NOT execute the code yourself. Only write it and save it.
""",
    tools=[write_file], # Can use write_file tool
    # Handoff the filename string to the Code Tester
    # *** FIX: Explicitly set on_handoff=None ***
    handoffs=[handoff(code_tester_agent, on_handoff=str, input_type=str)],
    model=model
)

# 1. Code Planner Agent
# This agent receives the initial user request, plans the code, and hands off to the Writer.
code_planner_agent = Agent[CodeContext]( # Specify Context type
    name="Code Planner",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
You are a helpful assistant that plans Python code implementations based on user requests.
Analyze the user's request carefully. Break it down into a clear, detailed plan for writing a single Python script.
The plan should describe the required functionality, steps, and overall structure.
Do NOT write the code yourself. Your output should be a detailed plan for the Code Writer agent.
Once the plan is complete, you MUST handoff to the 'Code Writer' agent using the appropriate tool, providing the plan as the input to the handoff tool.
Ensure the plan is clear enough for the Code Writer to directly implement the request.
""",
    # Handoff the plan (as a string) to the Code Writer
    # *** FIX: Explicitly set on_handoff=None ***
    handoffs=[handoff(code_writer_agent, on_handoff=str, input_type=str)],
    model=model
    # We are omitting guardrails for simplicity as they were not in the user's provided snippet
    # input_guardrails=[initial_request_guardrail], # Example if adding guardrails back
)


# --- Main Workflow Execution ---

async def run_codesmith_workflow(user_request: str):
    """Runs the code generation, writing, and testing workflow."""
    logging.info(f"--- Starting CodeSmith Workflow ---")
    logging.info(f"User Request: {user_request}")

    # Create context
    ctx = CodeContext(working_dir="./generated_code_run")
    # Ensure working directory exists
    os.makedirs(ctx.working_dir, exist_ok=True)
    logging.info(f"Working directory: {ctx.working_dir}")

    try:
        # Start the workflow by running the Planner Agent.
        # The Planner will handoff to Writer, which hands off to Tester automatically
        result = await Runner.run(
            code_planner_agent, # Start with the Planner
            user_request,
            context=ctx,
            # Tracing is already disabled globally via set_tracing_disabled(True)
            # run_config=RunConfig(tracing_disabled=True) # Alternative way per run
        )

        logging.info("\n--- Workflow Completed ---")
        # The final output comes from the last agent in the chain (CodeTesterAgent)
        print(f"Final Output (Execution Result):\n{result.final_output}")

    # We removed the specific GuardrailTripwireTriggered exception handling for simplicity,
    # but you might want to add general exception handling.
    except Exception as e:
        logging.error(f"\n--- Workflow Failed ---")
        print(f"An unexpected error occurred during the workflow: {e}")
        logging.exception("Detailed exception:")


# --- CLI Entry Point ---
if __name__ == "__main__":
    # Example Usage
    # Ensure you have set up your OPENROUTER_API_KEY environment variable.
    # Ensure you have replaced openrouter_model_name with a valid model name.

    # Get input from CLI arguments
    parser = argparse.ArgumentParser(description='CodeSmith Agent')
    parser.add_argument('input', type=str, help='自然语言描述的开发目标')
    args = parser.parse_args()
    user_input = args.input

    # Run the workflow
    asyncio.run(run_codesmith_workflow(user_input))