# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-agent AI system that uses OpenAI's agents library to create a collaborative workflow with three specialized agents:

1. **PlanAgent**: Breaks down complex tasks into detailed steps
2. **ExecuteAgent**: Executes individual steps with access to file system and shell commands
3. **JudgeAgent**: Evaluates execution results and provides feedback

## Architecture

The system follows a sequential workflow:
- PlanAgent creates a detailed plan with numbered steps
- ExecuteAgent executes each step with available tools
- JudgeAgent evaluates each execution result
- State is persisted in `workspace/plan_state.json`
- Execution logs are saved in `workspace/multi_agent_log.md`

### Key Components

- `multi_agent.py`: Main orchestration logic and agent definitions
- `agent_config.json`: Agent prompts and configuration (Chinese language)
- `workspace/`: Runtime directory for state and logs
- Uses OpenRouter API with DeepSeek model (deepseek-chat-v3-0324:free)

### Agent Tools

ExecuteAgent has access to:
- `execute_shell_command`: Safe shell command execution
- `read_file_content`: Read file contents
- `write_file_content`: Write/create files
- `list_directory`: List directory contents
- `check_file_exists`: Check file existence

## Development Commands

### Setup
```bash
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file with:
```
OPENROUTER_API_KEY=your_openrouter_api_key
```

### Running the System
```bash
python multi_agent.py
```

The system will:
1. Load or create a plan for the target goal
2. Execute each step sequentially
3. Save state after each step
4. Generate execution logs

### State Management
- Plan state is persisted in `workspace/plan_state.json`
- Execution continues from the last completed step on restart
- Logs are appended to `workspace/multi_agent_log.md`

## Configuration Notes

- All agent prompts are in Chinese as configured in `agent_config.json`
- The system uses structured output with Pydantic models
- Current target is hardcoded in `main()` function (line 243)
- No retry logic - each step executes once and moves to the next

## File Structure

- Root contains main application files
- `workspace/` directory is auto-created for runtime state
- `openai-agents-python/` directory may contain cloned source code for analysis