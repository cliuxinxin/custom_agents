import os
import json
from agents import Agent
from openai import AsyncOpenAI
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

# 加载 .env 文件
load_dotenv()
# 配置 OpenRouter 的信息
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_base_url = "https://openrouter.ai/api/v1"
openrouter_model_name = "deepseek/deepseek-chat-v3-0324:free"

# 创建 OpenAI 客户端，使用 OpenRouter 的信息
openai_client = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url=openrouter_base_url
)

# 创建 OpenAIChatCompletionsModel 实例
model = OpenAIChatCompletionsModel(
    model=openrouter_model_name,
    openai_client=openai_client
)

# 加载 agent 配置（指令/提示词）
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'agent_config.json')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    agent_config = json.load(f)

# 目标解析模块
class GoalInterpreterOutput(BaseModel):
    task: str
    input: str
    output: str
    modules: List[str]
    constraints: List[str]

goal_interpreter_agent = Agent(
    name="GoalInterpreterAgent",
    instructions=agent_config["GoalInterpreterAgent"],
    output_type=GoalInterpreterOutput,
    model=model
)

# 任务拆解模块
class SubTask(BaseModel):
    module: str
    sub_goal: str
    dependencies: List[str]

class TaskPlannerOutput(BaseModel):
    sub_tasks: List[SubTask]

task_planner_agent = Agent(
    name="TaskPlannerAgent",
    instructions=agent_config["TaskPlannerAgent"],
    output_type=TaskPlannerOutput,
    model=model
)

# 代码生成模块
code_writer_agent = Agent(
    name="CodeWriterAgent",
    instructions=agent_config["CodeWriterAgent"],
    model=model
)

# 执行与测试模块
executor_agent = Agent(
    name="ExecutorAgent",
    instructions=agent_config["ExecutorAgent"],
    model=model
)

# 错误分析模块
class ErrorAnalyzerOutput(BaseModel):
    error_type: str
    error_reason: str
    fix_suggestion: str

error_analyzer_agent = Agent(
    name="ErrorAnalyzerAgent",
    instructions=agent_config["ErrorAnalyzerAgent"],
    output_type=ErrorAnalyzerOutput,
    model=model
)

# 导出所有 agent 和相关类型
__all__ = [
    "model",
    "GoalInterpreterOutput",
    "goal_interpreter_agent",
    "SubTask",
    "TaskPlannerOutput",
    "task_planner_agent",
    "code_writer_agent",
    "executor_agent",
    "ErrorAnalyzerOutput",
    "error_analyzer_agent"
] 