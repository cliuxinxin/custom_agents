import os
import json
from typing import List
from pydantic import BaseModel
from agents import Agent, Runner, enable_verbose_stdout_logging, set_tracing_disabled
from openai import AsyncOpenAI
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

from dotenv import load_dotenv

load_dotenv()

# 启用日志
set_tracing_disabled(True)
enable_verbose_stdout_logging()

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

# 规划输出的数据结构
class PlanOutput(BaseModel):
    steps: List[str]

# 你可以根据需要自定义 instructions
instructions = (
    "你是一个软件项目经理。用户会输入一个需求，你需要使用WBS系统把目标拆分为工作包，"
    "每一个工作包都要清晰、具体，确保程序员可以按照步骤逐步完成任务。"
    "每一个工作包就是一行。"
)

planner_agent = Agent(
    name="PlannerAgent",
    instructions=instructions,
    output_type=PlanOutput,
    model=model
)



# 示例用法
import asyncio

async def main():
    question = "写一个baidu.com的爬虫"
    result = await Runner.run(planner_agent, input=question)
    print("规划步骤：")
    steps = result.final_output.steps
    for idx, step in enumerate(steps, 1):
        print(f"{idx}. {step}")
    # 写入文件
    with open("agent_plan.md", "w", encoding="utf-8") as f:
        f.write("规划步骤：\n")
        for idx, step in enumerate(steps, 1):
            f.write(f"{step}\n")

if __name__ == "__main__":
    asyncio.run(main())