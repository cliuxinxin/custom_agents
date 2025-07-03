import os
import json
import asyncio
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv

from agents import Agent, Runner, enable_verbose_stdout_logging, set_tracing_disabled
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

# Enable logging
set_tracing_disabled(True)
enable_verbose_stdout_logging()

# Load .env variables
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_base_url = "https://openrouter.ai/api/v1"
openrouter_model_name = "deepseek/deepseek-chat-v3-0324:free"

# Create OpenAI client and model via OpenRouter
from openai import AsyncOpenAI
openai_client = AsyncOpenAI(api_key=openrouter_api_key, base_url=openrouter_base_url)
model = OpenAIChatCompletionsModel(
    model=openrouter_model_name,
    openai_client=openai_client
)

# Load agent config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'agent_config.json')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    agent_config = json.load(f)

# ------------------ Schema ------------------
class PlanningInput(BaseModel):
    human_instruction: str
    agent_self_hint: str
    agent_target: str

class PlanStep(BaseModel):
    step_number: int
    task_description: str
    expected_output: str

class PlanOutput(BaseModel):
    steps: List[PlanStep]

class CriticOutput(BaseModel):
    passed: bool
    reason: str

# ------------------ Agents ------------------
planner_agent = Agent(
    name="PlannerAgent",
    instructions=agent_config["PlannerAgent"],
    model=model,
    output_type=PlanOutput
)

critic_agent = Agent(
    name="CriticAgent",
    instructions=agent_config["CriticAgent"],
    model=model,
    output_type=CriticOutput
)

# ------------------ Main Loop ------------------
LOG_PATH = os.path.join(os.path.dirname(__file__), 'run_log.md')

async def main():
    max_retries = 3
    retry_count = 0
    agent_target = "通过读取https://github.com/openai/openai-agents-python的代码，了解这个库怎么用，输出一个报告方便人类阅读"
    agent_self_hint = ""
    human_instruction = ""

    # 每次运行前清空旧日志
    open(LOG_PATH, "w", encoding="utf-8").close()

    PLAN_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'plan_output.json')

    while retry_count < max_retries:
        plan_input = PlanningInput(
            human_instruction=human_instruction,
            agent_self_hint=agent_self_hint,
            agent_target=agent_target
        )

        # Step 1: 生成计划
        plan_result = await Runner.run(planner_agent, input=json.dumps(plan_input.model_dump()))
        plan_output = plan_result.final_output

        # Step 2: 审核计划
        critic_input = {
            "steps": [step.model_dump() for step in plan_output.steps]
        }
        critic_result = await Runner.run(critic_agent, input=json.dumps(critic_input))
        critic_output = critic_result.final_output

        # 记录日志
        with open(LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(f"## 尝试第 {retry_count + 1} 次\n")
            log_file.write("### PlannerAgent 输入：\n")
            log_file.write(f"{json.dumps(plan_input.model_dump(), ensure_ascii=False, indent=2)}\n")
            log_file.write("### PlannerAgent 输出：\n")
            log_file.write(f"{json.dumps({'steps': [step.model_dump() for step in plan_output.steps]}, ensure_ascii=False, indent=2)}\n")
            log_file.write("### CriticAgent 输入：\n")
            log_file.write(f"{json.dumps(critic_input, ensure_ascii=False, indent=2)}\n")
            log_file.write("### CriticAgent 输出：\n")
            log_file.write(f"{json.dumps(critic_output.model_dump(), ensure_ascii=False, indent=2)}\n")
            if critic_output.passed:
                log_file.write("✅ 计划通过\n\n")
            else:
                log_file.write(f"❌ 失败原因：{critic_output.reason}\n\n")

        if critic_output.passed:
            print("✅ 计划通过：")
            for step in plan_output.steps:
                print(f"步骤{step.step_number}: {step.task_description} => {step.expected_output}")
            # 输出计划到文件，包含target和steps
            with open(PLAN_OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"target": agent_target, "steps": [step.model_dump() for step in plan_output.steps]}, f, ensure_ascii=False, indent=2)
            break
        else:
            retry_count += 1
            print(f"❌ 第 {retry_count} 次失败：{critic_output.reason}")
            machine_hint += f"\n尝试{retry_count}失败原因：{critic_output.reason}"
            agent_self_hint += f"\n上一次计划：{[step.task_description for step in plan_output.steps]}"

    if retry_count >= max_retries:
        print("❗三次都失败，请人类检查并更新指示后重试。")

if __name__ == "__main__":
    asyncio.run(main())
