import os
import json
import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv

from agents import Agent, Runner, enable_verbose_stdout_logging, set_tracing_disabled, function_tool
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
import subprocess

# ========== function_tool 工具 ==========
@function_tool
def execute_shell_command(command: List[str]) -> Dict[str, Any]:
    """安全执行外部命令（不抛异常），返回完整的执行结果。command 必须为字符串数组。"""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=False
        )
        return {
            "success": result.returncode == 0,
            "args": result.args,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        return {
            "success": False,
            "args": command,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e)
        }

@function_tool
def read_file_content(file_path: str) -> Dict[str, Any]:
    """读取文件内容"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {
                "success": True,
                "content": content,
                "file_path": file_path
            }
        else:
            return {
                "success": False,
                "error": f"文件不存在: {file_path}",
                "file_path": file_path
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }

@function_tool
def write_file_content(file_path: str, content: str) -> Dict[str, Any]:
    """写入文件内容"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {
            "success": True,
            "file_path": file_path,
            "message": f"文件已成功写入: {file_path}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }

@function_tool
def list_directory(directory_path: str = ".") -> Dict[str, Any]:
    """列出目录内容"""
    try:
        if os.path.exists(directory_path):
            items = os.listdir(directory_path)
            files = []
            directories = []
            for item in items:
                item_path = os.path.join(directory_path, item)
                if os.path.isfile(item_path):
                    files.append(item)
                elif os.path.isdir(item_path):
                    directories.append(item)
            return {
                "success": True,
                "directory": directory_path,
                "files": files,
                "directories": directories
            }
        else:
            return {
                "success": False,
                "error": f"目录不存在: {directory_path}",
                "directory": directory_path
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "directory": directory_path
        }

@function_tool
def check_file_exists(file_path: str) -> Dict[str, Any]:
    """检查文件是否存在"""
    exists = os.path.exists(file_path)
    return {
        "exists": exists,
        "file_path": file_path,
        "message": f"文件{'存在' if exists else '不存在'}: {file_path}"
    }

# ========== 路径与配置 ==========
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'workspace')
os.makedirs(OUTPUT_DIR, exist_ok=True)
PLAN_STATE_PATH = os.path.join(OUTPUT_DIR, 'plan_state.json')
LOG_PATH = os.path.join(OUTPUT_DIR, 'multi_agent_log.md')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'agent_config.json')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    agent_config = json.load(f)

# ========== openai模型配置 ==========
set_tracing_disabled(True)
enable_verbose_stdout_logging()
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_base_url = "https://openrouter.ai/api/v1"
openrouter_model_name = "deepseek/deepseek-chat-v3-0324:free"
from openai import AsyncOpenAI
openai_client = AsyncOpenAI(api_key=openrouter_api_key, base_url=openrouter_base_url)
model = OpenAIChatCompletionsModel(
    model=openrouter_model_name,
    openai_client=openai_client
)

# ========== schema ==========
# 只用于计划生成
class PlanStep(BaseModel):
    step_number: int
    task_description: str
    expected_output: str
    milestone: str

class PlanOutput(BaseModel):
    steps: List[PlanStep]

class ExecuteOutput(BaseModel):
    result: Any = None
    message: str = ""

class JudgeOutput(BaseModel):
    passed: bool
    reason: str = ""
    problem: str = ""
    suggestion: str = ""

# 只用于执行记录
class StepState(BaseModel):
    status: str = "pending"  # pending, passed, failed
    exec_result: Any = None
    judge_result: Any = None
    retry: int = 0
    human_hint: str = ""

class PlanState(BaseModel):
    target: str
    plan_steps: List[PlanStep]  # 静态计划内容
    step_states: List[StepState]  # 动态执行状态
    current_step: int = 0
    human_hint: str = ""

# ========== agent实例 ==========
plan_agent = Agent(
    name="PlanAgent",
    instructions=agent_config["PlanAgent"],
    model=model,
    output_type=PlanOutput
)
execute_agent = Agent(
    name="ExecuteAgent",
    instructions=agent_config["ExecuteAgent"],
    model=model,
    tools=[execute_shell_command, read_file_content, write_file_content, list_directory, check_file_exists],
    output_type=ExecuteOutput
)
judge_agent = Agent(
    name="JudgeAgent",
    instructions=agent_config["JudgeAgent"],
    model=model,
    output_type=JudgeOutput
)

# ========== 工具函数 ==========
def save_plan_state(state: PlanState):
    with open(PLAN_STATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(state.model_dump(), f, ensure_ascii=False, indent=2)

def load_plan_state() -> PlanState:
    if not os.path.exists(PLAN_STATE_PATH):
        return None
    with open(PLAN_STATE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    steps = [PlanStep(**step) for step in data['steps']]
    return PlanState(target=data['target'], steps=steps, current_step=data.get('current_step', 0), human_hint=data.get('human_hint', ""))

def write_log(content: str):
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(content + '\n')

def reset_log():
    open(LOG_PATH, 'w', encoding='utf-8').close()

# ========== 主流程 ==========
async def main():
    # 1. 初始化或读取计划
    plan_state = load_plan_state()
    if plan_state is None:
        # 首次运行，生成计划
        target = "通过读取https://github.com/openai/openai-agents-python的代码，了解这个库怎么用，输出一个报告方便人类阅读"
        plan_input = {
            "human_instruction": "",
            "agent_self_hint": "",
            "agent_target": target
        }
        plan_result = await Runner.run(plan_agent, input=json.dumps(plan_input, ensure_ascii=False))
        plan_steps = list(plan_result.final_output.steps)  # List[PlanStep]
        step_states = [StepState() for _ in plan_steps]
        plan_state = PlanState(target=target, plan_steps=plan_steps, step_states=step_states)
        save_plan_state(plan_state)
        reset_log()
        write_log(f"# 目标: {target}\n")
    else:
        write_log(f"\n# 继续执行，当前进度：{plan_state.current_step}/{len(plan_state.plan_steps)}\n")

    # 2. 执行主循环
    while plan_state.current_step < len(plan_state.plan_steps):
        step_idx = plan_state.current_step
        plan_step = plan_state.plan_steps[step_idx]  # 静态计划内容
        step_state = plan_state.step_states[step_idx]  # 动态执行状态
        if step_state.status == "passed":
            plan_state.current_step += 1
            continue

        retry_count = step_state.retry
        while retry_count < 3:
            write_log(f"\n## 步骤 {plan_step.step_number} 第{retry_count+1}次尝试\n- 任务: {plan_step.task_description}")
            # 执行
            exec_input = {
                "step": plan_step.model_dump(),
                "state": step_state.model_dump(),
                "plan": plan_state.model_dump(),
                "human_hint": step_state.human_hint or plan_state.human_hint
            }
            exec_result = await Runner.run(execute_agent, input=json.dumps(exec_input, ensure_ascii=False))
            step_state.exec_result = exec_result.final_output
            write_log(f"- 执行输出: {exec_result.final_output}")

            # 评审
            judge_input = {
                "step": plan_step.model_dump(),
                "state": step_state.model_dump(),
                "exec_result": exec_result.final_output.model_dump() if hasattr(exec_result.final_output, "model_dump") else exec_result.final_output,
                "plan": plan_state.model_dump()
            }
            judge_result = await Runner.run(judge_agent, input=json.dumps(judge_input, ensure_ascii=False))
            step_state.judge_result = judge_result.final_output
            write_log(f"- 评审输出: {judge_result.final_output}")

            # 判断
            if judge_result.final_output.passed:
                step_state.status = "passed"
                write_log("- 结果: 通过\n")
                plan_state.current_step += 1
                step_state.retry = retry_count + 1
                save_plan_state(plan_state)
                break
            else:
                retry_count += 1
                step_state.retry = retry_count
                write_log(f"- 未通过: {judge_result.final_output.reason}")
                # 判断问题环节
                problem = judge_result.final_output.problem
                suggestion = judge_result.final_output.suggestion
                if problem == "plan":
                    # 重新规划，保留已通过部分
                    plan_input = {
                        "human_instruction": plan_state.human_hint,
                        "agent_self_hint": "",
                        "agent_target": plan_state.target,
                        "history_steps": [s.model_dump() for s in plan_state.plan_steps]
                    }
                    plan_result = await Runner.run(plan_agent, input=json.dumps(plan_input, ensure_ascii=False))
                    new_plan_steps = list(plan_result.final_output.steps)
                    # 只更新未通过部分
                    for i, s in enumerate(plan_state.plan_steps):
                        if plan_state.step_states[i].status == "passed":
                            continue
                        if i < len(new_plan_steps):
                            plan_state.plan_steps[i] = new_plan_steps[i]
                    save_plan_state(plan_state)
                    write_log("- 计划已更新，继续执行\n")
                    break
                else:
                    # 执行问题，采纳建议重试
                    step_state.human_hint = suggestion
                    save_plan_state(plan_state)
                    if retry_count >= 3:
                        write_log("- 已达最大重试次数，等待人工输入\n")
                        print(f"[中断] 步骤{plan_step.step_number}连续失败3次，请输入人类意见后回车继续：")
                        human_hint = input("请输入人类意见：")
                        step_state.human_hint = human_hint
                        plan_state.human_hint = human_hint
                        step_state.retry = 0
                        save_plan_state(plan_state)
                        retry_count = 0  # 重置重试次数
        else:
            # 3次都失败，人工介入
            print(f"[中断] 步骤{plan_step.step_number}连续失败3次，已暂停。请修改plan_state.json或输入人类意见后重试。")
            break

    if plan_state.current_step >= len(plan_state.plan_steps):
        write_log("\n[全部完成] 所有步骤已执行并检查通过！\n")
        print("\n[全部完成] 所有步骤已执行并检查通过！")

if __name__ == "__main__":
    asyncio.run(main()) 