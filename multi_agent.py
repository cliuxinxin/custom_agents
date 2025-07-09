import os
import json
import asyncio
from typing import List, Dict, Any, Literal
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

# ========== deepseek模型配置 ==========
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

# ========== Agent schema ==========
class PlanStep(BaseModel):
    step_number: int
    task_description: str
    expected_output: str

class PlanOutput(BaseModel):
    steps: List[PlanStep]

class ExecuteOutput(BaseModel):
    result: Any = None
    message: str = ""

class JudgeOutput(BaseModel):
    passed: bool
    reason: str = ""
    problem: Literal["plan", "execute"]
    suggestion: str = ""

# ========== Step schema ==========
class StepTry(BaseModel):
    step_number: int
    status : str = ""
    human_hint: str = ""
    current_try: int = 0
    exec_input: Any = {}
    exec_result: Any = {}
    judge_result: Any = {}
    function_outputs: list = []  # 新增字段

class StepState(BaseModel):
    status: str = "pending"  # pending/passed/failed
    tries: list = []  # List[StepTry]
    current_try: int = 0
    target: str = ""
    current_step: int = 0

class PlanState(BaseModel):
    target: str
    current_step: int = 0
    current_try: int = 0
    plan_steps: List[PlanStep]  # 静态计划内容
    try_status: List[StepState]  # 动态执行状态
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
    plan_steps = [PlanStep(**step) for step in data['plan_steps']]
    step_states = []
    for s in data['try_status']:
        tries = [StepTry(**t) for t in s.get('tries',[])]
        step_states.append(StepState(
            status=s.get('status', 'pending'),
            tries=tries,
            current_try=s.get('current_try', 0),
            target=s.get('target', ''),
            current_step=s.get('current_step', 0)
        ))
    return PlanState(
        target=data['target'],
        plan_steps=plan_steps,
        try_status=step_states,
        current_step=data.get('current_step', 0),
        current_try=data.get('current_try', 0),
        human_hint=data.get('human_hint', "")
    )

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
            "agent_target": target
        }
        plan_result = await Runner.run(plan_agent, input=json.dumps(plan_input, ensure_ascii=False))
        plan_steps = list(plan_result.final_output.steps)  # List[PlanStep]
        # 初始化每个StepState时补充目标、全部计划、当前步骤编号
        step_states = [StepState(target=target, current_step=i, tries=[]) for i in range(len(plan_steps))]
        plan_state = PlanState(target=target, plan_steps=plan_steps, try_status=step_states)
        save_plan_state(plan_state)
        reset_log()
        write_log(f"# 目标: {target}\n")
    else:
        write_log(f"\n# 继续执行，当前进度：{plan_state.current_step}/{len(plan_state.plan_steps)}\n")

    # 2. 执行主循环
    while plan_state.current_step < len(plan_state.plan_steps):
        step_idx = plan_state.current_step
        plan_step = plan_state.plan_steps[step_idx]  # 静态计划内容
        step_state = plan_state.try_status[step_idx]  # 动态执行状态
        # 同步更新step_state的current_step、target
        step_state.current_step = step_idx
        step_state.target = plan_state.target
        if step_state.status == "passed":
            plan_state.current_step += 1
            continue

        retry_count = step_state.current_try
        while retry_count < 3:
            write_log(f"\n## 步骤 {plan_step.step_number} 第{retry_count+1}次尝试\n- 任务: {plan_step.task_description}")
            # 执行
            exec_input = {
                "step": plan_step.model_dump(),
                "state": {
                    "current_step": step_state.current_step,
                    "status": step_state.status,
                    "current_try": step_state.current_try
                },
                "human_hint": step_state.tries[-1].human_hint if step_state.tries else plan_state.human_hint
            }
            exec_result = await Runner.run(execute_agent, input=json.dumps(exec_input, ensure_ascii=False))
            # 评审
            judge_input = {
                "step": plan_step.model_dump(),
                "state": {
                    "current_step": step_state.current_step,
                    "status": step_state.status,
                    "current_try": step_state.current_try
                },
                "exec_result": exec_result.final_output.model_dump() if hasattr(exec_result.final_output, "model_dump") else exec_result.final_output
            }
            judge_result = await Runner.run(judge_agent, input=json.dumps(judge_input, ensure_ascii=False))
            # 记录本次尝试
            step_try = StepTry(
                step_number=plan_step.step_number,
                exec_input=exec_input,
                exec_result=exec_result.final_output.model_dump() if hasattr(exec_result.final_output, "model_dump") else exec_result.final_output,
                judge_result=judge_result.final_output.model_dump() if hasattr(judge_result.final_output, "model_dump") else judge_result.final_output,
                current_try=retry_count+1,
                human_hint=step_state.tries[-1].human_hint if step_state.tries else plan_state.human_hint
            )
            step_state.tries.append(step_try)
            step_state.current_try = retry_count + 1
            write_log(f"- 执行输出: {exec_result.final_output}")
            write_log(f"- 评审输出: {judge_result.final_output}")
            # 判断
            if judge_result.final_output.passed:
                step_state.status = "passed"
                write_log("- 结果: 通过\n")
                plan_state.current_step += 1
                save_plan_state(plan_state)
                break
            else:
                write_log(f"- 未通过: {judge_result.final_output.reason}")
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
                    new_step_states = []
                    for i, new_step in enumerate(new_plan_steps):
                        if i < len(plan_state.try_status) and plan_state.try_status[i].status == "passed":
                            # 已通过的步骤保留原状态
                            new_step_states.append(plan_state.try_status[i])
                        else:
                            # 新建未通过的步骤状态
                            new_step_states.append(StepState(
                                status="pending",
                                tries=[],
                                current_try=0,
                                target=plan_state.target,
                                current_step=i
                            ))
                    plan_state.plan_steps = new_plan_steps
                    plan_state.try_status = new_step_states
                    save_plan_state(plan_state)
                    write_log("- 计划已更新，继续执行\n")
                    break
                else:
                    # 执行问题，采纳建议重试
                    step_try.human_hint = suggestion
                    save_plan_state(plan_state)
                    if retry_count >= 2:
                        write_log("- 已达最大重试次数，等待人工输入\n")
                        print(f"[中断] 步骤{plan_step.step_number}连续失败3次，请输入人类意见后回车继续：")
                        human_hint = input("请输入人类意见：")
                        step_try.human_hint = human_hint
                        plan_state.human_hint = human_hint
                        step_state.current_try = 0
                        save_plan_state(plan_state)
                        retry_count = 0
                    else:
                        retry_count += 1
        else:
            print(f"[中断] 步骤{plan_step.step_number}连续失败3次，已暂停。请修改plan_state.json或输入人类意见后重试。")
            break

    if plan_state.current_step >= len(plan_state.plan_steps):
        write_log("\n[全部完成] 所有步骤已执行并检查通过！\n")
        print("\n[全部完成] 所有步骤已执行并检查通过！")

if __name__ == "__main__":
    asyncio.run(main()) 