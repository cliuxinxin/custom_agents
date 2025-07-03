import os
import json
import subprocess
from typing import List, Dict, Any
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

# ------------------ Agents ------------------
executor_agent = Agent(
    name="ExecutorAgent",
    instructions=agent_config["ExecutorAgent"],
    model=model,
    output_type=None
)
checker_agent = Agent(
    name="CheckerAgent",
    instructions=agent_config["CheckerAgent"],
    model=model,
    output_type=None
)
suggester_agent = Agent(
    name="SuggesterAgent",
    instructions=agent_config["SuggesterAgent"],
    model=model,
    output_type=None
)

# ------------------ 文件路径 ------------------
PLAN_PATH = os.path.join(os.path.dirname(__file__), 'plan_output.json')
STATE_PATH = os.path.join(os.path.dirname(__file__), 'execution_state.json')
LOG_PATH = os.path.join(os.path.dirname(__file__), 'execute_log.md')

# ------------------ 工具函数 ------------------
def load_plan() -> Dict[str, Any]:
    with open(PLAN_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {"current_step": 0, "steps": []}
    with open(STATE_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_state(state: Dict[str, Any]):
    with open(STATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def write_log(content: str):
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(content + '\n')

def reset_log():
    open(LOG_PATH, 'w', encoding='utf-8').close()

def execute_shell(command: str) -> Dict[str, Any]:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        return {"returncode": -1, "stdout": "", "stderr": str(e)}

# ------------------ 主流程 ------------------
def main():
    plan = load_plan()
    state = load_state()
    steps = plan["steps"]
    current_step = state.get("current_step", 0)
    total_steps = len(steps)
    reset_log()
    write_log(f"# 执行日志\n目标: {plan.get('target', '')}\n")

    for idx in range(current_step, total_steps):
        step = steps[idx]
        retry_count = 0
        success = False
        last_suggest = ""
        while retry_count < 3 and not success:
            write_log(f"## 步骤 {step['step_number']} 第{retry_count+1}次尝试\n- 任务: {step['task_description']}")
            print(f"\n>>> 执行步骤 {step['step_number']} (第{retry_count+1}次): {step['task_description']}")
            # 构造输入
            human_hint = ""  # 可扩展为人工输入
            exec_input = f"步骤内容: {step['task_description']}\n人类指导: {human_hint}\n反馈意见: {last_suggest}"
            # 用agent执行
            exec_result = Runner.run_sync(executor_agent, input=exec_input)
            # 兼容shell命令自动执行
            if "shell" in step['task_description'] or "命令" in step['task_description'] or step['task_description'].startswith("sh "):
                command = step['task_description'].replace('shell ', '').replace('命令 ', '').replace('sh ', '')
                shell_result = execute_shell(command)
                exec_result_dict = {
                    "agent_output": exec_result.final_output,
                    "returncode": shell_result["returncode"],
                    "stdout": shell_result["stdout"],
                    "stderr": shell_result["stderr"]
                }
            else:
                exec_result_dict = {"agent_output": exec_result.final_output}
            write_log(f"- 执行输出: {exec_result_dict.get('stdout', exec_result_dict.get('agent_output',''))}\n- 执行错误: {exec_result_dict.get('stderr','')}\n- agent输出: {exec_result_dict.get('agent_output','')}")
            # 检查步骤
            check_input = {
                "target": plan.get("target", ""),
                "step": step,
                "exec_result": exec_result_dict,
                "human_hint": human_hint
            }
            check_result = Runner.run_sync(checker_agent, input=json.dumps(check_input))
            write_log(f"- 检查结果: {check_result.final_output}")
            print(f"[检查结果] {check_result.final_output}")
            # 判断是否通过
            if isinstance(check_result.final_output, dict):
                passed = check_result.final_output.get("passed", True)
            else:
                passed = "通过" in str(check_result.final_output) or "success" in str(check_result.final_output).lower()
            if passed:
                success = True
                write_log(f"- 结果: 通过\n")
                # 记录状态
                state['steps'].append({
                    "step_number": step['step_number'],
                    "task_description": step['task_description'],
                    "exec_result": exec_result_dict,
                    "check_result": check_result.final_output,
                    "retry": retry_count+1
                })
                state['current_step'] = idx + 1
                save_state(state)
            else:
                # 失败时建议修改
                suggest_input = {
                    "target": plan.get("target", ""),
                    "step": step,
                    "exec_result": exec_result_dict,
                    "check_result": check_result.final_output,
                    "human_hint": human_hint
                }
                suggest_result = Runner.run_sync(suggester_agent, input=json.dumps(suggest_input))
                last_suggest = suggest_result.final_output
                write_log(f"- 建议: {suggest_result.final_output}")
                print(f"[建议] {suggest_result.final_output}")
                retry_count += 1
                if retry_count < 3:
                    write_log(f"- 重试: 第{retry_count+1}次\n")
                else:
                    write_log(f"- 已达最大重试次数，人工介入。\n")
                    print("[中断] 执行失败已达三次，请根据建议修改计划或人工介入后重试。")
                    # 记录最终失败状态
                    state['steps'].append({
                        "step_number": step['step_number'],
                        "task_description": step['task_description'],
                        "exec_result": exec_result_dict,
                        "check_result": check_result.final_output,
                        "suggest": suggest_result.final_output,
                        "retry": retry_count
                    })
                    state['current_step'] = idx
                    save_state(state)
                    write_log(f"\n---\n# 执行中断，等待人工干预。\n")
                    return
    else:
        write_log("\n[全部完成] 所有步骤已执行并检查通过！\n")
        print("\n[全部完成] 所有步骤已执行并检查通过！")

if __name__ == "__main__":
    main() 