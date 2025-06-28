import argparse
import asyncio
import os
import subprocess
import sys
from enum import Enum
from typing import List
from agents import Runner, enable_verbose_stdout_logging, set_tracing_disabled
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import json
# 导入agent和相关类型
from agent import (
    model,
    GoalInterpreterOutput,
    goal_interpreter_agent,
    SubTask,
    TaskPlannerOutput,
    task_planner_agent,
    code_writer_agent,
    executor_agent,
    ErrorAnalyzerOutput,
    error_analyzer_agent
)

set_tracing_disabled(True)
enable_verbose_stdout_logging()
logging.basicConfig(level=logging.DEBUG)

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

# 1. 用户输入层（CLI）
def get_user_input():
    parser = argparse.ArgumentParser(description='CodeSmith Agent')
    parser.add_argument('input', type=str, help='自然语言描述的开发目标')
    args = parser.parse_args()
    return args.input

# 2. Agent Core Loop（大脑系统）
# (1) 目标解析模块
async def run_goal_interpreter(input_task):
    result = await Runner.run(goal_interpreter_agent, input_task)
    return result.final_output

# (2) 任务拆解模块
async def run_task_planner(goal_info):
    input_list = [{"role": "user", "content": str(goal_info)}]
    result = await Runner.run(task_planner_agent, input_list)
    return result.final_output

# (3) 代码生成模块
async def run_code_writer(sub_tasks, work_dir):
    for sub_task in sub_tasks:
        input_list = [{
            "role": "user",
            "content": f"模块: {getattr(sub_task, 'module', '')}\n子目标: {getattr(sub_task, 'sub_goal', '')}\n依赖: {', '.join(getattr(sub_task, 'dependencies', []))}"
        }]
        result = await Runner.run(code_writer_agent, input_list)
        code = result.final_output
        file_name = f"{getattr(sub_task, 'module', 'module')}.py"
        file_path = os.path.join(work_dir, file_name)
        with open(file_path, 'w') as f:
            f.write(code)

# (4) 执行与测试模块
async def run_executor(work_dir, main_file):
    main_path = os.path.join(work_dir, main_file)
    try:
        result = subprocess.run([sys.executable, main_path], capture_output=True, text=True, check=True)
        output = result.stdout
        error = None
    except subprocess.CalledProcessError as e:
        output = e.stdout
        error = e.stderr
    return output, error

# (5) 错误分析模块
async def run_error_analyzer(error_log):
    result = await Runner.run(error_analyzer_agent, error_log)
    return result.final_output

# (6) 记忆与上下文模块
class MemoryManager:
    def __init__(self, save_path="memory.json"):
        self.save_path = save_path
        self.global_goal = None
        self.current_progress = None
        self.sub_task_status = {}
        self.failure_attempts = {}
        self.fix_logs = {}
        self.human_feedback = ""  # 新增：人工补充信息

    def save(self):
        data = self.__dict__.copy()
        with open(self.save_path, "w") as f:
            json.dump(data, f)

    def load(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, "r") as f:
                data = json.load(f)
                self.__dict__.update(data)

    def update_human_feedback(self, feedback):
        self.human_feedback = feedback
        self.save()

    def update_global_goal(self, goal):
        self.global_goal = goal

    def update_current_progress(self, progress):
        self.current_progress = progress

    def update_sub_task_status(self, sub_task, status):
        self.sub_task_status[sub_task] = status

    def increment_failure_attempts(self, sub_task):
        if sub_task not in self.failure_attempts:
            self.failure_attempts[sub_task] = 0
        self.failure_attempts[sub_task] += 1

    def add_fix_log(self, sub_task, log):
        if sub_task not in self.fix_logs:
            self.fix_logs[sub_task] = []
        self.fix_logs[sub_task].append(log)

# 3. 可控性与护栏设计
# (1) 目标注入机制
def inject_goal_prompt(prompt, global_goal, sub_task):
    return f'你正在为目标"{global_goal}"中的子任务"{sub_task}"编写代码。{prompt}'

# (2) 状态驱动（FSM）
class AgentState(Enum):
    START = "START"
    GOAL_PARSED = "GOAL_PARSED"
    TASKS_PLANNED = "TASKS_PLANNED"
    CODE_WRITTEN = "CODE_WRITTEN"
    TESTED = "TESTED"
    FIXING_ERROR = "FIXING_ERROR"
    DONE = "DONE"

# (3) 错误限速与终止机制
MAX_FAILURE_ATTEMPTS = 3

def check_failure_attempts(memory_manager, sub_task):
    if sub_task in memory_manager.failure_attempts and memory_manager.failure_attempts[sub_task] >= MAX_FAILURE_ATTEMPTS:
        return True
    return False

# (4) 错误分类处理机制
def classify_error(error_log):
    if "invalid syntax" in error_log:
        return "SyntaxError", "提示模型重写该行代码"
    elif "No module named" in error_log:
        return "ImportError", "检查是否写错模块名/requirements"
    elif "index out of range" in error_log:
        return "IndexError", "检查列表长度与访问边界"
    elif "ConnectionError" in error_log:
        return "RequestError", "加重试机制或代理"
    return "UnknownError", "无明确修复策略"

# 4. 整体流程实现
async def main():
    # 用户输入
    input_task = get_user_input()

    # 初始化记忆管理器，尝试加载历史
    memory_manager = MemoryManager()
    memory_manager.load()
    if memory_manager.current_progress:
        print(f"检测到未完成任务，当前进度：{memory_manager.current_progress}")
        resume = input("是否继续上次任务？(y/n): ")
        if resume.lower() != "y":
            memory_manager = MemoryManager()  # 重置
            memory_manager.update_global_goal(input_task)
    else:
        memory_manager.update_global_goal(input_task)

    # 目标解析
    if memory_manager.current_progress != "GOAL_PARSED":
        goal_info = await run_goal_interpreter(input_task)
        memory_manager.update_current_progress("GOAL_PARSED")
        memory_manager.save()
    else:
        goal_info = None  # 可根据需要从memory恢复

    # 任务拆解
    if memory_manager.current_progress != "TASKS_PLANNED":
        sub_tasks_info = await run_task_planner(goal_info)
        memory_manager.update_current_progress("TASKS_PLANNED")
        memory_manager.save()
    else:
        sub_tasks_info = None  # 可根据需要从memory恢复

    # 代码生成
    work_dir = "workspace"
    os.makedirs(work_dir, exist_ok=True)
    if memory_manager.current_progress != "CODE_WRITTEN":
        await run_code_writer(sub_tasks_info.sub_tasks, work_dir)
        memory_manager.update_current_progress("CODE_WRITTEN")
        memory_manager.save()

    # 执行与测试
    main_file = "main.py"  # 假设主程序文件名为 main.py
    output, error = await run_executor(work_dir, main_file)
    memory_manager.update_current_progress("TESTED")
    memory_manager.save()

    while error:
        # 错误分析
        error_analysis = await run_error_analyzer(error)
        memory_manager.increment_failure_attempts(main_file)
        memory_manager.add_fix_log(main_file, error_analysis.fix_suggestion)
        memory_manager.save()

        # 错误限速与终止机制检查
        if check_failure_attempts(memory_manager, main_file):
            print("连续失败次数达到上限，进入 HumanHelpNeeded 状态")
            # 允许人工输入补充信息
            feedback = input("请补充有用的信息（如 humanloop 分析、修复建议等），回车跳过：\n")
            if feedback.strip():
                memory_manager.update_human_feedback(feedback)
            break

        # 代码修复，注入人工补充信息
        fix_prompt = inject_goal_prompt(
            error_analysis.fix_suggestion + "\n" + memory_manager.human_feedback,
            input_task, "修复代码"
        )
        fix_result = await Runner.run(code_writer_agent, fix_prompt)
        fix_code = fix_result.final_output
        fix_file_path = os.path.join(work_dir, main_file)
        with open(fix_file_path, 'w') as f:
            f.write(fix_code)

        # 再次执行与测试
        output, error = await run_executor(work_dir, main_file)
        memory_manager.update_current_progress("TESTED")
        memory_manager.save()

    if not error:
        memory_manager.update_current_progress("DONE")
        memory_manager.save()
        print("项目已成功运行，输出结果：", output)

if __name__ == "__main__":
    asyncio.run(main()) 