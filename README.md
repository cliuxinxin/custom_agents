# custom_agents

## 项目简介

本项目基于 [openai-agents-python](https://github.com/openai/openai-agents-python) 框架，结合自定义配置，实现了自动化的智能体任务规划与审核流程。通过 `plan_agent.py`，可自动生成并审核结构化的任务计划，适用于需求分析、项目拆解、自动化规划等场景。

## 依赖安装

建议使用 Python 3.8 及以上版本。

```bash
pip install -r requirements.txt
```

依赖列表（见 requirements.txt）：
- openai-agents
- openai
- python-dotenv

## 环境配置

需在项目根目录下创建 `.env` 文件，内容示例：

```
OPENROUTER_API_KEY=你的OpenRouter API Key
```

如需自定义模型或API地址，可在 `plan_agent.py` 中修改：
- `openrouter_base_url`
- `openrouter_model_name`

## 主要功能

- 通过 LLM 智能体自动生成任务计划（PlannerAgent）
- 对任务计划进行自动审核（CriticAgent）
- 支持结构化输出、日志记录、失败重试
- 可扩展自定义 agent 配置（见 agent_config.json）

## 使用方法

### 1. 运行主流程

```bash
python plan_agent.py
```

运行后会自动生成：
- `plan_output.json`：结构化的任务计划输出
- `run_log.md`：详细的运行日志

### 2. 参考示例

- `references/agent-basic.md`：包含基础 agent 用法、函数工具、handoff 等多种示例
- `references/run_scrtipt.py`：演示如何通过脚本运行 agent

### 3. 主要文件说明

- `plan_agent.py`：主流程脚本，自动生成和审核任务计划
- `agent_config.json`：agent 配置文件，可自定义指令
- `plan_output.json`：最近一次的任务计划输出
- `run_log.md`：最近一次的运行日志
- `requirements.txt`：依赖列表

## 输出说明

- `plan_output.json` 示例：
```json
{
  "target": "通过读取https://github.com/openai/openai-agents-python的代码，了解这个库怎么用，输出一个报告方便人类阅读",
  "steps": [
    {"step_number": 1, "task_description": "...", "expected_output": "..."},
    ...
  ]
}
```
- `run_log.md` 记录了每次运行的输入、输出、审核过程及结果。

## 常见问题

1. **API Key 报错**：请确保 `.env` 文件已正确配置 OpenRouter API Key。
2. **依赖缺失**：请先执行 `pip install -r requirements.txt`。
3. **自定义 agent 行为**：可编辑 `agent_config.json`，调整 agent 指令。
4. **模型/接口报错**：可尝试更换 `plan_agent.py` 中的模型名或 API 地址。

## 参考
- [openai-agents-python 官方文档](https://github.com/openai/openai-agents-python)
- 项目内 `references/` 目录下的示例和说明 