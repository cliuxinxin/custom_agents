import subprocess
import sys

# 构建命令，使用当前python解释器
cmd = [sys.executable, 'new_agent.py']

# 运行命令并捕获输出
result = subprocess.run(cmd, capture_output=True, text=True)

print('stdout:')
print(result.stdout)
print('stderr:')
print(result.stderr) 

# output
'''
stdout:
Tracing is disabled. Not creating trace Agent workflow
Setting current trace: no-op
Tracing is disabled. Not creating span <agents.tracing.span_data.AgentSpanData object at 0x11021b750>
Running agent PlannerAgent (turn 1)
Tracing is disabled. Not creating span <agents.tracing.span_data.GenerationSpanData object at 0x1101a93d0>
[
  {
    "content": "你是一个专业的软件项目规划 Agent，输入是一个任务需求，输出是结构化的工作包列表。\n每个工作包包含以下字段：\nid: 任务的唯一标识符（如 step1, step2）\ntitle: 任务名称，简短明了\ndescription: 明确要实现的功能和方法，粒度控制在可以写成一个函数或一个 Python 文件的级别\ndependencies: 本任务依赖的上一个或多个任务的 id\n\n请确保输出是一个 JSON 对象，结构符合 Task 的定义。\n",
    "role": "system"
  },
  {
    "role": "user",
    "content": "写一个baidu.com的爬虫"
  }
]
Tools:
[]
Stream: False
Tool choice: NOT_GIVEN
Response format: {'type': 'json_schema', 'json_schema': {'name': 'final_output', 'strict': True, 'schema': {'$defs': {'Task': {'properties': {'id': {'title': 'Id', 'type': 'string'}, 'title': {'title': 'Title', 'type': 'string'}, 'description': {'title': 'Description', 'type': 'string'}, 'dependencies': {'default': [], 'items': {'type': 'string'}, 'title': 'Dependencies', 'type': 'array'}}, 'required': ['id', 'title', 'description', 'dependencies'], 'title': 'Task', 'type': 'object', 'additionalProperties': False}}, 'properties': {'steps': {'items': {'$ref': '#/$defs/Task'}, 'title': 'Steps', 'type': 'array'}}, 'required': ['steps'], 'title': 'PlanOutput', 'type': 'object', 'additionalProperties': False}}}

LLM resp:
{
  "content": "{\n  \"steps\": [\n    {\n      \"dependencies\": [],\n      \"description\": \"完成百度首页的HTTP请求代码编写，获取网页HTML源代码\",\n      \"id\": \"step1\",\n      \"title\": \"创建基础HTTP请求\"\n    },\n    {\n      \"dependencies\": [\"step1\"],\n      \"description\": \"使用BeautifulSoup或lxml等HTML解析库解析获取到的HTML源代码\",\n      \"id\": \"step2\",\n      \"title\": \"HTML解析器配置\"\n    },\n    {\n      \"dependencies\": [\"step2\"],\n      \"description\": \"从解析后的HTML中提取所需信息，如搜索结果、链接等\",\n      \"id\": \"step3\",\n      \"title\": \"数据提取实现\"\n    },\n    {\n      \"dependencies\": [\"step3\"],\n      \"description\": \"将提取的数据保存为JSON或其他格式的文件\",\n      \"id\": \"step4\",\n      \"title\": \"数据存储功能\"\n    },\n    {\n      \"dependencies\": [\"step1\"],\n      \"description\": \"为爬虫添加异常处理和日志记录功能，确保稳定运行\",\n      \"id\": \"step5\",\n      \"title\": \"错误处理和日志\"\n    }\n  ]\n}",
  "refusal": null,
  "role": "assistant",
  "annotations": null,
  "audio": null,
  "function_call": null,
  "tool_calls": null,
  "reasoning": null
}

Resetting current trace
规划步骤：
1. id='step1' title='创建基础HTTP请求' description='完成百度首页的HTTP请求代码编写，获取网页HTML源代码' dependencies=[]
2. id='step2' title='HTML解析器配置' description='使用BeautifulSoup或lxml等HTML解析库解析获取到的HTML源代码' dependencies=['step1']
3. id='step3' title='数据提取实现' description='从解析后的HTML中提取所需信息，如搜索结果、链接等' dependencies=['step2']
4. id='step4' title='数据存储功能' description='将提取的数据保存为JSON或其他格式的文件' dependencies=['step3']
5. id='step5' title='错误处理和日志' description='为爬虫添加异常处理和日志记录功能，确保稳定运行' dependencies=['step1']

stderr:
'''