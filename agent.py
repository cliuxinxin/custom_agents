from agents import Agent, Runner
from openai import AsyncOpenAI
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from dotenv import load_dotenv
import os

from agents import set_tracing_disabled

set_tracing_disabled(True)

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

# 初始化 Agent
agent = Agent(
    name="DeepSeek助手",
    instructions="你是一个乐于助人的中文助手。",
    model=model
)

async def main():
    result = await Runner.run(agent, input="用中文写一首关于春天的诗。")
    print(result.final_output)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 