from agents import Agent, Runner,trace
from openai import AsyncOpenAI
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from dotenv import load_dotenv
import os
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from pydantic import BaseModel
from agents import set_tracing_disabled
from agents import enable_verbose_stdout_logging


set_tracing_disabled(True)
enable_verbose_stdout_logging()

#===============================================
# 自定义openrouter模型
#===============================================
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

#===============================================
# 初始化 Agent
#===============================================

# 定义一个助手，用于回答问题
agent = Agent(
    name="DeepSeek助手",
    instructions="你是一个乐于助人的中文助手。",
    model=model
)

# 定义一个助手，用于判断请求是否是中文
class JudgeResult(BaseModel):
    is_chinese: bool
    reason: str

judge_agent = Agent(
    name="判断助手",
    instructions="你只会判断请求是否是中文。",
    output_type=JudgeResult,
    model=model
)

# 定义一个助手，用于生成故事大纲
story_outline_agent = Agent(
    name="故事大纲助手",
    instructions="根据用户输入，生成一个故事大纲。",
    model=model
)


# 定义一个助手，用于判断故事大纲的质量
class OutlineCheckerOutput(BaseModel):
    good_quality: bool
    is_scifi: bool


outline_checker_agent = Agent(
    name="故事大纲判断助手",
    instructions="阅读给定的大纲，判断其质量。同时，确定它是否是科幻故事。",
    output_type=OutlineCheckerOutput,
    model=model
)

# 定义一个助手，用于生成故事
story_agent = Agent(
    name="故事生成助手",
    instructions="根据给定的大纲，生成一个故事。",
    output_type=str,
    model=model
)

#===============================================
# 生成故事
#===============================================

async def GenerateStory():
    input_prompt = "写一个关于未来的科幻故事。"

    # 1. 生成大纲
    outline_result = await Runner.run(
        story_outline_agent,
        input_prompt,
    )
    print("大纲生成完成")

    # 2. 检查大纲
    outline_checker_result = await Runner.run(
        outline_checker_agent,
        outline_result.final_output,
    )

    # 3. 如果大纲质量不好或不是科幻故事，则停止
    assert isinstance(outline_checker_result.final_output, OutlineCheckerOutput)
    if not outline_checker_result.final_output.good_quality:
        print("大纲质量不好，停止生成。")
        exit(0)

    if not outline_checker_result.final_output.is_scifi:
        print("大纲不是科幻故事，停止生成。")
        exit(0)

    print("大纲质量好且是科幻故事，继续生成故事。")

    # 4. 生成故事
    story_result = await Runner.run(
        story_agent,
        outline_result.final_output,
    )
    print(f"故事: {story_result.final_output}")

async def main():
    await GenerateStory()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 


