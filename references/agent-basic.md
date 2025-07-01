Hello world example
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.

Handoffs example
from agents import Agent, Runner
import asyncio

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
)


async def main():
    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print(result.final_output)
    # ¡Hola! Estoy bien, gracias por preguntar. ¿Y tú, cómo estás?


if __name__ == "__main__":
    asyncio.run(main())

Functions example
import asyncio

from agents import Agent, Runner, function_tool


@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."


agent = Agent(
    name="Hello world",
    instructions="You are a helpful agent.",
    tools=[get_weather],
)


async def main():
    result = await Runner.run(agent, input="What's the weather in Tokyo?")
    print(result.final_output)
    # The weather in Tokyo is sunny.


if __name__ == "__main__":
    asyncio.run(main())

The agent loop
When you call Runner.run(), we run a loop until we get a final output.

We call the LLM, using the model and settings on the agent, and the message history.
The LLM returns a response, which may include tool calls.
If the response has a final output (see below for more on this), we return it and end the loop.
If the response has a handoff, we set the agent to the new agent and go back to step 1.
We process the tool calls (if any) and append the tool responses messages. Then we go to step 1.
There is a max_turns parameter that you can use to limit the number of times the loop executes.

Final output
Final output is the last thing the agent produces in the loop.

If you set an output_type on the agent, the final output is when the LLM returns something of that type. We use structured outputs for this.
If there's no output_type (i.e. plain text responses), then the first LLM response without any tool calls or handoffs is considered as the final output.
As a result, the mental model for the agent loop is:

If the current agent has an output_type, the loop runs until the agent produces structured output matching that type.
If the current agent does not have an output_type, the loop runs until the current agent produces a message without any tool calls/handoffs.

# openrouter agent example
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from dotenv import load_dotenv

load_dotenv()

 配置 OpenRouter 的信息
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

# 加载 agent 配置（指令/提示词）
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'agent_config.json')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    agent_config = json.load(f)

# 目标解析模块
class GoalInterpreterOutput(BaseModel):
    task: str
    input: str
    output: str
    modules: List[str]
    constraints: List[str]

goal_interpreter_agent = Agent(
    name="GoalInterpreterAgent",
    instructions=agent_config["GoalInterpreterAgent"],
    output_type=GoalInterpreterOutput,
    model=model
)

# enable logging
from agents import Runner, enable_verbose_stdout_logging, set_tracing_disabled
set_tracing_disabled(True)
enable_verbose_stdout_logging()