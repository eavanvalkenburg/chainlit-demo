import chainlit as cl
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
)
from semantic_kernel.functions import kernel_function


# Example Native Plugin (Tool)
class WeatherPlugin:
    @kernel_function(name="get_weather", description="Gets the weather for a city")
    def get_weather(self, city: str) -> str:
        """Retrieves the weather for a given city."""
        if "nieuwegein" in city.lower():
            return f"The weather in {city} is 20°C and sunny."
        elif "seattle" in city.lower():
            return f"The weather in {city} is 15°C and cloudy."
        else:
            return f"Sorry, I don't have the weather for {city}."


@cl.on_chat_start
async def on_chat_start():
    agent = ChatCompletionAgent(
        name="WeatherAgent",
        service=OpenAIChatCompletion(),
        plugins=[WeatherPlugin()],
        instructions="You are a helpful assistant that provides weather information.",
    )
    cl.SemanticKernelFilter(kernel=agent.kernel)
    cl.user_session.set("agent", agent)


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: sk.Kernel
    thread = cl.user_session.get("thread", None)  # type: ChatCompletionAgentThread

    # Create a Chainlit message for the response stream
    answer = cl.Message(content="")
    async for response in agent.invoke_stream(messages=message.content, thread=thread):
        if response.content and response.content.content:
            await answer.stream_token(response.content.content)
        thread = response.thread
    await answer.update()
    cl.user_session.set("thread", thread)
    # Send the final message
    await answer.send()
