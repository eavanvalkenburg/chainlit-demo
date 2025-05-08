import logging
import chainlit as cl


from agents import create_agents


logging.basicConfig(level=logging.warning)


@cl.on_chat_start
async def on_chat_start():
    # Github Plugin and Agent
    pa_agent = await create_agents()
    cl.user_session.set("agent", pa_agent)
    cl.SemanticKernelFilter(kernel=pa_agent.kernel)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Latest issues",
            message="What are the latest python issues in Microsoft Semantic Kernel?",
        ),
        cl.Starter(
            label="Untriaged issues",
            message="Are there any issues with both labels `python` and `triage`?",
        ),
        cl.Starter(
            label="Fix the latest python issue",
            message="Take the latest python issue and suggest a fix.",
        ),
    ]


@cl.on_message
async def on_message(message: cl.Message):
    # retrieve the current agent and thread
    agent = cl.user_session.get("agent")
    thread = cl.user_session.get("thread", None)

    # Create a Chainlit message for the response stream
    answer = cl.Message(content="")
    async for response in agent.invoke_stream(messages=message.content, thread=thread):
        if response.content and response.content.content:
            await answer.stream_token(response.content.content)
        thread = response.thread
    # Update the thread in the user session
    await answer.update()
    # Set the thread in the user session
    cl.user_session.set("thread", thread)
    # Send the final message
    await answer.send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
