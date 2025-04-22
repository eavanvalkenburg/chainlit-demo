import os
import chainlit as cl

from azure.ai.projects.models import CodeInterpreterTool
from azure.identity.aio import DefaultAzureCredential

from semantic_kernel.agents import (
    ChatCompletionAgent,
    AzureAIAgent,
    AzureAIAgentSettings,
)
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
)
from semantic_kernel.connectors.mcp import MCPStdioPlugin
from semantic_kernel.core_plugins.time_plugin import TimePlugin


@cl.on_chat_start
async def on_chat_start():
    github_plugin = MCPStdioPlugin(
        name="Github",
        description="Github Plugin",
        command="docker",
        args=[
            "run",
            "-i",
            "--rm",
            "-e",
            "GITHUB_PERSONAL_ACCESS_TOKEN",
            "ghcr.io/github/github-mcp-server",
        ],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")},
    )
    await github_plugin.connect()
    file_plugin = MCPStdioPlugin(
        name="FileViewer",
        description="File Viewer Plugin",
        command="docker",
        args=[
            "run",
            "-i",
            "--rm",
            "--mount",
            "type=bind,src=C:/Work/sk/semantic-kernel/python,dst=/projects,ro",
            "mcp/filesystem",
            "/projects",
        ],
    )
    await file_plugin.connect()
    creds = DefaultAzureCredential()
    creds = await creds.__aenter__()
    client = AzureAIAgent.create_client(credential=creds)
    client = await client.__aenter__()
    cl.user_session.set("client", client)
    # 1. Create an agent with a code interpreter on the Azure AI agent service
    code_interpreter = CodeInterpreterTool()
    agent_definition = await client.agents.create_agent(
        model=AzureAIAgentSettings().model_deployment_name,
        tools=code_interpreter.definitions,
        tool_resources=code_interpreter.resources,
        instructions="You are a helpful assistant that helps the user identify issues, and looks for the relevant files in the repository. "
        "You are focused on Microsoft Semantic Kernel and always use the `python` tag in addition to other tags if needed.",
    )

    # 2. Create a Semantic Kernel agent for the Azure AI agent
    code_agent = AzureAIAgent(
        client=client,
        definition=agent_definition,
        plugins=[file_plugin],
    )
    cl.user_session.set("code_agent", code_agent)
    github_agent = ChatCompletionAgent(
        name="GithubAgent",
        service=OpenAIChatCompletion(),
        plugins=[github_plugin],
        instructions="You are a helpful assistant that helps with github related tasks. You are focused on "
        "Microsoft Semantic Kernel and always use the `python` tag in addition to other tags if needed.",
    )
    pa_agent = ChatCompletionAgent(
        name="PersonalAssistant",
        service=OpenAIChatCompletion(),
        plugins=[github_agent, code_agent, TimePlugin()],
        instructions="You are a helpful assistant that helps me with all manner of tasks, you have access to your own assistants.",
    )
    cl.SemanticKernelFilter(kernel=pa_agent.kernel)
    cl.user_session.set("agent", pa_agent)


@cl.on_chat_end
async def on_chat_end():
    client = cl.user_session.get("client")
    code_agent = cl.user_session.get("code_agent")
    await client.agents.delete_agent(code_agent.definition.id)
    await client.__aexit__(None, None, None)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Latest issues",
            message="What are the latest python issues in Microsoft Semantic Kernel?",
        ),
        cl.Starter(
            label="Are there any issues with the labels `python` and `triage`?",
            message="Explain superconductors like I'm five years old.",
        ),
        cl.Starter(
            label="Fix the latest python issue",
            message="Take the latest python issue and suggest a fix.",
        ),
    ]


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


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
