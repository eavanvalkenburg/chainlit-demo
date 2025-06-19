import os
from pathlib import Path

from semantic_kernel.agents import (
    ChatCompletionAgent,
)
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAITextEmbedding,
)
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
from semantic_kernel.connectors.mcp import MCPStdioPlugin
from semantic_kernel.core_plugins.time_plugin import TimePlugin
from semantic_kernel.connectors.memory import ChromaCollection
from semantic_kernel.functions import KernelPlugin

from data.parse import DocsEntries


async def create_agents():
    # File Plugin
    file_plugin = MCPStdioPlugin(
        name="FileViewer",
        description="File Viewer Plugin",
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "C:/Work/sk/semantic-kernel/python",
        ],
    )
    await file_plugin.connect()
    chroma: ChromaCollection[str, DocsEntries] = ChromaCollection(
        collection_name="docs",
        data_model_type=DocsEntries,
        embedding_generator=OpenAITextEmbedding(),
        persist_directory=str(Path.cwd() / "data" / "chroma"),
    )
    text_search = chroma.as_text_search()
    func = text_search.create_search(
        function_name="DocsSearch",
        description="Searches the Semantic Kernel docs for relevant information",
        top=2,
        vector_property_name="embedding",
    )
    docs_agent = ChatCompletionAgent(
        name="DocsAgent",
        service=OllamaChatCompletion(),
        plugins=[
            KernelPlugin(
                name="Docs",
                functions=[func],
            ),
            file_plugin,
        ],
        instructions="You are a helpful assistant that helps with documentation related tasks. You are focused on "
        "Microsoft Semantic Kernel and you can refer to the documentation for help. Always use that instead of trying to guess.",
    )

    github_plugin = MCPStdioPlugin(
        name="Github",
        description="Github Plugin",
        command="C:\\Work\\github-mcp-server\\github-mcp-server.exe",
        args=["stdio"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")},
    )
    await github_plugin.connect()
    github_agent = ChatCompletionAgent(
        name="GithubAgent",
        service=OpenAIChatCompletion(),
        plugins=[github_plugin],
        instructions="You are a helpful assistant that helps with github related tasks. You are focused on "
        "Microsoft Semantic Kernel and always use the `python` tag in addition to other tags if needed."
        "Multiple tags are allowed, so use that.",
    )

    # create the main agent
    pa_agent = ChatCompletionAgent(
        name="PersonalAssistant",
        service=OpenAIChatCompletion(),
        plugins=[github_agent, docs_agent, TimePlugin()],
        instructions="You are a helpful assistant that helps me with all manner of tasks related to Semantic Kernel (or SK), you have access to your own assistants.",
    )
    return pa_agent


async def main():
    # Github Plugin and Agent
    agent = await create_agents()

    thread = None
    first = True
    message = "how do the docs define Process Framework?"
    while True:
        # ask for input
        if first:
            first = False
        else:
            message = input("What do you want to ask? ")
        if message.lower() == "exit":
            break

        # call the pa_agent with the message
        answer = await agent.get_response(messages=message, thread=thread)
        print(answer.content)
        thread = answer.thread


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
