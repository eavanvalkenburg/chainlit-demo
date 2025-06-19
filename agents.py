from contextlib import asynccontextmanager
from azure.identity.aio import DefaultAzureCredential

from semantic_kernel.agents import (
    AgentRegistry,
    AzureAIAgent,
    AzureAIAgentSettings,
    OpenAIResponsesAgent,
    MagenticOrchestration,
    StandardMagenticManager,
)
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatMessageContent


# Define the YAML string for the sample
web_search_agent_spec = """
type: openai_responses
name: WebSearchAgent
description: Agent with web search tool, use this to gather recent information.
instructions: >
  Find answers to the user's questions using the provided tool.
model:
  id: ${OpenAI:ChatModelId}
  connection:
    api_key: ${OpenAI:ApiKey}
tools:
  - type: web_search
    description: Search the internet for recent information.
    options:
      search_context_size: high
"""
code_agent_spec = """
type: foundry_agent
name: CodeInterpreterAgent
description: Agent with code interpreter tool, use this to run code, for instance to analyze data.
instructions: >
  Use the code interpreter tool to answer questions that require code to be generated
  and executed.
model:
  id: ${AzureAI:ChatModelId}
  connection:
    endpoint: ${AzureAI:Endpoint}
tools:
  - type: code_interpreter
"""
settings = AzureAIAgentSettings()  # ChatModelId & Endpoint come from env vars


@asynccontextmanager
async def create_agents():
    # web search agent
    client = OpenAIResponsesAgent.create_client()

    # Create the Responses Agent from the YAML spec
    # Note: the extras can be provided in the short-format (shown below) or
    # in the long-format (as shown in the YAML spec, with the `OpenAI:` prefix).
    # The short-format is used here for brevity
    web_search_agent: OpenAIResponsesAgent = await AgentRegistry.create_from_yaml(
        yaml_str=web_search_agent_spec,
        client=client,
    )
    async with (
        DefaultAzureCredential() as creds,
        AzureAIAgent.create_client(credential=creds) as client,
    ):
        try:
            # Create the AzureAI Agent from the YAML spec
            # Note: the extras can be provided in the short-format (shown below) or
            # in the long-format (as shown in the YAML spec, with the `AzureAI:` prefix).
            # The short-format is used here for brevity
            code_agent: AzureAIAgent = await AgentRegistry.create_from_yaml(
                yaml_str=code_agent_spec,
                client=client,
                settings=settings,
            )

            yield [web_search_agent, code_agent]

        finally:
            # Cleanup: Delete the thread and agent
            if code_agent:
                await client.agents.delete_agent(code_agent.id)


def agent_response_callback(message: ChatMessageContent) -> None:
    """Observer function to print the messages from the agents."""
    print(f"\033[94m**{message.name}**\n{message.content}\033[0m")


async def main():
    # Github Plugin and Agent
    async with create_agents() as agents:
        magentic_orchestration = MagenticOrchestration(
            members=agents,
            manager=StandardMagenticManager(
                chat_completion_service=AzureChatCompletion()
            ),
            agent_response_callback=agent_response_callback,
        )

        # 2. Create a runtime and start it
        runtime = InProcessRuntime()
        runtime.start()

        try:
            # 3. Invoke the orchestration with a task and the runtime
            orchestration_result = await magentic_orchestration.invoke(
                task=(
                    "I am preparing a report on the my customer, they a large bank based in France."
                    "They are called Societe Generale. "
                    "Please prepare a report on their IT strategy and especially their use of Gen AI, "
                    "I'm most interested in their use of Agent and code based solutions."
                    "Also include their latest financial results, in basic tables and simple terms, but no charts."
                ),
                runtime=runtime,
            )
            # 4. Wait for the results
            value = await orchestration_result.get()
            print(f"\033[92m\nFinal result:\n{value}\033[0m")
        except Exception as e:
            print(f"An error occurred during orchestration: {e}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
