import logging
import asyncio
from enum import Enum
import os
from pathlib import Path
from typing import ClassVar, Literal
from uuid import uuid4
from azure.identity.aio import DefaultAzureCredential

from dotenv import load_dotenv
from markdown_pdf import MarkdownPdf, Section
from pydantic import BaseModel, Field
from semantic_kernel import Kernel
from semantic_kernel.agents import (
    AgentRegistry,
    AzureAIAgent,
    AzureAIAgentSettings,
    ChatCompletionAgent,
)
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.functions import kernel_function, KernelArguments
from semantic_kernel.processes import ProcessBuilder
from semantic_kernel.processes.kernel_process import (
    KernelProcessStep,
    KernelProcessStepContext,
    KernelProcessStepState,
    KernelProcessStateMetadata,
)
from semantic_kernel.processes.local_runtime.local_kernel_process import start

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

guid = str(uuid4())


def get_state_file_path() -> Path:
    """Get the path to the state file."""
    return Path(__file__).parent / "process_data" / f"process-{guid}.json"


# step 1: Use agent with search to find info
class ProcessState(BaseModel):
    customer: str = ""
    report_content: str = ""
    final_report: str = ""
    whiteboard: list[str] = Field(
        default_factory=list, description="Whiteboard for notes and code snippets"
    )

    @classmethod
    def read_state(cls):
        """Read the state from the file."""
        state_file = get_state_file_path()
        if not state_file.exists():
            return cls()
        with open(state_file, "r") as f:
            return cls.model_validate_json(f.read())

    def write_state(self):
        """Write the state to the file."""
        state_file = get_state_file_path()
        with open(state_file, "w") as f:
            f.write(self.model_dump_json(indent=4))


class CopyWriterResponses(BaseModel):
    """Model to hold the responses from the copywriter agent."""

    action: Literal["code", "search", "finished", "clarification"] = "finished"
    content: str = ""


class CopyWriterStep(KernelProcessStep[ProcessState]):
    """A step that uses an agent to generate a report."""

    WRITE_REPORT: ClassVar[str] = "write_report"
    agent: ChatCompletionAgent | None = None
    state: ProcessState | None = None

    class Events(Enum):
        """Events for the CopyWriterStep."""

        RequestCode = "requestCode"
        RequestSearch = "requestSearch"
        RequestClarification = "requestClarification"
        ReportComplete = "reportComplete"

    async def activate(self, state: KernelProcessStepState[ProcessState]):
        self.state = state.state

    async def load_agent(self):
        self.agent = await AgentRegistry.create_from_file(
            file_path=Path(__file__).parent / "agents" / "copy_writer_agent_spec.yaml",
            service=OpenAIChatCompletion(),
        )

    @kernel_function(name=WRITE_REPORT)
    async def generate_report(self, context: KernelProcessStepContext):
        """Generate a report based on the search results."""
        # Here we would use the code interpreter agent to generate a report
        # For simplicity, we will just simulate this with a placeholder response
        self.state = ProcessState.read_state()
        if self.agent is None:
            await self.load_agent()

        copy_writer_agent_response = await self.agent.get_response(
            self.state.customer,
            arguments=KernelArguments(
                customer=self.state.customer,
                whiteboard="\n".join(self.state.whiteboard),
                settings=OpenAIChatPromptExecutionSettings(
                    response_format=CopyWriterResponses
                ),
            ),
        )
        response = CopyWriterResponses.model_validate_json(
            copy_writer_agent_response.content.content
        )
        match response.action:
            case "code":
                # If the agent requests code generation, we would handle that here
                # For now, we will just log it
                await context.emit_event(
                    process_event=self.Events.RequestCode,
                    data=response.content,
                )
            case "search":
                # If the agent requests additional search, we would handle that here
                await context.emit_event(
                    process_event=self.Events.RequestSearch,
                    data=response.content,
                )
            case "finished":
                self.state.report_content = response.content
                self.state.write_state()
                await context.emit_event(
                    process_event=self.Events.ReportComplete,
                    data=response.content,
                )
            case "clarification":
                await context.emit_event(
                    process_event=self.Events.RequestClarification,
                    data=response.content,
                )


class SearchStep(KernelProcessStep[ProcessState]):
    """A step that uses an agent to perform a web search."""

    SEARCH: ClassVar[str] = "search"

    class Events(Enum):
        """Events for the SearchStep."""

        SearchComplete = "searchComplete"

    async def activate(self, state: KernelProcessStepState[ProcessState]):
        self.state = state.state

    @kernel_function(name=SEARCH)
    async def get_additional_info(
        self, question: str, context: KernelProcessStepContext
    ):
        """Do a search for more info."""
        self.state = ProcessState.read_state()

        async with (
            DefaultAzureCredential() as creds,
            AzureAIAgent.create_client(credential=creds) as client,
        ):
            load_dotenv()
            agent = await AgentRegistry.create_from_file(
                file_path=Path(__file__).parent
                / "agents"
                / "web_search_azure_agent_spec.yaml",
                client=client,
                extras={"AgentId": os.getenv("AZURE_AI_AGENT_BING")},
            )

            response = await agent.get_response(messages=question)
            self.state.whiteboard.append(
                f"Search request: {question}\n, Search response: {response.content.content}\n"
            )
            self.state.write_state()
            # Emit the user input event
            await context.emit_event(process_event=self.Events.SearchComplete)


# step 2: write a first report using a agent, with handoff pattern with code interpreter agent


class CodeGenerationStep(KernelProcessStep[ProcessState]):
    """A step that uses an agent to generate code."""

    CODE_GENERATION: ClassVar[str] = "code_generation"

    class Events(Enum):
        """Events for the CodeGenerationStep."""

        CoderComplete = "coderComplete"

    async def activate(self, state: KernelProcessStepState[ProcessState]):
        self.state = state.state

    @kernel_function(name=CODE_GENERATION)
    async def generate_code(self, content: str, context: KernelProcessStepContext):
        """Generate code based on the search results."""
        # Here we would use the code interpreter agent to generate code
        # For simplicity, we will just simulate this with a placeholder response
        self.state = ProcessState.read_state()
        settings = AzureAIAgentSettings()  # ChatModelId & Endpoint come from env vars
        async with (
            DefaultAzureCredential() as creds,
            AzureAIAgent.create_client(credential=creds) as client,
        ):
            # Create the AzureAI Agent from the YAML spec
            # Note: the extras can be provided in the short-format (shown below) or
            # in the long-format (as shown in the YAML spec, with the `AzureAI:` prefix).
            # The short-format is used here for brevity
            code_agent: AzureAIAgent = await AgentRegistry.create_from_file(
                file_path=Path(__file__).parent / "agents" / "code_agent_spec.yaml",
                client=client,
                settings=settings,
            )
            response = await code_agent.get_response(
                messages=f"Generate and execute code to analyze the following content: {content}"
            )

            self.state.whiteboard.append(
                f"Code request: {content}\n, Code response: {response.content.content}\n"
            )
            self.state.write_state()

            await context.emit_event(process_event=self.Events.CoderComplete)

            if code_agent:
                await client.agents.delete_agent(code_agent.id)
            if response.thread and response.thread.id:
                await client.agents.threads.delete(response.thread.id)


# step 3: generate final report and create pdf


class ReportEditorStep(KernelProcessStep[ProcessState]):
    """A step that generates the final report."""

    FINAL_REPORT: ClassVar[str] = "final_report"
    agent: ChatCompletionAgent | None = None

    class Events(Enum):
        """Events for the ReportEditorStep."""

        ReportGenerated = "reportGenerated"

    async def activate(self, state: KernelProcessStepState[ProcessState]):
        self.state = state.state

    async def load_agent(self):
        """Create and return the report writer agent."""
        self.agent = await AgentRegistry.create_from_file(
            file_path=Path(__file__).parent / "agents" / "editor_agent_spec.yaml",
            service=OpenAIChatCompletion(),
        )

    @kernel_function(name=FINAL_REPORT)
    async def generate_final_report(self, context: KernelProcessStepContext):
        """Generate the final report."""
        self.state = ProcessState.read_state()
        if self.agent is None:
            await self.load_agent()

        final_report = await self.agent.get_response(self.state.report_content)

        self.state.final_report = final_report.content.content
        self.state.write_state()

        print("Report generated!")

        # Emit the final report generated event
        await context.emit_event(process_event=self.Events.ReportGenerated)


class UserStep(KernelProcessStep[ProcessState]):
    """A step that waits for user approval before generating the final report."""

    USER_APPROVAL: ClassVar[str] = "user_approval"
    USER_CLARIFICATION: ClassVar[str] = "user_clarification"

    class Events(Enum):
        """Events for the UserStep."""

        ReportApproved = "reportApproved"
        RequestEdits = "requestEdits"
        ClarificationProvided = "clarificationProvided"

    async def activate(self, state: KernelProcessStepState[ProcessState]):
        self.state = state.state

    @kernel_function(name=USER_APPROVAL)
    async def wait_for_user_approval(self, context: KernelProcessStepContext):
        """Wait for user approval before proceeding."""
        self.state = ProcessState.read_state()
        print("Please review the report content:")
        print(self.state.report_content)
        user_input = input("Do you approve the report? (type `yes` to approve): ")
        if user_input.lower() == "yes":
            await context.emit_event(process_event=self.Events.ReportApproved)
        else:
            print(
                "Report not approved. Feedback provided, please make changes and try again."
            )
            self.state.customer += f" Please make changes to the report based on this feedback: {user_input}"
            self.state.write_state()
            await context.emit_event(process_event=self.Events.RequestEdits)

    @kernel_function(name=USER_CLARIFICATION)
    async def request_user_clarification(
        self, clarification: str, context: KernelProcessStepContext
    ):
        """Request clarification from the user."""
        self.state = ProcessState.read_state()
        print("User requested clarification:", clarification)
        user_input = input("Please provide clarification: ")
        self.state.whiteboard.append(
            f"Clarification Question: {clarification} User clarification: {user_input}\n"
        )
        self.state.write_state()
        await context.emit_event(process_event=self.Events.ClarificationProvided)


class PDFGenStep(KernelProcessStep[ProcessState]):
    """A step that generates a PDF from the final report."""

    PDF_GENERATION: ClassVar[str] = "pdf_generation"

    class Events(Enum):
        """Events for the PDFGenStep."""

        PDFGenerated = "pdfGenerated"

    async def activate(self, state: KernelProcessStepState[ProcessState]):
        self.state = state.state

    @kernel_function(name=PDF_GENERATION)
    async def generate_pdf(self, context: KernelProcessStepContext):
        """Generate a PDF from the final report."""
        self.state = ProcessState.read_state()
        state_file = get_state_file_path()
        pdf = MarkdownPdf(toc_level=2, optimize=True)
        pdf.add_section(Section(self.state.final_report))
        pdf.save(file_name=state_file.with_suffix(".pdf"))
        print("PDF generated!")
        print("You can find the PDF at:", state_file.with_suffix(".pdf"))
        await context.emit_event(process_event=self.Events.PDFGenerated)


# step 4: send report to customer


def create_process():
    process = ProcessBuilder(name="ReportGenerationProcess", version="reportWriter.v1")

    # Define the steps on the process builder based on their types, not concrete objects
    writer = process.add_step(CopyWriterStep, name="copyWriter", aliases=["writer"])
    search = process.add_step(SearchStep, name="webSearch", aliases=["search"])
    coder = process.add_step(
        CodeGenerationStep, name="codeGeneration", aliases=["coder"]
    )
    editor = process.add_step(ReportEditorStep, name="reportEditor", aliases=["editor"])
    user = process.add_step(UserStep, name="userStep", aliases=["user"])
    pdf_generator = process.add_step(
        PDFGenStep, name="pdfGenerator", aliases=["pdfGen"]
    )

    # Define the input event that starts the process and where to send it
    process.on_input_event(event_id="startProcess").send_event_to(target=writer)
    # Define the event that triggers the next step in the process
    # For the user step, send the user input to the response step
    # routes from the report generation step to the search step and code generation step
    # code
    writer.on_event(event_id=CopyWriterStep.Events.RequestCode).send_event_to(
        target=coder
    )
    # search
    writer.on_event(event_id=CopyWriterStep.Events.RequestSearch).send_event_to(
        target=search
    )
    # clarification
    writer.on_event(event_id=CopyWriterStep.Events.RequestClarification).send_event_to(
        target=user,
        function_name=UserStep.USER_CLARIFICATION,
    )
    # finished
    writer.on_event(event_id=CopyWriterStep.Events.ReportComplete).send_event_to(
        target=editor
    )

    # back to report generation step
    search.on_event(event_id=SearchStep.Events.SearchComplete).send_event_to(
        target=writer
    )
    coder.on_event(event_id=CodeGenerationStep.Events.CoderComplete).send_event_to(
        target=writer
    )
    user.on_event(event_id=UserStep.Events.ClarificationProvided).send_event_to(
        target=writer
    )

    # final report writing step
    editor.on_event(event_id=ReportEditorStep.Events.ReportGenerated).send_event_to(
        target=user, function_name=UserStep.USER_APPROVAL
    )

    # user approval step
    user.on_event(event_id=UserStep.Events.ReportApproved).send_event_to(
        target=pdf_generator
    )

    # PDF generation step
    pdf_generator.on_event(event_id=PDFGenStep.Events.PDFGenerated).stop_process()

    return process


async def run_process(customer: str | None = None):
    """Run the process to generate a report."""

    print(f"Process GUID: {guid}")

    process_builder = create_process()

    state_file = get_state_file_path()
    if not state_file.exists():
        with open(state_file, "w") as f:
            f.write("{}")
    if not customer:
        customer = "Societe Generale. They are a large bank based in France."
    process_state = ProcessState(customer=customer)
    process_state.write_state()

    print(f"Starting process with task: {customer}")
    # Start the process
    loaded_metadata = KernelProcessStateMetadata(
        type="Process",
        name="ReportGenerationProcess",
        id=guid,
        versionInfo="reportWriter.v1",
        state=process_state,
    )
    process = process_builder.build(state_metadata=loaded_metadata)
    process.state.state = process_state
    async with await start(
        process=process,
        kernel=Kernel(),
        initial_event="startProcess",
    ) as running_process:
        state = await running_process.get_state()
        process_state_metadata = state.to_process_state_metadata()
        with open(state_file.with_suffix(".out.json"), "w") as f:
            f.write(process_state_metadata.model_dump_json(indent=4, exclude_none=True))


if __name__ == "__main__":
    asyncio.run(run_process())
