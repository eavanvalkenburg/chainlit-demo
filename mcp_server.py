# /// script # noqa: CPY001
# dependencies = [
#   "semantic-kernel[mcp]",
# ]
# ///
# Copyright (c) Microsoft. All rights reserved.
import uvicorn
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route
import logging
import nest_asyncio

from agents import create_agents


logger = logging.getLogger(__name__)


async def run(port: int) -> None:
    nest_asyncio.apply()
    pa_agent = await create_agents()

    server = pa_agent.as_mcp_server()

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as (
            read_stream,
            write_stream,
        ):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )

    starlette_app = Starlette(
        debug=True,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )
    try:
        uvicorn.run(starlette_app, host="0.0.0.0", port=port)  # nosec
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run(port=8000))
