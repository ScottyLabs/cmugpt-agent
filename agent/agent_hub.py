import os
from typing import List, Dict, Any
from dedalus_labs import AsyncDedalus, DedalusRunner
from schema import AgentResponse, UserInput
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

# Initialize the Dedalus client
client = AsyncDedalus(api_key=os.getenv("DEDALUS_API_KEY", ""))


async def run_agent(
    user_input: UserInput,
    mcp_servers: List[str] = ["cmugpt-mcp-server"],
    model: str = "openai/gpt-4o",
    message_history: List[Dict[str, str]] = None,
) -> AgentResponse:
    """
    Runs the Dedalus Agent with structured input/output.

    Args:
        user_input: UserInput object containing the query and optional context.
        mcp_servers: List of MCP server identifiers for tool access.
        model: The base model to use via Dedalus.
        message_history: Optional conversation history for multi-turn interactions.

    Returns:
        AgentResponse: Structured response with thought, action, and response_text.
    """
    runner = DedalusRunner(client)

    # Build messages list with history
    messages = message_history or []
    messages.append({"role": "user", "content": user_input.query})

    response = await runner.run(
        input=messages,
        # no need to use messages= parameter because input is a List[Dict],
        # so will include conversation history
        model=model,
        mcp_servers=mcp_servers,
        response_format=AgentResponse,
        instructions=(
            "You are the CMUGPT Agent. Assist users with CMU campus information. "
            "Use provided MCP tools to query data. Always return structured JSON "
            "with thought, action, tool_calls, response_text, and metadata fields."
        ),
    )

    # Parse string response back to AgentResponse model
    if isinstance(response.final_output, str):
        parsed = json.loads(response.final_output)
        return AgentResponse(**parsed)

    return response.final_output
