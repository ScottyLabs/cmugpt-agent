from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ActionType(StrEnum):
    """Types of actions the agent can take"""

    QUERY = "query"
    RETRIEVE = "retrieve"
    SEARCH = "search"
    COMPUTE = "compute"
    RESPOND = "respond"


class ToolCall(BaseModel):
    """Represents a tool/MCP call made by the agent"""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(..., description="Name of the MCP tool called")
    parameters: BaseModel | None = Field(default=None, description="Tool parameters")
    result: str | int | float | bool | list[str | int | float | bool] | None = Field(
        default=None, description="Result from the tool call"
    )


class Thought(BaseModel):
    """Represents the agent's reasoning"""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(..., description="The agent's thought process")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence level (0-1)"
    )


class Metadata(BaseModel):
    """Additional context metadata"""

    model_config = ConfigDict(extra="forbid")

    pass  # Add specific fields as needed


class AgentResponse(BaseModel):
    """Structured response from the CMU-GPT Agent"""

    model_config = ConfigDict(extra="forbid")

    thought: Thought = Field(..., description="Agent's reasoning and confidence")
    action: ActionType = Field(..., description="Type of action taken")
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="MCP tools called"
    )
    response_text: str = Field(..., description="Final response to user")
    metadata: Metadata = Field(
        default_factory=Metadata, description="Additional context"
    )


class UserInput(BaseModel):
    """Input from the Surface to the Agent"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., description="User's question or request")
    context: dict[str, str] | None = Field(default=None, description="Optional context")
    user_id: str | None = Field(default=None, description="Identifier for the user")
