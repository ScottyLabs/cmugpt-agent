"""The maps_show_map tool, the model's channel for map decisions.

The model decides when a map belongs on the answer and which places it shows
by calling this tool. The argument schema is an enum of catalog codes, so a
nonexistent place fails validation before reaching the guard, which
revalidates anyway. The tool returns a confirmation string only. The guard in
agent/cmu_maps.py builds the URL and payload, never the model.

Kept separate from agent/cmu_maps.py so that module remains framework-free.
"""

from enum import StrEnum

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from .buildings import LOCATION_ID_TO_LABEL
from .cmu_maps import SHOW_MAP_TOOL_NAME

# Codes like 2SC are not valid Python identifiers, so member names carry a
# prefix. Only the bare code values appear in the JSON schema and in the args
# the model sends.
BuildingCode = StrEnum(
    "BuildingCode",
    {f"B_{code}": code for code in sorted(LOCATION_ID_TO_LABEL)},
)


class ShowMapArgs(BaseModel):
    """Arguments for maps_show_map."""

    model_config = ConfigDict(extra="forbid")

    # The runtime-built enum is opaque to ty, which sees a variable rather
    # than a type. Pydantic and langchain resolve it and emit a plain schema
    # enum of the codes.
    destination: BuildingCode = Field(  # ty: ignore[invalid-type-form]
        ...,
        description="Building the user asked about; the map centers on it.",
    )
    origin: BuildingCode | None = Field(  # ty: ignore[invalid-type-form]
        default=None,
        description=(
            "Starting point of a walking route. Set ONLY when the user stated "
            "or clearly implied where they are starting from (possibly in an "
            "earlier message); otherwise omit it."
        ),
    )


def _label(code: str) -> str:
    return LOCATION_ID_TO_LABEL.get(code, code)


def _show_map(destination: str, origin: str | None = None) -> str:
    # StrEnum members are str, so str annotations stay truthful for the
    # validated values langchain passes in.
    dest = str(destination)
    src = str(origin) if origin is not None else None
    if src and src != dest:
        return (
            f"Map attached: route from {_label(src)} ({src}) "
            f"to {_label(dest)} ({dest})."
        )
    return f"Map attached: {_label(dest)} ({dest})."


def build_show_map_tool() -> StructuredTool:
    """The tool object to bind alongside the MCP tools when maps are enabled."""
    return StructuredTool.from_function(
        func=_show_map,
        name=SHOW_MAP_TOOL_NAME,
        description=(
            "Attach the interactive campus map to your answer: one building, "
            "or a walking route when `origin` is also set. Call it whenever "
            "the user asks where something is, how to get somewhere, or what "
            "is near a place. It only displays the map - it returns no data, "
            "so still call data tools for hours, menus, rooms, or courses."
        ),
        args_schema=ShowMapArgs,
    )
