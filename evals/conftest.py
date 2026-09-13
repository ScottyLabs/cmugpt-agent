"""Live evaluations call OpenRouter and the CMU MCP server, so they need real
keys and incur API costs. The default `uv run pytest` collects only
tests/unit. Run these with `uv run pytest evals`. When the keys are absent,
every test here is skipped.
"""

import pytest

from cmugpt.settings import get_settings


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    settings = get_settings()
    needed = {
        "OPENROUTER_API_KEY": settings.openrouter_api_key,
        "MCP_SERVER_URL": settings.mcp_server_url,
    }
    missing = [name for name, value in needed.items() if not value]
    if not missing:
        return
    skip = pytest.mark.skip(reason=f"live evals need {', '.join(missing)}")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
