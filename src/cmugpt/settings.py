"""Configuration read from environment variables.

get_settings() builds a new Settings object on every call rather than caching
one at import time. Tests rely on this to set variables per test, and it lets
a rotated OPENROUTER_API_KEY or AGENT_SHARED_SECRET take effect without a
restart.
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Loads .env (searched upward from this package) without overriding variables
# that are already set. CI depends on that precedence: it sets DATABASE_URL=""
# to select the in-memory store.
load_dotenv()

_PRODUCTION_VALUES = {"prod", "production"}


class Settings(BaseSettings):
    """Field names map to environment variables of the same name in upper case."""

    model_config = SettingsConfigDict(extra="ignore")

    # OpenRouter serves chat, memory extraction, and chat titles.
    openrouter_api_key: str = ""
    # Base URL of the CMU MCP server that publishes the campus tools.
    mcp_server_url: str = ""
    # An OpenAI key, distinct from the OpenRouter key. Used for
    # text-embedding-3-large (memory search) and the moderation endpoint.
    # When unset, recall falls back to recency order and moderation is skipped.
    openai_api_key: str = ""
    # Bearer token the Surface server presents on every request. Empty disables
    # authentication, which is acceptable only in local development.
    agent_shared_secret: str = ""
    # Postgres with pgvector. Empty means the in-memory store.
    database_url: str = ""
    # Comma-separated browser origins allowed by CORS.
    allowed_origins: str = "https://cmugpt.com"
    title_model: str = "qwen/qwen3.7-flash"
    memory_extraction_model: str = "qwen/qwen3.7-flash"
    # SQLite file that stores the per-user daily token budget.
    token_usage_db: str = "/tmp/cmugpt_token_usage.sqlite3"
    port: int = 5000
    # Any of these set to prod or production marks a production deployment.
    agent_env: str = ""
    app_env: str = ""
    environment: str = ""
    secretspec_profile: str = ""

    @property
    def is_production(self) -> bool:
        flags = (
            self.agent_env,
            self.app_env,
            self.environment,
            self.secretspec_profile,
        )
        return any(flag.strip().lower() in _PRODUCTION_VALUES for flag in flags)

    @property
    def allowed_origin_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


def get_settings() -> Settings:
    """Return a Settings object built from the current environment."""
    return Settings()
