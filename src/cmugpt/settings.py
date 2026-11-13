"""Configuration read from environment variables.

get_settings() builds a new Settings object on every call rather than caching
one at import time. Tests rely on this to set variables per test, and it lets
a rotated OPENROUTER_API_KEY or AGENT_SHARED_SECRET take effect without a
restart.
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Loads .env (searched upward from this package) without overriding variables that are already set. CI depends on that precedence: it sets DATABASE_URL="" to select the in-memory store.
load_dotenv()

_PRODUCTION_VALUES = {"prod", "production"}


class Settings(BaseSettings):
    """Field names map to environment variables of the same name in upper case."""

    model_config = SettingsConfigDict(extra="ignore")

    # OpenRouter serves chat, memory extraction, and chat titles.
    openrouter_api_key: str = ""
    # Base URL of the CMU MCP server that publishes the campus tools.
    mcp_server_url: str = ""
    # An OpenAI key, distinct from the OpenRouter key. Used for text-embedding-3-large (memory search) and the moderation endpoint.
    # When unset, recall falls back to recency order and moderation is skipped.
    openai_api_key: str = ""
    # Bearer token the Surface server presents on every request. When empty,
    # the protected routes refuse every request unless
    # AGENT_ALLOW_UNAUTHENTICATED is set.
    agent_shared_secret: str = ""
    # Opt-in for running without AGENT_SHARED_SECRET, for local development only. Production refuses to start with it set.
    agent_allow_unauthenticated: bool = False
    # Postgres with pgvector. Empty means the in-memory store.
    database_url: str = ""
    # Comma-separated browser origins allowed by CORS.
    allowed_origins: str = "https://cmugpt.com"
    # Comma-separated OpenRouter models a caller may request. Requests naming any other model are rejected, so a caller cannot choose an expensive model. Keep in sync with AGENT_MODELS in the Surface.
    allowed_models: str = (
        "openai/gpt-5.6-luna,deepseek/deepseek-v4-pro,minimax/minimax-m3,"
        "z-ai/glm-5.2,google/gemini-3.5-flash-lite"
    )
    title_model: str = "qwen/qwen3.7-flash"
    memory_extraction_model: str = "qwen/qwen3.7-flash"
    # SQLite file that stores the per-user daily token budget.
    token_usage_db: str = "/tmp/cmugpt_token_usage.sqlite3"
    port: int = 5055
    # "production" or "prod" makes startup fail without DATABASE_URL and an
    # AGENT_SHARED_SECRET of at least 32 characters. The prod profile of secretspec.toml sets it.
    agent_env: str = ""

    @property
    def is_production(self) -> bool:
        return self.agent_env.strip().lower() in _PRODUCTION_VALUES

    @property
    def allowed_model_list(self) -> list[str]:
        return [m.strip() for m in self.allowed_models.split(",") if m.strip()]

    @property
    def allowed_origin_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


def get_settings() -> Settings:
    """Return a Settings object built from the current environment."""
    return Settings()
