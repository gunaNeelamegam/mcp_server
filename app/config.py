"""Single source of truth for which plug-ins are active and how to reach them.

Every other module reads settings from here (via `get_settings()`) instead of
touching `os.environ` directly, and instead of hardcoding provider choices.
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Active plug-ins
    chat_model_provider: Literal["ollama", "claude"] = "ollama"
    embedding_provider: Literal["ollama"] = "ollama"
    search_strategy: Literal["dense", "keyword", "hybrid"] = "hybrid"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"

    # Claude
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"

    # Vector DB
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # Session store
    redis_url: str = "redis://localhost:6379/0"

    # MCP servers
    shipments_mcp_url: str = "http://localhost:7001"
    inbound_mcp_url: str = "http://localhost:7002"
    outbound_mcp_url: str = "http://localhost:7003"
    inventory_mcp_url: str = "http://localhost:7004"
    returns_mcp_url: str = "http://localhost:7005"

    # App
    log_level: str = "INFO"
    request_timeout_seconds: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()
