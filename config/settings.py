"""Configuration management for AUTOSAR HLD AI Assistant.

Uses Pydantic BaseSettings to load and validate typed configuration from
environment variables and .env files.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable overrides and validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application metadata
    app_env: Literal["development", "testing", "production"] = "development"
    app_name: str = "AUTOSAR HLD AI Assistant"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # Base directory paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("./data"))
    documents_dir: Path = Field(default_factory=lambda: Path("./data/documents"))
    chroma_dir: Path = Field(default_factory=lambda: Path("./data/chroma_db"))
    sqlite_db_path: Path = Field(default_factory=lambda: Path("./data/autosar_assistant.db"))

    # Embeddings configuration
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # LLM Provider configuration
    default_llm_provider: Literal["mock", "ollama", "gemini", "openai"] = "mock"

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3:8b"

    # Gemini settings
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"

    # OpenAI settings
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Networking settings
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    frontend_port: int = 8501

    @field_validator("data_dir", "documents_dir", "chroma_dir", "sqlite_db_path", mode="before")
    @classmethod
    def parse_path(cls, value: str | Path) -> Path:
        """Convert string path to Path object."""
        if isinstance(value, str):
            return Path(value)
        return value

    def ensure_directories(self) -> None:
        """Ensure all required runtime data and cache directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        # Ensure parent directory for SQLite DB exists
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve cached singleton Settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
