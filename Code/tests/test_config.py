"""Tests for configuration and logging setup (Phase 1)."""

import os
from pathlib import Path
import pytest
from config.settings import Settings, get_settings
from core.logger import get_logger


def test_settings_default_values():
    """Verify that default settings load with expected values and types."""
    settings = Settings()

    assert settings.app_name == "AUTOSAR HLD AI Assistant"
    assert settings.default_llm_provider in ["mock", "ollama", "gemini", "openai"]
    assert settings.embedding_model_name == "all-MiniLM-L6-v2"
    assert settings.api_port == 8000
    assert settings.frontend_port == 8501
    assert isinstance(settings.data_dir, Path)
    assert isinstance(settings.documents_dir, Path)
    assert isinstance(settings.chroma_dir, Path)
    assert isinstance(settings.sqlite_db_path, Path)


def test_settings_directory_auto_creation(tmp_path: Path):
    """Verify that ensure_directories creates all expected directories."""
    custom_data_dir = tmp_path / "custom_data"
    custom_docs_dir = custom_data_dir / "documents"
    custom_chroma_dir = custom_data_dir / "chroma_db"
    custom_sqlite_path = custom_data_dir / "db" / "assistant.db"

    settings = Settings(
        data_dir=custom_data_dir,
        documents_dir=custom_docs_dir,
        chroma_dir=custom_chroma_dir,
        sqlite_db_path=custom_sqlite_path,
    )

    assert not custom_data_dir.exists()
    assert not custom_docs_dir.exists()
    assert not custom_chroma_dir.exists()

    settings.ensure_directories()

    assert custom_data_dir.exists() and custom_data_dir.is_dir()
    assert custom_docs_dir.exists() and custom_docs_dir.is_dir()
    assert custom_chroma_dir.exists() and custom_chroma_dir.is_dir()
    assert custom_sqlite_path.parent.exists() and custom_sqlite_path.parent.is_dir()


def test_settings_environment_override(monkeypatch: pytest.MonkeyPatch):
    """Verify that environment variables take precedence over default settings."""
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("API_PORT", "9090")
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "ollama")

    settings = Settings()

    assert settings.app_env == "testing"
    assert settings.log_level == "DEBUG"
    assert settings.api_port == 9090
    assert settings.default_llm_provider == "ollama"


def test_get_settings_caching():
    """Verify that get_settings returns a valid singleton Settings instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_logger_initialization():
    """Verify that get_logger produces a properly configured logger instance."""
    logger = get_logger("test_module", log_level="DEBUG")
    assert logger.name == "test_module"
    assert len(logger.handlers) >= 1
    assert logger.level == 10  # logging.DEBUG is 10
