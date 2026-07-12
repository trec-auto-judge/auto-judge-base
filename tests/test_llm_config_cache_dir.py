"""Regression: unset CACHE_DIR must yield cache_dir=None, not Path('.')."""

from pathlib import Path

from autojudge_base.llm_config import LlmConfigBase


def test_unset_cache_dir_is_none(monkeypatch):
    monkeypatch.delenv("CACHE_DIR", raising=False)
    monkeypatch.delenv("LLM_CACHE_DIR", raising=False)
    assert LlmConfigBase.from_env().cache_dir is None


def test_empty_cache_dir_is_none(monkeypatch):
    monkeypatch.setenv("CACHE_DIR", "")
    monkeypatch.delenv("LLM_CACHE_DIR", raising=False)
    assert LlmConfigBase.from_env().cache_dir is None


def test_set_cache_dir_roundtrips(monkeypatch):
    monkeypatch.setenv("CACHE_DIR", "./cache")
    assert LlmConfigBase.from_env().cache_dir == Path("./cache")


def test_llm_cache_dir_fallback(monkeypatch):
    monkeypatch.delenv("CACHE_DIR", raising=False)
    monkeypatch.setenv("LLM_CACHE_DIR", "/tmp/x")
    assert LlmConfigBase.from_env().cache_dir == Path("/tmp/x")


def test_base_url_accepts_litellm_convention(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_BASE", "http://gpu:3001/v1")
    assert LlmConfigBase.from_env().base_url == "http://gpu:3001/v1"


def test_base_url_prefers_canonical_name(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://canonical/v1")
    monkeypatch.setenv("OPENAI_API_BASE", "http://legacy/v1")
    assert LlmConfigBase.from_env().base_url == "http://canonical/v1"
