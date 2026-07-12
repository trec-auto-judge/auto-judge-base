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
