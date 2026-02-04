"""
Unit tests for model preference resolution.

These tests verify the model resolution logic without making real LLM calls.
"""

import pytest
from textwrap import dedent

from trec_auto_judge.llm import MinimaLlmConfig
from trec_auto_judge.llm_resolver import (ModelPreferences, AvailableModels,  ModelResolver, ModelResolutionError)


class TestModelPreferences:
    """Tests for ModelPreferences dataclass."""

    def test_from_dict_with_preferences(self):
        data = {
            "model_preferences": ["gpt-4o", "gpt-4-turbo"],
            "on_no_match": "use_default",
        }
        prefs = ModelPreferences.from_dict(data)

        assert prefs.preferences == ("gpt-4o", "gpt-4-turbo")
        assert prefs.on_no_match == "use_default"

    def test_from_dict_empty(self):
        prefs = ModelPreferences.from_dict({})

        assert prefs.preferences == ()
        assert prefs.on_no_match == "use_default"

    def test_from_dict_defaults_on_no_match(self):
        data = {"model_preferences": ["model-a"]}
        prefs = ModelPreferences.from_dict(data)

        assert prefs.on_no_match == "use_default"

    def test_from_yaml(self, tmp_path):
        config_file = tmp_path / "llm-config.yml"
        config_file.write_text(
        dedent("""
            model_preferences:
            - "gpt-4o"
            - "claude-3-sonnet"
            on_no_match: "use_default"
        """))

        prefs = ModelPreferences.from_yaml(config_file)

        assert prefs.preferences == ("gpt-4o", "claude-3-sonnet")
        assert prefs.on_no_match == "use_default"

    def test_from_yaml_empty_file(self, tmp_path):
        config_file = tmp_path / "llm-config.yml"
        config_file.write_text("")

        prefs = ModelPreferences.from_yaml(config_file)

        assert prefs.preferences == ()


class TestAvailableModels:
    """Tests for AvailableModels dataclass."""

    def test_from_yaml(self, tmp_path):
        config_file = tmp_path / "available_models.yml"
        config_file.write_text(dedent("""\
            models:
              gpt-4o:
                base_url: "https://api.openai.com/v1"
                model_id: "gpt-4o-2024-08-06"
                api_key_env: "OPENAI_API_KEY"
                enabled: true
              disabled-model:
                base_url: "https://api.example.com/v1"
                model_id: "disabled"
                enabled: false
            default_model: "gpt-4o"
            aliases:
              gpt4: "gpt-4o"
            """))

        available = AvailableModels.from_yaml(config_file)

        assert "gpt-4o" in available.models
        assert "disabled-model" not in available.models
        assert "disabled-model" in available.disabled
        assert available.default_model == "gpt-4o"
        assert available.aliases["gpt4"] == "gpt-4o"

    def test_from_env_with_legacy_vars(self, monkeypatch):
        """Test fallback to OPENAI_MODEL/OPENAI_BASE_URL env vars."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
        monkeypatch.setenv("OPENAI_MODEL", "test-model")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        # Ensure no available_models.yml exists
        monkeypatch.setenv("AUTOJUDGE_AVAILABLE_MODELS", "")

        available = AvailableModels.from_env()

        assert "test-model" in available.models
        assert available.default_model == "test-model"
        config = available.models["test-model"]
        assert config.model == "test-model"
        assert config.base_url == "https://api.example.com/v1"

    def test_get_enabled_models(self):
        config1 = MinimaLlmConfig(base_url="https://a.com", model="model-a")
        config2 = MinimaLlmConfig(base_url="https://b.com", model="model-b")

        available = AvailableModels(
            models={"model-a": config1, "model-b": config2}
        )

        assert set(available.get_enabled_models()) == {"model-a", "model-b"}

    def test_resolve_alias(self):
        available = AvailableModels(
            models={},
            aliases={"gpt4": "gpt-4o", "claude": "claude-3-sonnet"}
        )

        assert available.resolve_alias("gpt4") == "gpt-4o"
        assert available.resolve_alias("claude") == "claude-3-sonnet"
        assert available.resolve_alias("unknown") == "unknown"


class TestModelResolver:
    """Tests for ModelResolver resolution logic."""

    @pytest.fixture
    def resolver_with_models(self):
        """Create resolver with test models."""
        config_a = MinimaLlmConfig(base_url="https://a.com/v1", model="model-a")
        config_b = MinimaLlmConfig(base_url="https://b.com/v1", model="model-b")
        config_c = MinimaLlmConfig(base_url="https://c.com/v1", model="model-c")

        available = AvailableModels(
            models={"model-a": config_a, "model-b": config_b, "model-c": config_c},
            default_model="model-c",
            aliases={"alias-a": "model-a"},
        )
        return ModelResolver(available=available)

    def test_resolve_first_available(self, resolver_with_models):
        """First preference that's available should be selected."""
        prefs = ModelPreferences(preferences=("model-a", "model-b"))

        config = resolver_with_models.resolve(prefs)

        assert config.model == "model-a"
        assert config.base_url == "https://a.com/v1"

    def test_resolve_skips_unavailable(self, resolver_with_models):
        """Unavailable models should be skipped."""
        prefs = ModelPreferences(preferences=("nonexistent", "model-b"))

        config = resolver_with_models.resolve(prefs)

        assert config.model == "model-b"

    def test_resolve_via_alias(self, resolver_with_models):
        """Aliases should resolve to canonical names."""
        prefs = ModelPreferences(preferences=("alias-a",))

        config = resolver_with_models.resolve(prefs)

        assert config.model == "model-a"

    def test_resolve_use_default_on_no_match(self, resolver_with_models):
        """When on_no_match='use_default', fall back to default model."""
        prefs = ModelPreferences(
            preferences=("nonexistent-1", "nonexistent-2"),
            on_no_match="use_default"
        )

        config = resolver_with_models.resolve(prefs)

        assert config.model == "model-c"  # default_model

    def test_resolve_error_on_no_match(self, resolver_with_models):
        """When on_no_match='error', raise ModelResolutionError."""
        prefs = ModelPreferences(
            preferences=("nonexistent-1", "nonexistent-2"),
            on_no_match="error"
        )

        with pytest.raises(ModelResolutionError) as exc_info:
            resolver_with_models.resolve(prefs)

        assert "No model available" in str(exc_info.value)
        assert "nonexistent-1" in str(exc_info.value)

    def test_resolve_empty_preferences_with_default(self, resolver_with_models):
        """Empty preferences with use_default should return default model."""
        prefs = ModelPreferences(preferences=(), on_no_match="use_default")

        config = resolver_with_models.resolve(prefs)

        assert config.model == "model-c"

    def test_resolve_empty_preferences_error(self, resolver_with_models):
        """Empty preferences with on_no_match='error' should raise."""
        prefs = ModelPreferences(preferences=(), on_no_match="error")

        with pytest.raises(ModelResolutionError):
            resolver_with_models.resolve(prefs)


class TestIntegration:
    """Integration tests combining components."""

    def test_full_resolution_flow(self, tmp_path, monkeypatch):
        """Test complete flow: YAML -> preferences -> resolution -> config."""
        # Create available models config
        available_file = tmp_path / "available_models.yml"
        available_file.write_text(dedent("""\
            models:
              gpt-4o:
                base_url: "https://api.openai.com/v1"
                model_id: "gpt-4o-2024-08-06"
              claude-3-sonnet:
                base_url: "https://api.anthropic.com/v1"
                model_id: "claude-3-sonnet-20240229"
            default_model: "gpt-4o"
            """))
        monkeypatch.setenv("AUTOJUDGE_AVAILABLE_MODELS", str(available_file))

        # Create participant's llm-config.yml
        llm_config = tmp_path / "llm-config.yml"
        llm_config.write_text(dedent("""\
            model_preferences:
              - "claude-3-sonnet"
              - "gpt-4o"
            """))

        # Load and resolve
        prefs = ModelPreferences.from_yaml(llm_config)
        resolver = ModelResolver.from_env()
        config = resolver.resolve(prefs)

        assert config.model == "claude-3-sonnet-20240229"
        assert config.base_url == "https://api.anthropic.com/v1"