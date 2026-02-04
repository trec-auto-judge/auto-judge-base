# model_resolver.py
"""Model preference resolution for AutoJudge framework.

Allows participants to declare an ordered list of model preferences,
resolved against an organizer-controlled pool of available models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from ..llm.llm_config import MinimaLlmConfig, BatchConfig, _env_str


class ModelResolutionError(Exception):
    """Raised when no model can be resolved from preferences."""
    pass


@dataclass(frozen=True)
class ModelPreferences:
    """Participant-declared model preferences."""
    preferences: Sequence[str]  # Ordered list, first available wins
    on_no_match: str = "use_default"  # "use_default" | "error"

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelPreferences":
        """Create from dictionary (parsed YAML)."""
        return cls(
            preferences=tuple(data.get("model_preferences", [])),
            on_no_match=data.get("on_no_match", "use_default"),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelPreferences":
        """
        Load from llm-config.yml.

        Expected format:
            model_preferences:
              - "gpt-4o"
              - "gpt-4-turbo"
            on_no_match: "use_default"  # optional, defaults to "use_default"
        """
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


@dataclass
class AvailableModels:
    """Organizer-controlled pool of available models."""
    models: Dict[str, MinimaLlmConfig] = field(default_factory=dict)
    default_model: Optional[str] = None
    aliases: Dict[str, str] = field(default_factory=dict)
    disabled: set[str] = field(default_factory=set)

    @classmethod
    def from_yaml(cls, path: Path) -> "AvailableModels":
        """Load from available_models.yml configuration file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        models = {}
        disabled = set()
        batch_config = BatchConfig.from_env()

        for name, spec in data.get("models", {}).items():
            if not spec.get("enabled", True):
                disabled.add(name)
                continue

            api_key_env = spec.get("api_key_env", "OPENAI_API_KEY")
            api_key = _env_str(api_key_env)

            models[name] = MinimaLlmConfig(
                base_url=spec["base_url"].rstrip("/"),
                model=spec["model_id"],
                api_key=api_key,
                batch=batch_config,
            )

        default_model = data.get("default_model")
        if not default_model:
            raise ValueError("Organizer config must set 'default_model'")
        if default_model not in models:
            raise ValueError(
                f"default_model '{default_model}' is not an enabled model. "
                f"Enabled models: {list(models.keys())}"
            )

        return cls(
            models=models,
            default_model=default_model,
            aliases=data.get("aliases", {}),
            disabled=disabled,
        )

    @classmethod
    def from_env(cls) -> "AvailableModels":
        """
        Load from environment or default paths.

        Priority:
        1. AUTOJUDGE_AVAILABLE_MODELS env var pointing to YAML file
        2. ~/.autojudge/available_models.yml
        3. Fallback: create from legacy OPENAI_MODEL/OPENAI_BASE_URL env vars
        """
        # Check explicit path first
        config_path = _env_str("AUTOJUDGE_AVAILABLE_MODELS")
        if config_path and Path(config_path).exists():
            return cls.from_yaml(Path(config_path))

        # Check default user path
        default_path = Path.home() / ".autojudge" / "available_models.yml"
        if default_path.exists():
            return cls.from_yaml(default_path)

        # Fallback: create from legacy environment variables
        try:
            config = MinimaLlmConfig.from_env()
            return cls(
                models={config.model: config},
                default_model=config.model,
            )
        except RuntimeError:
            return cls(models={})

    def get_enabled_models(self) -> List[str]:
        """Return list of enabled model names."""
        return list(self.models.keys())

    def resolve_alias(self, name: str) -> str:
        """Resolve model name through aliases."""
        return self.aliases.get(name, name)


@dataclass
class ModelResolver:
    """Resolves participant preferences against available pool."""
    available: AvailableModels

    def resolve(self, preferences: ModelPreferences) -> MinimaLlmConfig:
        """
        Resolve first matching model from preferences against available pool.

        Raises ModelResolutionError if no match and on_no_match="error".
        """
        for preferred in preferences.preferences:
            canonical = self.available.resolve_alias(preferred)
            if canonical in self.available.models:
                return self.available.models[canonical]

        # No match found
        if preferences.on_no_match == "use_default" and self.available.default_model:
            default_config = self.available.models.get(self.available.default_model)
            if default_config:
                return default_config

        enabled = self.available.get_enabled_models()
        raise ModelResolutionError(
            f"No model available from preferences {list(preferences.preferences)}. "
            f"Available models: {enabled}"
        )

    @classmethod
    def from_env(cls) -> "ModelResolver":
        """Create resolver from environment configuration."""
        return cls(available=AvailableModels.from_env())