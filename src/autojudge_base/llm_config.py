"""
Minimal LLM configuration protocol for AutoJudgeBase.

This module defines a protocol that LLM configuration classes must implement.
The full MinimaLlmConfig from the minima-llm package implements this protocol.

For simple use cases, LlmConfigBase provides a minimal implementation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
import os
import yaml


@runtime_checkable
class LlmConfigProtocol(Protocol):
    """Protocol for LLM configuration classes."""

    model: str
    cache_dir: Optional[Path]

    @classmethod
    def from_env(cls) -> "LlmConfigProtocol":
        """Load configuration from environment variables."""
        ...

    @classmethod
    def from_yaml(cls, path: Path) -> "LlmConfigProtocol":
        """Load configuration from a YAML file."""
        ...

    @classmethod
    def from_dict(cls, data: dict, default: "LlmConfigProtocol") -> "LlmConfigProtocol":
        """Overlay dict values onto a default config."""
        ...


@dataclass
class LlmConfigBase:
    """
    Minimal LLM configuration implementation.

    For full features (OpenAI, Azure, batching, DSPy integration),
    use MinimaLlmConfig from the minima-llm package.
    """

    model: str = "gpt-4o-mini"
    cache_dir: Optional[Path] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "LlmConfigBase":
        """Load from environment variables."""
        return cls(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            cache_dir=Path(os.getenv("LLM_CACHE_DIR")) if os.getenv("LLM_CACHE_DIR") else None,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    @classmethod
    def from_dict(cls, data: dict, default: "LlmConfigBase") -> "LlmConfigBase":
        """Overlay dict values onto a default config."""
        from dataclasses import replace
        kwargs = {}
        if "model" in data:
            kwargs["model"] = data["model"]
        if "cache_dir" in data:
            kwargs["cache_dir"] = Path(data["cache_dir"]) if data["cache_dir"] else None
        if "api_key" in data:
            kwargs["api_key"] = data["api_key"]
        if "base_url" in data:
            kwargs["base_url"] = data["base_url"]
        return replace(default, **kwargs) if kwargs else default

    @classmethod
    def from_yaml(cls, path: Path) -> "LlmConfigBase":
        """Load from YAML file, using env as base."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data, default=cls.from_env())

    @classmethod
    def from_cli(cls, default: "LlmConfigBase", **cli_args) -> "LlmConfigBase":
        """Overlay individual CLI flags onto a config."""
        data = {k: v for k, v in cli_args.items() if v is not None}
        return cls.from_dict(data, default=default)


def load_llm_config(
    yaml_path: Optional[Path] = None,
    **cli_overrides
) -> LlmConfigBase:
    """
    Load LLM config with standard layering: env -> yaml -> cli.

    Args:
        yaml_path: Optional path to YAML config file
        **cli_overrides: CLI flag overrides (model, cache_dir, etc.)

    Returns:
        Configured LlmConfigBase instance
    """
    if yaml_path:
        config = LlmConfigBase.from_yaml(yaml_path)
    else:
        config = LlmConfigBase.from_env()

    return LlmConfigBase.from_cli(config, **cli_overrides)
