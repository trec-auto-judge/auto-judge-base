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


@runtime_checkable
class LlmConfigProtocol(Protocol):
    """Protocol for LLM configuration classes."""

    model: str
    cache_dir: Optional[Path]

    @classmethod
    def from_env(cls) -> "LlmConfigProtocol":
        """Load configuration from environment variables."""
        ...


@dataclass
class LlmConfigBase:
    """
    Minimal LLM configuration implementation.

    For full features (batching, transport, retries), judges can use the
    raw config dict to instantiate MinimaLlmConfig from the minima-llm package:

        from minima_llm import MinimaLlmConfig, OpenAIMinimaLlm
        full_config = MinimaLlmConfig.from_dict(llm_config.raw)
        backend = OpenAIMinimaLlm(full_config)
    """

    model: str = "gpt-4o-mini"
    cache_dir: Optional[Path] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    raw: dict = field(default_factory=dict)  # Full config for backends that need more

    @classmethod
    def from_env(cls) -> "LlmConfigBase":
        """Load from environment variables."""
        # Note: Path("") is Path(".") and truthy, so the empty-string fallback
        # must be resolved BEFORE constructing the Path — otherwise an unset
        # CACHE_DIR silently enables caching in the current directory.
        cache = os.getenv("CACHE_DIR") or os.getenv("LLM_CACHE_DIR")
        return cls(
            model=os.getenv("OPENAI_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini")),
            cache_dir=Path(cache) if cache else None,
            api_key=os.getenv("OPENAI_API_KEY"),
            # OPENAI_API_BASE is the litellm-convention name for the same thing;
            # accept both so judges work under either injection convention.
            base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"),
            raw={},
        )






