"""
Minimal LLM configuration for AutoJudgeBase.

This provides a basic implementation that can be overridden by
MinimaLlmConfig from the minima-llm package for full features.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import Optional
import yaml


# ----------------------------
# Helper functions
# ----------------------------

def _env_str(name: str) -> Optional[str]:
    v = os.getenv(name)
    return None if v in (None, "") else v


def _env_int(name: str, default: int) -> int:
    v = _env_str(name)
    return default if v is None else int(v)


def _env_float(name: str, default: float) -> float:
    v = _env_str(name)
    return default if v is None else float(v)


def _env_opt_int(name: str, default: Optional[int]) -> Optional[int]:
    v = _env_str(name)
    if v is None:
        return default
    if v.strip().lower() in ("none", "null"):
        return None
    return int(v)


def _first_non_none(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v is not None:
            return v
    return None


# ----------------------------
# Batch configuration (stub)
# ----------------------------

@dataclass(frozen=True)
class BatchConfig:
    """Configuration for async batch execution."""
    num_workers: int = 64
    max_failures: Optional[int] = 25
    heartbeat_s: float = 10.0
    stall_s: float = 300.0
    print_first_failures: int = 5
    keep_failure_summaries: int = 20

    @classmethod
    def from_env(cls) -> "BatchConfig":
        return cls(
            num_workers=_env_int("BATCH_NUM_WORKERS", 64),
            max_failures=_env_opt_int("BATCH_MAX_FAILURES", 25),
            heartbeat_s=_env_float("BATCH_HEARTBEAT_S", 10.0),
            stall_s=_env_float("BATCH_STALL_S", 300.0),
            print_first_failures=_env_int("BATCH_PRINT_FIRST_FAILURES", 5),
            keep_failure_summaries=_env_int("BATCH_KEEP_FAILURE_SUMMARIES", 20),
        )


# ----------------------------
# Parasail batch configuration (stub)
# ----------------------------

@dataclass(frozen=True)
class ParasailBatchConfig:
    """Configuration for Parasail Batch API."""
    llm_batch_prefix: Optional[str] = None
    prefix: Optional[str] = None
    state_dir: Optional[str] = None
    poll_interval_s: float = 30.0
    max_poll_hours: float = 24.0
    max_batch_size: int = 50000

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "ParasailBatchConfig":
        if not data:
            return cls()
        return cls(
            llm_batch_prefix=data.get("llm_batch_prefix"),
            prefix=data.get("prefix"),
            state_dir=data.get("state_dir"),
            poll_interval_s=float(data.get("poll_interval_s", 30.0)),
            max_poll_hours=float(data.get("max_poll_hours", 24.0)),
            max_batch_size=int(data.get("max_batch_size", 50000)),
        )


# ----------------------------
# LLM configuration
# ----------------------------

@dataclass(frozen=True)
class MinimaLlmConfig:
    """
    Minimal LLM configuration for AutoJudgeBase.

    For full features (batching, transport control, advanced retry),
    install minima-llm and use its MinimaLlmConfig instead.
    """

    # Endpoint
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None

    # Batch execution (composed)
    batch: BatchConfig = field(default_factory=BatchConfig)

    # Parasail batch mode (composed)
    parasail: ParasailBatchConfig = field(default_factory=ParasailBatchConfig)

    # Transport / backpressure
    max_outstanding: int = 32
    rpm: int = 600
    timeout_s: float = 60.0

    # Retry / backoff
    max_attempts: int = 6
    base_backoff_s: float = 0.5
    max_backoff_s: float = 20.0
    jitter: float = 0.2

    # Cooldown
    cooldown_floor_s: float = 0.0
    cooldown_cap_s: float = 30.0
    cooldown_halflife_s: float = 20.0

    # HTTP
    compress_gzip: bool = False

    # Cache
    cache_dir: Optional[str] = None
    force_refresh: bool = False

    def with_model(self, model: str) -> "MinimaLlmConfig":
        """Return a new config with the model replaced."""
        return replace(self, model=model)

    # Backward compatibility properties
    @property
    def num_workers(self) -> int:
        return self.batch.num_workers

    @property
    def max_failures(self) -> Optional[int]:
        return self.batch.max_failures

    @property
    def heartbeat_s(self) -> float:
        return self.batch.heartbeat_s

    @property
    def stall_s(self) -> float:
        return self.batch.stall_s

    @property
    def print_first_failures(self) -> int:
        return self.batch.print_first_failures

    @property
    def keep_failure_summaries(self) -> int:
        return self.batch.keep_failure_summaries

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        return base_url.rstrip("/")

    @classmethod
    def from_env(cls) -> "MinimaLlmConfig":
        """Construct from environment variables."""
        base_url = _env_str("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        model = _env_str("OPENAI_MODEL") or "gpt-4o-mini"
        api_key = _first_non_none(
            _env_str("OPENAI_API_KEY"),
            _env_str("OPENAI_TOKEN"),
        )

        return cls(
            base_url=cls._normalize_base_url(base_url),
            model=model,
            api_key=api_key,
            batch=BatchConfig.from_env(),
            max_outstanding=_env_int("MAX_OUTSTANDING", 32),
            rpm=_env_int("RPM", 600),
            timeout_s=_env_float("TIMEOUT_S", 60.0),
            max_attempts=_env_int("MAX_ATTEMPTS", 50),
            base_backoff_s=_env_float("BASE_BACKOFF_S", 0.5),
            max_backoff_s=_env_float("MAX_BACKOFF_S", 20.0),
            jitter=_env_float("JITTER", 0.2),
            cooldown_floor_s=_env_float("COOLDOWN_FLOOR_S", 0.0),
            cooldown_cap_s=_env_float("COOLDOWN_CAP_S", 60.0),
            cooldown_halflife_s=_env_float("COOLDOWN_HALFLIFE_S", 20.0),
            compress_gzip=(_env_int("COMPRESS_GZIP", 0) != 0),
            cache_dir=_env_str("CACHE_DIR"),
            force_refresh=(_env_int("CACHE_FORCE_REFRESH", 0) != 0),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "MinimaLlmConfig":
        """Load from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        base_url = data.get("base_url") or _env_str("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        model = data.get("model") or _env_str("OPENAI_MODEL") or "gpt-4o-mini"
        api_key = data.get("api_key") or _first_non_none(
            _env_str("OPENAI_API_KEY"),
            _env_str("OPENAI_TOKEN"),
        )

        return cls(
            base_url=cls._normalize_base_url(base_url),
            model=model,
            api_key=api_key,
            batch=BatchConfig.from_env(),
            parasail=ParasailBatchConfig.from_dict(data.get("parasail")),
            max_outstanding=data.get("max_outstanding", _env_int("MAX_OUTSTANDING", 32)),
            rpm=data.get("rpm", _env_int("RPM", 600)),
            timeout_s=data.get("timeout_s", _env_float("TIMEOUT_S", 60.0)),
            max_attempts=data.get("max_attempts", _env_int("MAX_ATTEMPTS", 50)),
            base_backoff_s=data.get("base_backoff_s", _env_float("BASE_BACKOFF_S", 0.5)),
            max_backoff_s=data.get("max_backoff_s", _env_float("MAX_BACKOFF_S", 20.0)),
            jitter=data.get("jitter", _env_float("JITTER", 0.2)),
            cooldown_floor_s=data.get("cooldown_floor_s", _env_float("COOLDOWN_FLOOR_S", 0.0)),
            cooldown_cap_s=data.get("cooldown_cap_s", _env_float("COOLDOWN_CAP_S", 60.0)),
            cooldown_halflife_s=data.get("cooldown_halflife_s", _env_float("COOLDOWN_HALFLIFE_S", 20.0)),
            compress_gzip=data.get("compress_gzip", (_env_int("COMPRESS_GZIP", 0) != 0)),
            cache_dir=data.get("cache_dir", _env_str("CACHE_DIR")),
            force_refresh=data.get("force_refresh", (_env_int("CACHE_FORCE_REFRESH", 0) != 0)),
        )
