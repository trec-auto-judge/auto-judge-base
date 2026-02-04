"""
Minimal LLM configuration for AutoJudgeBase.

This module provides stub implementations that can be overridden
by the full minima-llm package when installed.
"""

from .llm_config import (
    MinimaLlmConfig,
    BatchConfig,
    ParasailBatchConfig,
    _env_str,
    _env_int,
    _env_float,
    _env_opt_int,
    _first_non_none,
)

__all__ = [
    "MinimaLlmConfig",
    "BatchConfig",
    "ParasailBatchConfig",
    "_env_str",
    "_env_int",
    "_env_float",
    "_env_opt_int",
    "_first_non_none",
]
