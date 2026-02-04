"""
Path resolution utilities for AutoJudge workflow outputs.

Provides consistent file naming conventions:
- Nuggets: {filebase}.nuggets.jsonl
- Qrels: {filebase}.qrels.txt
- Leaderboard: {filebase}.eval.txt
- Config: {filebase}.config.yml
"""

from pathlib import Path


def resolve_nugget_file_path(filebase: Path) -> Path:
    """
    Resolve nugget file path from filebase.

    If filebase already has a recognized extension (.jsonl, .json), use as-is.
    Otherwise, append .nuggets.jsonl extension.
    """
    if filebase.suffix in (".jsonl", ".json"):
        return filebase
    return filebase.parent / f"{filebase.name}.nuggets.jsonl"


def resolve_leaderboard_file_path(filebase: Path) -> Path:
    """
    Resolve leaderboard file path from filebase.

    Returns:
        {filebase}.eval.txt
    """
    if filebase.name.endswith(".eval.txt"):
        return filebase
    return filebase.parent / f"{filebase.name}.eval.txt"


def resolve_config_file_path(filebase: Path) -> Path:
    """Resolve config file path: {filebase}.config.yml"""
    if filebase.suffix in (".yml", ".yaml"):
        return filebase
    return filebase.parent / f"{filebase.name}.config.yml"


def resolve_responses_file_path(filebase: Path) -> Path:
    """Resolve augmented responses file path: {filebase}.responses.jsonl"""
    if filebase.suffix in (".jsonl",):
        return filebase
    return filebase.parent / f"{filebase.name}.responses.jsonl"


def resolve_qrels_file_path(filebase: Path) -> Path:
    """
    Resolve qrels file path from filebase.

    Returns:
        {filebase}.qrels.txt
    """
    if filebase.name.endswith(".qrels.txt"):
        return filebase
    return filebase.parent / f"{filebase.name}.qrels.txt"


def load_qrels_from_path(path: Path):
    """
    Load qrels from file.

    Args:
        path: Path to qrels file (.qrels format)

    Returns:
        Loaded Qrels instance
    """
    from ..qrels.qrels import read_qrel_file

    return read_qrel_file(path)


def load_nugget_banks_from_path(path: Path, nugget_banks_type: type):
    """
    Load nugget banks from file or directory.

    Args:
        path: Path to nugget banks file (.jsonl/.json) or directory
        nugget_banks_type: NuggetBanks container class (e.g., NuggetBanks, NuggetizerNuggetBanks)

    Returns:
        Loaded NuggetBanks instance

    Raises:
        ValueError: If path is neither file nor directory
    """
    from ..nugget_data.io import (
        load_nugget_banks_generic,
        load_nugget_banks_from_directory_generic,
    )

    if path.is_file():
        return load_nugget_banks_generic(path, nugget_banks_type)
    elif path.is_dir():
        return load_nugget_banks_from_directory_generic(path, nugget_banks_type)
    else:
        raise ValueError(f"Path {path} is neither a file nor directory")