
import gzip
import hashlib
import importlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, TextIO, Type, TypeVar, Union
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime, timezone


def import_nugget_banks_type(type_path: str) -> Type:
    """
    Import a NuggetBanks container type from a dotted path.

    Args:
        type_path: Dotted import path, e.g., "trec_auto_judge.nugget_data.NuggetBanks"

    Returns:
        The container class (must satisfy NuggetBanksProtocol)

    Raises:
        ImportError: If module or class not found
        AttributeError: If class not found in module

    Example:
        container_type = import_nugget_banks_type("trec_auto_judge.nugget_data.NuggetBanks")
        banks = load_nugget_banks_generic(path, container_type)
    """
    module_path, class_name = type_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# =============================================================================
# Generic I/O for both NuggetBanks and NuggetizerNuggetBanks
# =============================================================================

B = TypeVar("B", bound=BaseModel)  # Bank type
C = TypeVar("C", bound=BaseModel)  # Container type (must have banks dict and from_banks_list)


def _load_json(f: TextIO, bank_model: Type[B], container_model: Type[C]) -> C:
    """Load from single-bank JSON format."""
    data = json.load(f)
    bank = bank_model.model_validate(data)
    return container_model.from_banks_list([bank])


def _load_jsonl(f: TextIO, bank_model: Type[B], container_model: Type[C]) -> C:
    """Load from JSONL format (one bank per line)."""
    banks = []
    for line in f:
        line = line.strip()
        if line:
            data = json.loads(line)
            bank = bank_model.model_validate(data)
            banks.append(bank)
    return container_model.from_banks_list(banks)


def _load_nugget_banks_from_file(
    source: Union[str, Path],
    bank_model: Type[B],
    container_model: Type[C],
) -> C:
    """
    Load nugget banks from file.

    Args:
        source: Path to JSON, JSON.gz, JSONL, or JSONL.gz file
        bank_model: The bank model class (e.g., NuggetBank, NuggetizerNuggetBank)
        container_model: The container model class (e.g., NuggetBanks, NuggetizerNuggetBanks)

    Returns:
        Container with loaded banks
    """
    path = Path(source)
    str_path = str(path).lower()
    open_fn = gzip.open if str_path.endswith(".gz") else open

    with open_fn(path, mode="rt", encoding="utf-8") as f:
        if ".jsonl" in str_path:
            return _load_jsonl(f, bank_model, container_model)
        else:
            return _load_json(f, bank_model, container_model)


def _load_nugget_banks_from_directory(
    directory: Union[str, Path],
    bank_model: Type[B],
    container_model: Type[C],
) -> C:
    """
    Load nugget banks from directory containing per-topic files.

    Searches for: *.json, *.json.gz, *.jsonl, *.jsonl.gz
    """
    path = Path(directory)
    all_banks: List[B] = []

    patterns = ["*.json", "*.json.gz", "*.jsonl", "*.jsonl.gz"]
    for pattern in patterns:
        for file_path in path.glob(pattern):
            loaded = _load_nugget_banks_from_file(file_path, bank_model, container_model)
            all_banks.extend(loaded.banks.values())

    return container_model.from_banks_list(all_banks)


def _write_nugget_banks(
    nugget_banks: C,
    out: Union[str, Path],
    format: str = "jsonl",
) -> None:
    """
    Write nugget banks to file(s).

    Args:
        nugget_banks: Container to write (must have .banks dict)
        out: Output path (file for jsonl, directory for directory format)
        format: "jsonl" (default) or "directory"
    """
    path = Path(out)

    if format == "directory":
        path.mkdir(parents=True, exist_ok=True)
        for query_id, bank in nugget_banks.banks.items():
            safe_id = query_id.replace("/", "_").replace("\\", "_")
            bank_path = path / f"{safe_id}.json.gz"
            with gzip.open(bank_path, mode="wt", encoding="utf-8") as f:
                f.write(bank.model_dump_json(exclude_none=True))
    else:  # jsonl
        str_path = str(path).lower()
        open_fn = gzip.open if str_path.endswith(".gz") else open
        with open_fn(path, mode="wt", encoding="utf-8") as f:
            for bank in nugget_banks.banks.values():
                f.write(bank.model_dump_json(exclude_none=True) + "\n")


# =============================================================================
# Factory for creating type-specialized I/O functions
# =============================================================================

from typing import Callable, Tuple

def make_io_functions(
    bank_model: Type[B],
    container_model: Type[C],
) -> Tuple[
    Callable[[Union[str, Path]], C],
    Callable[[Union[str, Path]], C],
    Callable[[C, Union[str, Path], str], None],
]:
    """
    Create type-specialized I/O functions for a nugget bank container.

    Args:
        bank_model: The bank model class (e.g., NuggetBank, NuggetizerNuggetBank)
        container_model: The container model class (e.g., NuggetBanks, NuggetizerNuggetBanks)

    Returns:
        Tuple of (load_from_file, load_from_directory, write_banks)

    Example:
        load_file, load_dir, write = make_io_functions(NuggetizerNuggetBank, NuggetizerNuggetBanks)
        banks = load_file("nuggets.jsonl")
    """
    def load_from_file(source: Union[str, Path]) -> C:
        return _load_nugget_banks_from_file(source, bank_model, container_model)

    def load_from_directory(directory: Union[str, Path]) -> C:
        return _load_nugget_banks_from_directory(directory, bank_model, container_model)

    def write_banks(nugget_banks: C, out: Union[str, Path], format: str = "jsonl") -> None:
        _write_nugget_banks(nugget_banks, out, format)

    return load_from_file, load_from_directory, write_banks


# =============================================================================
# Protocol-based generic I/O (uses get_bank_model() from NuggetBanksProtocol)
# =============================================================================

from .protocols import NuggetBanksProtocol

P = TypeVar("P", bound=NuggetBanksProtocol)


def load_nugget_banks_generic(
    source: Union[str, Path],
    container_type: Type[P],
) -> P:
    """
    Load nugget banks using protocol-based type discovery.

    Args:
        source: Path to JSON, JSONL, or gzipped file
        container_type: Container class (must have get_bank_model() method)

    Returns:
        Loaded container instance

    Example:
        banks = load_nugget_banks_generic("nuggets.jsonl", NuggetBanks)
        banks = load_nugget_banks_generic("nuggets.jsonl", NuggetizerNuggetBanks)
    """
    bank_model = container_type.get_bank_model()
    return _load_nugget_banks_from_file(source, bank_model, container_type)


def load_nugget_banks_from_directory_generic(
    directory: Union[str, Path],
    container_type: Type[P],
) -> P:
    """
    Load nugget banks from directory using protocol-based type discovery.

    Args:
        directory: Directory containing per-topic files
        container_type: Container class (must have get_bank_model() method)

    Returns:
        Loaded container instance
    """
    bank_model = container_type.get_bank_model()
    return _load_nugget_banks_from_directory(directory, bank_model, container_type)


def write_nugget_banks_generic(
    banks: NuggetBanksProtocol,
    out: Union[str, Path],
    format: str = "jsonl",
) -> None:
    """
    Write nugget banks using protocol-based type discovery.

    Derives types from the instance - no need to specify format.

    Args:
        banks: Any NuggetBanksProtocol instance
        out: Output path
        format: "jsonl" (default) or "directory"

    Example:
        write_nugget_banks_generic(banks, "output.jsonl")
    """
    _write_nugget_banks(banks, out, format)