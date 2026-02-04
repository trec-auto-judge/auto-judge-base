from typing import Any, List, Optional, Dict, Sequence, Type

from pydantic import BaseModel

from .nuggetizer_types import *

Importance = Literal["vital", "okay", "failed"]
Assignment = Literal["support", "partial_support", "not_support"]


class NuggetizerNugget(BaseModel):
    """AutoJudge wrapper for Nuggetizer nuggets."""

    text:str
    reasoning: Optional[str]=None
    trace: Optional[Any] = None
    importance: Optional[Importance] = None # e.g., "vital" | "okay" | "failed"
    assignment: Optional[Assignment] = None  # e.g., "support", "partial_support", "not_support"


    @classmethod
    def from_nuggetizer_nugget(cls, n: "BaseNugget") -> "NuggetizerNugget":
        """
        Collapse any of the AutoNuggetizer dataclass variants into the unified NuggetizerNugget view.

        Works for:
          - BaseNugget / Nugget
          - ScoredNugget
          - AssignedNugget
          - AssignedScoredNugget
        """
        importance = getattr(n, "importance", None)
        assignment = getattr(n, "assignment", None)

        # Optional: normalize strings (in case upstream uses None/"" or unexpected values)
        # If you prefer strict behavior, delete these two blocks and let Pydantic raise.
        if importance is not None and importance not in ("vital", "okay", "failed"):
            importance = None
        if assignment is not None and assignment not in ("support", "partial_support", "not_support"):
            assignment = None

        return cls(
            text=n.text,
            reasoning=n.reasoning,
            trace=n.trace,
            importance=importance,
            assignment=assignment,
        )


class NuggetizerNuggetBank(BaseModel):
    """Container for a set of nugget questions or claims, tied to a query and optionally including metadata."""

    query: str
    qid: str
    full_query: Any = None
    nuggets: List[NuggetizerNugget] = []

    @property
    def query_id(self) -> str:
        """Canonical identifier for the topic/query (alias for qid)."""
        return self.qid


class NuggetizerNuggetBanks(BaseModel):
    """Container for multiple Nuggetizer NuggetBanks, keyed by qid."""

    format_version: str = "v4"
    banks: Dict[str, NuggetizerNuggetBank] = {}

    @classmethod
    def get_bank_model(cls) -> Type[NuggetizerNuggetBank]:
        """Return the bank model class for this container type."""
        return NuggetizerNuggetBank

    @classmethod
    def from_banks_list(
        cls, banks: List[NuggetizerNuggetBank], overwrite: bool = False
    ) -> "NuggetizerNuggetBanks":
        result: Dict[str, NuggetizerNuggetBank] = {}
        for bank in banks:
            if bank.qid in result and not overwrite:
                raise ValueError(f"Duplicate qid: {bank.qid}")
            result[bank.qid] = bank
        return cls(banks=result)

    def verify(self, expected_topic_ids: Sequence[str], warn: bool = False) -> None:
        """Verify nugget banks against expected topic IDs."""
        from ..verification import NuggetBanksVerification  # Local import avoids cycle
        NuggetBanksVerification(self, expected_topic_ids=expected_topic_ids, warn=warn).all()


# Curried I/O functions for NuggetizerNuggetBanks
from ..io import make_io_functions

load_nuggetizer_banks_from_file, load_nuggetizer_banks_from_directory, write_nuggetizer_banks = \
    make_io_functions(NuggetizerNuggetBank, NuggetizerNuggetBanks)