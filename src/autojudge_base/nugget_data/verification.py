"""Verification for nugget banks completeness and consistency."""

import sys
from typing import Optional, Sequence, TYPE_CHECKING

from autojudge_base.utils import format_preview

if TYPE_CHECKING:
    from .protocols import NuggetBanksProtocol


class NuggetBanksVerificationError(Exception):
    """Raised when nugget banks verification fails."""
    pass


class NuggetBanksVerification:
    """
    Fluent verifier for nugget banks.

    Chain verification methods to run multiple checks:

        NuggetBanksVerification(banks, topic_ids).complete_topics().non_empty_banks()

    Or run all checks:

        NuggetBanksVerification(banks, topic_ids).all()

    Each method raises NuggetBanksVerificationError on failure (fail-fast).
    """

    def __init__(
        self,
        nugget_banks: "NuggetBanksProtocol",
        expected_topic_ids: Sequence[str],
        warn: Optional[bool] = False,
    ):
        """
        Initialize verifier.

        Args:
            nugget_banks: The nugget banks container to verify
            expected_topic_ids: The expected topic IDs to verify against
            warn: If True, print warnings instead of raising exceptions
        """
        self.nugget_banks = nugget_banks
        self.expected_topic_ids = expected_topic_ids
        self.warn = warn

    def _raise_or_warn(self, err: NuggetBanksVerificationError):
        if self.warn:
            print(f"Nugget Verification Warning: {err}", file=sys.stderr)
        else:
            raise err

    def complete_topics(self) -> "NuggetBanksVerification":
        """
        Verify every topic has a nugget bank entry.

        Raises:
            NuggetBanksVerificationError: If any topic is missing a nugget bank
        """
        bank_ids = set(self.nugget_banks.banks.keys())
        topic_ids = set(self.expected_topic_ids)

        missing = topic_ids - bank_ids
        if missing:
            missing_list = sorted(missing)
            self._raise_or_warn(NuggetBanksVerificationError(
                f"Missing nugget banks for {len(missing_list)} topic(s): {format_preview(missing_list)}"
            ))

        return self

    def no_extra_topics(self) -> "NuggetBanksVerification":
        """
        Verify no nugget banks exist for non-existent topics.

        Raises:
            NuggetBanksVerificationError: If banks exist for unknown topics
        """
        bank_ids = set(self.nugget_banks.banks.keys())
        topic_ids = set(self.expected_topic_ids)

        extra = bank_ids - topic_ids
        if extra:
            extra_list = sorted(extra)
            self._raise_or_warn(NuggetBanksVerificationError(
                f"Nugget banks for {len(extra_list)} unknown topic(s): {format_preview(extra_list)}"
            ))

        return self

    def non_empty_banks(self) -> "NuggetBanksVerification":
        """
        Verify each nugget bank has at least one nugget.

        Raises:
            NuggetBanksVerificationError: If any bank is empty
        """
        empty_banks = []

        for query_id, bank in self.nugget_banks.banks.items():
            # Check for nuggets - different bank types have different structures
            # NuggetBank has nugget_bank (questions) and claim_bank (claims)
            # NuggetizerNuggetBank has nuggets list
            has_nuggets = False

            # Try NuggetBank structure (nugget_bank dict of questions)
            if hasattr(bank, 'nugget_bank') and bank.nugget_bank:
                has_nuggets = True
            # Try NuggetBank claim structure
            elif hasattr(bank, 'claim_bank') and bank.claim_bank:
                has_nuggets = True
            # Try NuggetizerNuggetBank structure (nuggets list)
            elif hasattr(bank, 'nuggets') and bank.nuggets:
                has_nuggets = True

            if not has_nuggets:
                empty_banks.append(query_id)

        if empty_banks:
            self._raise_or_warn(NuggetBanksVerificationError(
                f"Empty nugget banks for {len(empty_banks)} topic(s): {format_preview(sorted(empty_banks))}"
            ))

        return self

    def all(self) -> "NuggetBanksVerification":
        """
        Run all verification checks.

        Checks run in order (fail-fast):
        1. complete_topics - every topic has a bank
        2. no_extra_topics - no banks for unknown topics
        3. non_empty_banks - each bank has nuggets

        Returns:
            self for chaining
        """
        return self.complete_topics().no_extra_topics().non_empty_banks()