"""Verification for qrels completeness and consistency."""

import sys
from typing import Dict, Optional, Sequence, Tuple, TYPE_CHECKING

from autojudge_base.utils import format_preview

if TYPE_CHECKING:
    from .qrels import Qrels, QrelRow


class QrelsVerificationError(Exception):
    """Raised when qrels verification fails."""
    pass


class QrelsVerification:
    """
    Fluent verifier for qrels.

    Chain verification methods to run multiple checks:

        QrelsVerification(qrels, expected_topic_ids).complete_topics().no_duplicates()

    Or run all checks:

        QrelsVerification(qrels, expected_topic_ids).all()

    Each method raises QrelsVerificationError on failure (fail-fast).
    """

    def __init__(
        self,
        qrels: "Qrels",
        expected_topic_ids: Sequence[str],
        warn: Optional[bool]=False
    ):
        """
        Initialize verifier.

        Args:
            qrels: The qrels to verify
            expected_topic_ids: The expected topic IDs to verify against
            warn: If True, print warnings instead of raising exceptions
        """
        self.qrels = qrels
        self.expected_topic_ids = expected_topic_ids
        self.warn = warn

    def _raise_or_warn(self, err: QrelsVerificationError):
        if self.warn:
            print(f"Qrels Verification Warning: {err}", file=sys.stderr)
        else:
            raise err


    def complete_topics(self) -> "QrelsVerification":
        """
        Verify every expected topic has at least one qrel row.

        Raises:
            QrelsVerificationError: If any topic is missing qrels
        """
        expected = set(self.expected_topic_ids)
        seen = set()

        for r in self.qrels.rows:
            if r.topic_id in expected:
                seen.add(r.topic_id)

        missing = expected - seen
        if missing:
            missing_list = sorted(missing)
            raise QrelsVerificationError(
                f"Missing qrels for {len(missing_list)} topic(s): {format_preview(missing_list)}"
            )

        return self

    def no_extra_topics(self) -> "QrelsVerification":
        """
        Verify no qrels exist for non-expected topics.

        Raises:
            QrelsVerificationError: If qrels exist for unknown topics
        """
        expected = set(self.expected_topic_ids)
        extras = {r.topic_id for r in self.qrels.rows} - expected

        if extras:
            extra_list = sorted(extras)
            raise QrelsVerificationError(
                f"Qrels for {len(extra_list)} unexpected topic(s): {format_preview(extra_list)}"
            )

        return self

    def no_duplicates(self) -> "QrelsVerification":
        """
        Verify no duplicate (topic_id, doc_id) pairs exist.

        Raises:
            QrelsVerificationError: If duplicates are found
        """
        checked: Dict[Tuple[str, str], "QrelRow"] = {}

        for r in self.qrels.rows:
            key = (r.topic_id, r.doc_id)
            if key in checked:
                raise QrelsVerificationError(
                    f"Duplicate qrel for topic={r.topic_id} doc_id={r.doc_id}: "
                    f"first={checked[key]}, duplicate={r}"
                )
            checked[key] = r

        return self

    def all(self) -> "QrelsVerification":
        """
        Run all verification checks.

        Checks run in order (fail-fast):
        1. no_duplicates - no duplicate (topic_id, doc_id) pairs
        2. complete_topics - every expected topic has qrels
        3. no_extra_topics - no qrels for unknown topics

        Returns:
            self for chaining
        """
        return self.no_duplicates().complete_topics().no_extra_topics()
