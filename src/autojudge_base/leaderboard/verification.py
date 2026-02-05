"""Verification for leaderboard completeness and consistency."""

import sys
from itertools import groupby
from typing import Optional, Sequence, Set, TYPE_CHECKING

from autojudge_base.utils import format_preview

if TYPE_CHECKING:
    from .leaderboard import Leaderboard, OnMissing


class LeaderboardVerificationError(Exception):
    """Raised when leaderboard verification fails."""
    pass


class LeaderboardVerification:
    """
    Fluent verifier for leaderboards.

    Chain verification methods to run multiple checks:

        LeaderboardVerification(leaderboard, on_missing="fix_aggregate").complete_measures().same_topics_per_run()

    Or run all checks:

        LeaderboardVerification(leaderboard, on_missing="fix_aggregate", expected_topic_ids=topics).all()

    Each method raises LeaderboardVerificationError on failure (fail-fast).
    """

    def __init__(
        self,
        leaderboard: "Leaderboard",
        on_missing: "OnMissing",
        expected_topic_ids: Optional[Sequence[str]] = None,
        warn: Optional[bool]=False
    ):
        """
        Initialize verifier.

        Args:
            leaderboard: The leaderboard to verify
            on_missing: How to handle missing entries
            expected_topic_ids: Optional set of expected topic IDs
            warn: If True, print warnings instead of raising exceptions
        """
        self.leaderboard = leaderboard
        self.expected_topic_ids = expected_topic_ids
        self.warn = warn
        self.on_missing = on_missing

    def _raise_or_warn(self, err: LeaderboardVerificationError):
        if self.warn:
            print(f"Leaderboard Verification Warning: {err}", file=sys.stderr)
        else:
            raise err

    def complete_measures(self, include_all_row: bool = True) -> "LeaderboardVerification":
        """
        Verify that every (run_id, topic_id) entry contains all measures.

        Args:
            include_all_row: If True, also check synthetic all_topic_id rows

        Raises:
            LeaderboardVerificationError: If any entry is missing or has extra measures
        """
        required = set(self.leaderboard.measures)
        all_topic_id = self.leaderboard.all_topic_id

        missing_reports: list[str] = []
        for e in self.leaderboard.entries:
            if not include_all_row and e.topic_id == all_topic_id:
                continue
            present = set(e.values.keys())
            missing = required - present
            extra = present - required
            if missing or extra:
                parts = []
                if missing:
                    parts.append(f"missing={sorted(missing)}")
                if extra:
                    parts.append(f"extra={sorted(extra)}")
                missing_reports.append(f"{e.run_id}/{e.topic_id}: " + ", ".join(parts))

        if missing_reports and self.on_missing != "fix_aggregate":
            self._raise_or_warn(LeaderboardVerificationError(
                "Leaderboard entries do not match the measure schema:\n  " + format_preview(missing_reports, limit=25, separator="\n  ")
            ))

        return self

    def same_topics_per_run(self, include_all_row: bool = False) -> "LeaderboardVerification":
        """
        Verify that all runs have the same set of topic_ids.

        This catches cases where run A has topics {t1,t2} but run B has {t1,t3},
        which makes per-run comparisons misleading.

        Args:
            include_all_row: If True, include synthetic all_topic_id rows

        Raises:
            LeaderboardVerificationError: If runs have different topic sets
        """
        all_topic_id = self.leaderboard.all_topic_id
        by_run: dict[str, Set[str]] = {}

        for e in self.leaderboard.entries:
            if not include_all_row and e.topic_id == all_topic_id:
                continue
            by_run.setdefault(e.run_id, set()).add(e.topic_id)

        if not by_run:
            return self

        runs = sorted(by_run.keys())
        reference_run = runs[0]
        reference_topics = by_run[reference_run]

        diffs: list[str] = []
        for r in runs[1:]:
            tset = by_run[r]
            missing = reference_topics - tset
            extra = tset - reference_topics
            if missing or extra:
                parts = []
                if missing:
                    parts.append(f"missing_topics={sorted(missing)}")
                if extra:
                    parts.append(f"extra_topics={sorted(extra)}")
                diffs.append(f"{r} vs {reference_run}: " + ", ".join(parts))

        if diffs:
            self._raise_or_warn(LeaderboardVerificationError(
                f"Runs do not share the same topic set (reference={reference_run}):\n  " + format_preview(diffs, limit=25, separator="\n  ")
            ))

        return self

    def complete_topics(self, include_all_row: bool = False) -> "LeaderboardVerification":
        """
        Verify that every expected topic has at least one entry.

        Requires expected_topic_ids to be set in constructor.

        Args:
            include_all_row: If True, include synthetic all_topic_id rows

        Raises:
            LeaderboardVerificationError: If any expected topic is missing
        """
        if self.on_missing == "fix_aggregate":
            # Not checking for complete topics
            return self

        if self.expected_topic_ids is None:
            raise RuntimeError("Must set `expected_topic_ids` to non-null in order to check `complete_topics`")

        all_topic_id = self.leaderboard.all_topic_id
        expected = set(self.expected_topic_ids)


        for run, leaderboard_run_entries in groupby(sorted(self.leaderboard.entries
                                                           , key=lambda e: e.run_id)
                                                    , key=lambda e:e.run_id):
            seen = set()

            for e in leaderboard_run_entries:
                if not include_all_row and e.topic_id == all_topic_id:
                    continue
                if e.topic_id in expected:
                    seen.add(e.topic_id)

            missing = expected - seen
            if missing:
                missing_list = sorted(missing)
                self._raise_or_warn(LeaderboardVerificationError(
                    f"Run {run}: Missing leaderboard entries for {len(missing_list)} topic(s): {format_preview(missing_list)}"
                ))

        return self

    def no_extra_topics(self, include_all_row: bool = False) -> "LeaderboardVerification":
        """
        Verify no entries exist for non-expected topics.

        Requires expected_topic_ids to be set in constructor.

        Args:
            include_all_row: If True, include synthetic all_topic_id rows in check

        Raises:
            LeaderboardVerificationError: If entries exist for unknown topics
        """
        if self.expected_topic_ids is None:
            return self

        all_topic_id = self.leaderboard.all_topic_id
        expected = set(self.expected_topic_ids)
        extras = set()

        for e in self.leaderboard.entries:
            if not include_all_row and e.topic_id == all_topic_id:
                continue
            if e.topic_id not in expected and e.topic_id != all_topic_id:
                extras.add(e.topic_id)

        if extras:
            extra_list = sorted(extras)
            self._raise_or_warn(LeaderboardVerificationError(
                f"Leaderboard entries for {len(extra_list)} unexpected topic(s): {format_preview(extra_list)}"
            ))

        return self

    def all(self, include_all_row: bool = True) -> "LeaderboardVerification":
        """
        Run all verification checks.

        Checks run in order (fail-fast):
        1. complete_measures - every entry has all measures
        2. complete_topics - every expected topic has entries (if expected_topic_ids set)
        3. no_extra_topics - no entries for unknown topics (if expected_topic_ids set)
        4. same_topics_per_run - all runs have same topic set

        Args:
            include_all_row: If True, include synthetic all_topic_id rows in measure check

        Returns:
            self for chaining
        """

        if self.expected_topic_ids is not None:
            return (
                self.complete_measures(include_all_row=include_all_row)
                .complete_topics()
                .no_extra_topics()
                # .same_topics_per_run()   # omitted because that is already covered by the previous calls.
            )
        else:
            return (
                self.complete_measures(include_all_row=include_all_row)
                .same_topics_per_run()  # Fall back when we don't know the true topics.
            )
