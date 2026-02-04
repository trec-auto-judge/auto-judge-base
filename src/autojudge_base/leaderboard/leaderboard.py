from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Set

from autojudge_base.utils import format_preview
from .verification import LeaderboardVerification, LeaderboardVerificationError
from .format_detection import format_error_with_hint

MeasureName = str
AggFn = Callable[[Sequence[Any]], Any]
CastFn = Callable[[Any], Any]
OnMissing = Literal["default", "warn", "error", "fix_aggregate"]
LeaderboardFormat = Literal["trec_eval", "tot", "ir_measures", "ranking", "jsonl"]


#  ==== DataClasses for data storage and serialization ===  

@dataclass(frozen=True)
class LeaderboardEntry:
    """One row in a leaderboard: (run_id, topic_id) plus a mapping of measure -> value."""
    run_id: str
    topic_id: str
    values: Dict[MeasureName, Any]


@dataclass(frozen=True)
class Leaderboard:
    """
    Thin serialization vessel for leaderboard results.

    - `measures` defines the measure names.
    - `entries` contains per-topic rows and and per-measure `all_topic_id` rows.
    
    Developer note:
    - Aggregation logic lives in LeaderboardBuilder.
    """
    measures: Tuple[MeasureName, ...]
    entries: Tuple[LeaderboardEntry, ...]
    all_topic_id: str = "all"
    
    def all_measure_names(self) -> Tuple[MeasureName, ...]:
        """Return measure names in schema order."""
        return self.measures

    def write(
        self,
        output: Path,
        format: LeaderboardFormat = "tot",
    ) -> None:
        """
        Write the leaderboard as tab-separated lines.

        Args:
            output: Path to write to
            format: Column order
                - "trec_eval": measure topic value
                - "tot": run measure topic value
                - "ir_measures": run topic measure value

        Only measures present in each entry are written (allows sparse rows).
        """
        lines: List[str] = []
        for e in self.entries:
            for m in self.all_measure_names():
                if m in e.values:
                    if format == "tot":
                        lines.append("\t".join([e.run_id, m, e.topic_id, str(e.values[m])]))
                    elif format == "trec_eval":
                        lines.append("\t".join([m, e.topic_id, str(e.values[m])]))
                    elif format == "ir_measures":
                        lines.append("\t".join([e.run_id, e.topic_id, m, str(e.values[m])]))
                    else:
                        raise ValueError(f"Unknown format: {format!r}")

        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing leaderboard to {output.absolute()}")   # ToDo: use a logger
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: Path,
        format: LeaderboardFormat,
        has_header: bool = False,
    ) -> "Leaderboard":
        """
        Load a leaderboard from file or directory.

        Args:
            path: Path to leaderboard file or directory of files
            format: Column order (whitespace-separated)
                - "trec_eval": measure topic value (3 cols, run_id from filename)
                - "tot": run measure topic value (4 cols)
                - "ir_measures": run topic measure value (4 cols)
                - "ranking": topic Q0 doc_id rank score run (6 cols)
            has_header: If True, skip the first line (header row)

        If path is a directory, all files are loaded and merged into a single
        leaderboard. For trec_eval format, each file's name becomes the run_id.
        """
        if path.is_dir():
            return cls._load_directory(path, format, has_header)
        else:
            return cls._load_file(path, format, has_header)

    @classmethod
    def _load_directory(
        cls,
        directory: Path,
        format: LeaderboardFormat,
        has_header: bool,
    ) -> "Leaderboard":
        """Load and merge all leaderboard files from a directory."""
        # Find all files (exclude hidden files and subdirectories)
        files = sorted([f for f in directory.iterdir() if f.is_file() and not f.name.startswith(".")])

        if not files:
            raise ValueError(f"No files found in directory: {directory}")

        print(f"Loading {len(files)} files from {directory}", file=sys.stderr)

        # Collect all entries and measures across files
        all_entry_values: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
        measure_names: List[str] = []
        measure_set: Set[str] = set()

        for file_path in files:
            leaderboard = cls._load_file(file_path, format, has_header)
            for entry in leaderboard.entries:
                for measure, value in entry.values.items():
                    all_entry_values[(entry.run_id, entry.topic_id)][measure] = value
                    if measure not in measure_set:
                        measure_names.append(measure)
                        measure_set.add(measure)

        # Build merged entries
        entries = [
            LeaderboardEntry(run_id=run_id, topic_id=topic_id, values=values)
            for (run_id, topic_id), values in all_entry_values.items()
        ]

        return cls(
            measures=tuple(measure_names),
            entries=tuple(entries),
        )

    @classmethod
    def _load_file(
        cls,
        path: Path,
        format: LeaderboardFormat,
        has_header: bool,
    ) -> "Leaderboard":
        """Load a leaderboard from a single file."""
        text = path.read_text(encoding="utf-8")
        lines = text.strip().split("\n")

        if has_header and lines:
            lines = lines[1:]

        # Collect values grouped by (run_id, topic_id)
        entry_values: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
        measure_names: List[str] = []
        measure_set: Set[str] = set()

        if format == "ranking":
            print(
                "Warning: Loading ranking format with semantic mapping:\n"
                "  ranking doc_id -> leaderboard run_id\n"
                "  ranking run_id -> leaderboard measure\n"
                "  ranking score -> leaderboard value\n"
                "  ranking rank -> ignored",
                file=sys.stderr,
            )

        for line in lines:
            if not line:
                continue
            parts = line.split()

            if format == "trec_eval":
                if len(parts) != 3:
                    raise ValueError(format_error_with_hint("trec_eval", 3, len(parts), line, lines))
                measure, topic_id, value = parts
                run_id = path.name
            elif format == "tot":
                if len(parts) != 4:
                    raise ValueError(format_error_with_hint("tot", 4, len(parts), line, lines))
                run_id, measure, topic_id, value = parts
            elif format == "ir_measures":
                if len(parts) != 4:
                    raise ValueError(format_error_with_hint("ir_measures", 4, len(parts), line, lines))
                run_id, topic_id, measure, value = parts
            elif format == "ranking":
                if len(parts) != 6:
                    raise ValueError(format_error_with_hint("ranking", 6, len(parts), line, lines))
                topic_id, _q0, run_id, _rank, value, measure = parts
            elif format == "jsonl":
                obj = json.loads(line)
                run_id = obj["run_id"]
                topic_id = str(obj["topic_id"])
                measure = obj["measure"]
                value = str(obj["value"])
            else:
                raise ValueError(f"Unknown format: {format!r}")

            entry_values[(run_id, topic_id)][measure] = value
            if measure not in measure_set:
                measure_names.append(measure)
                measure_set.add(measure)

        # Build entries
        entries = [
            LeaderboardEntry(run_id=run_id, topic_id=topic_id, values=values)
            for (run_id, topic_id), values in entry_values.items()
        ]

        return cls(
            measures=tuple(measure_names),
            entries=tuple(entries),
        )

    def verify(self,  on_missing:OnMissing, expected_topic_ids: Sequence[str], warn:Optional[bool]=False):
        LeaderboardVerification(leaderboard = self, warn=warn, expected_topic_ids=expected_topic_ids, on_missing=on_missing) \
            .complete_measures(include_all_row=True) \
            .complete_topics()


def mean_of_floats(values: Sequence[Any]) -> float:
    """Aggregate numeric values via arithmetic mean (values cast to float)."""
    return mean(float(v) for v in values)


@dataclass(frozen=True)
class MeasureSpec:
    """
    Build-time definition of a measure.

    - `name`: key used in entry.values and output.
    - `aggregate`: computes the synthetic per-run value from per-topic values.
    - `cast`: normalizes/validates per-topic values when rows are added.
    - `default`: value used for missing (run_id, topic_id) pairs when build() fills gaps.
    """
    name: MeasureName
    aggregate: AggFn = mean_of_floats
    cast: CastFn = float
    default: Any = 0.0


@dataclass(frozen=True)
class LeaderboardSpec:
    """
    Build-time schema for a leaderboard.

    The spec defines all valid measure names with aggregator and caster. 
    Storing values for different names will raise an error.
    """
    measures: Tuple[MeasureSpec, ...]
    all_topic_id: str = "all"

    @property
    def names(self) -> Tuple[MeasureName, ...]:
        """Measure names in schema order."""
        return tuple(m.name for m in self.measures)

    @property
    def name_set(self) -> set[MeasureName]:
        """Measure names as a set for fast validation."""
        return set(self.names)

    def cast_values(self, values: Mapping[MeasureName, Any]) -> Dict[MeasureName, Any]:
        """
        Cast/normalize measure values using each MeasureSpec.cast.

        Assumes `values` contains all required measure keys.
        """
        return {m.name: m.cast(values[m.name]) for m in self.measures}


#  ==== Convenient Builder for Leaderboards ===

class LeaderboardBuilder:
    """
    Builder/assembler for Leaderboard.

    Responsibilities:
    - Collect per-topic rows (hand-filled or record-derived).
    - Validate measure keys (fail fast on typos/missing keys).
    - Cast values according to the spec.
    - Compute synthetic per-run `all_topic_id` rows using each measure's aggregator.
    """

    def __init__(self, spec: LeaderboardSpec):
        """Create a builder for a specific leaderboard specification."""
        self.spec = spec
        self._rows: List[LeaderboardEntry] = []

    def add(
        self,
        *,
        run_id: str,
        topic_id: str,
        values: Optional[Dict[MeasureName, Any]] = None,
        **kw: Any,
    ) -> None:
        """
        Add one per-topic row.

        Provide either:
        - `values={...}` (a dict of measure -> value), OR
        - keyword args (e.g., GRADE=..., IS_MATCH=...).

        This method is strict by default:
        - Unknown measure keys raise KeyError.
        - Missing measure keys raise KeyError.
        """
        if values is None:
            values = kw
        elif kw:
            raise TypeError("Pass either values= or keyword measures, not both.")

        extra = set(values) - self.spec.name_set
        missing = self.spec.name_set - set(values)
        if extra:
            raise KeyError(f"Unknown measure(s): {sorted(extra)}")
        if missing:
            raise KeyError(f"Missing measure(s): {sorted(missing)}")

        casted = self.spec.cast_values(values)
        self._rows.append(LeaderboardEntry(run_id=run_id, topic_id=topic_id, values=casted))

    def add_records(
        self,
        records: Iterable[Any],
        *,
        run_id: Callable[[Any], str],
        topic_id: Callable[[Any], str],
        get_values: Callable[[Any], Dict[MeasureName, Any]],
    ) -> None:
        """
        Add multiple rows from an iterable of arbitrary record objects.

        The caller supplies functions to extract:
        - `run_id(record)`
        - `topic_id(record)`
        - `get_values(record)` -> {measure_name: value, ...}
        """
        for r in records:
            self.add(run_id=run_id(r), topic_id=topic_id(r), values=get_values(r))

    def entries(self) -> tuple[LeaderboardEntry, ...]:
        """Return the currently staged per-topic entries (no synthetic 'all' rows)."""
        return tuple(self._rows)


    def _detect_missing_run_topic(
        self,
        expected_topic_ids: Sequence[str],
    ) -> List[tuple[str, str]]:
        """
        Detect missing (run_id, topic_id) pairs.

        Returns list of (run_id, topic_id) tuples for each run that is missing
        expected topics.
        """
        existing_run_topic: Dict[str, Set[str]] = defaultdict(set)
        for e in self._rows:
            if e.topic_id != self.spec.all_topic_id:
                existing_run_topic[e.run_id].add(e.topic_id)

        expected_set = set(expected_topic_ids)
        missing: List[tuple[str, str]] = []
        for run_id in existing_run_topic.keys():
            for topic_id in expected_set - existing_run_topic[run_id]:
                missing.append((run_id, topic_id))
        return missing

    def _compute_aggregates(
        self,
        entries: List[LeaderboardEntry],
        phantom_defaults: List[tuple[str, str]],
    ) -> List[LeaderboardEntry]:
        """
        Compute "all" row aggregates from entries and phantom defaults.

        Args:
            entries: Per-topic entries to aggregate
            phantom_defaults: (run_id, topic_id) pairs to include in aggregation
                using MeasureSpec.default values (no actual entries created)

        Returns:
            List of aggregate "all" row entries, one per run_id
        """
        by_run: Dict[str, Dict[MeasureName, List[Any]]] = defaultdict(lambda: defaultdict(list))

        # Collect values from actual entries
        for e in entries:
            if e.topic_id == self.spec.all_topic_id:
                continue
            for k, v in e.values.items():
                by_run[e.run_id][k].append(v)

        # Include phantom defaults
        for run_id, _ in phantom_defaults:
            for ms in self.spec.measures:
                if ms.default is not None:
                    by_run[run_id][ms.name].append(ms.default)

        # Build aggregate rows
        all_rows: List[LeaderboardEntry] = []
        for run_id, m2vals in by_run.items():
            agg_vals: Dict[MeasureName, Any] = {}
            for ms in self.spec.measures:
                vals = m2vals.get(ms.name, [])
                if vals:
                    agg_vals[ms.name] = ms.aggregate(vals)
            all_rows.append(LeaderboardEntry(run_id=run_id, topic_id=self.spec.all_topic_id, values=agg_vals))

        return all_rows

    def build(
        self,
        expected_topic_ids: Optional[Sequence[str]] = None,
        on_missing: OnMissing = "default",
    ) -> Leaderboard:
        """
        Build a Leaderboard with synthetic per-run `all_topic_id` rows.

        The returned Leaderboard contains:
        - all per-topic rows that were added
        - plus one additional row per run_id with topic_id == spec.all_topic_id

        Args:
            expected_topic_ids: If provided, handles missing (run_id, topic_id) pairs.
            on_missing: When expected_topic_ids is provided and gaps exist:
                - "default": silently create per-topic entries with defaults
                - "warn": create per-topic entries with defaults and print warning
                - "fix_aggregate": only fill defaults for "all" row aggregation (no per-topic entries)
                - "error": raise ValueError listing missing (run_id, topic_id) pairs
        """
        # Step 1: Detect missing pairs
        all_missing: List[tuple[str, str]] = []
        if expected_topic_ids is not None:
            all_missing = self._detect_missing_run_topic(expected_topic_ids)

        # Step 2: Handle missing based on mode
        filled_rows: List[LeaderboardEntry] = []
        phantom_defaults: List[tuple[str, str]] = []

        if all_missing:
            formatted_pairs = [f"({r}, {t})" for r, t in sorted(all_missing)]
            if on_missing == "error":
                raise ValueError(
                    f"Missing leaderboard entries for {len(all_missing)} (run_id, topic_id) pair(s): {format_preview(formatted_pairs)}"
                )

            if on_missing == "warn":
                print(f"Leaderboard Warning: {len(all_missing)} missing entries: {format_preview(formatted_pairs)}", file=sys.stderr)

            if on_missing in ("default", "warn"):
                # Create actual per-topic entries
                default_values = {ms.name: ms.default for ms in self.spec.measures if ms.default is not None}
                if default_values:
                    for run_id, topic_id in all_missing:
                        filled_rows.append(LeaderboardEntry(run_id=run_id, topic_id=topic_id, values=default_values))
            elif on_missing == "fix_aggregate":
                phantom_defaults = all_missing

        # Step 3: Compute aggregates
        all_entries = self._rows + filled_rows
        all_rows = self._compute_aggregates(all_entries, phantom_defaults)

        return Leaderboard(
            measures=self.spec.names,
            entries=tuple(all_entries + all_rows),
            all_topic_id=self.spec.all_topic_id,
        )


#  === Example aggregators (optional helpers) ====

def mean_of_ints(values: Sequence[Any]) -> float:
    """Aggregate numeric values via arithmetic mean (values cast to float)."""
    return mean(float(v) for v in values)


def mean_of_bools(values: Sequence[Any]) -> float:
    """Aggregate booleans via mean of {0.0, 1.0}."""
    return mean(1.0 if bool(v) else 0.0 for v in values)
