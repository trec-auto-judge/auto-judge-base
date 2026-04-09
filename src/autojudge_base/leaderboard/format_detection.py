"""Format detection for leaderboard TSV files.

Provides helpful error messages when format mismatches occur by analyzing
file content to suggest the correct format.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Set

LeaderboardFormat = Literal["trec_eval", "tot", "ir_measures", "ranking", "unknown"]

# Format specifications
FORMAT_SPECS = {
    "trec_eval": (3, "{measure} {topic} {value}"),
    "tot": (4, "{run} {measure} {topic} {value}"),
    "ir_measures": (4, "{run} {topic} {measure} {value}"),
    "ranking": (6, "{topic} Q0 {doc_id} {rank} {score} {run}"),
}


@dataclass
class FormatHint:
    """Hint about detected TSV format."""
    possible_formats: List[LeaderboardFormat]
    reason: str
    has_header: bool


def detect_format(lines: List[str]) -> FormatHint:
    """Detect possible TSV formats based on column count and headers.

    Format specs:
        - trec_eval (3 cols): {measure} {topic} {value}
        - ir_measures (4 cols): {run} {topic} {measure} {value}
        - tot (4 cols): {run} {measure} {topic} {value}
        - ranking (6 cols): {topic} Q0 {doc_id} {rank} {score} {run}

    Args:
        lines: Lines from the file (first 10-20 are sufficient)

    Returns:
        FormatHint with list of possible formats and reason
    """
    if not lines:
        return FormatHint([], "Empty file", False)

    # Parse first few non-empty lines
    sample_rows = []
    for line in lines[:20]:
        line = line.strip()
        if line and not line.startswith("#"):
            parts = line.split()
            sample_rows.append(parts)

    if not sample_rows:
        return FormatHint([], "No data rows", False)

    # Check if first row looks like a header
    RUN_ID_NAMES = {"run_id", "runtag", "run", "runid", "system"}
    TOPIC_ID_NAMES = {"topic_id", "request_id", "query_id", "narrative_id", "topicid", "queryid"}
    METRIC_NAMES = {"metric", "measure", "eval_metric"}

    first_row_lower = [col.lower() for col in sample_rows[0]]

    def is_numeric(val: str) -> bool:
        try:
            float(val)
            return True
        except ValueError:
            return False

    # If last column is numeric, first row is likely data (value column)
    # If last column is non-numeric, first row might be header
    has_numeric_last = is_numeric(sample_rows[0][-1]) if sample_rows[0] else False

    # Check for known header names
    has_run_col = any(col in RUN_ID_NAMES for col in first_row_lower)
    has_topic_col = any(col in TOPIC_ID_NAMES for col in first_row_lower)
    has_metric_col = any(col in METRIC_NAMES for col in first_row_lower)

    # Detect header
    has_header = not has_numeric_last and (has_run_col or has_topic_col or has_metric_col)

    # Get column count from data rows
    data_rows = sample_rows[1:] if has_header and len(sample_rows) > 1 else sample_rows
    if not data_rows:
        return FormatHint([], "No data rows after header", has_header)

    col_counts = [len(row) for row in data_rows]
    typical_cols = max(set(col_counts), key=col_counts.count)

    # Determine possible formats based on column count
    if typical_cols == 3:
        return FormatHint(
            ["trec_eval"],
            "3 columns matches trec_eval: {measure} {topic} {value}",
            has_header,
        )

    if typical_cols == 4:
        # Can't reliably distinguish tot from ir_measures without more info
        return FormatHint(
            ["tot", "ir_measures"],
            "4 columns could be tot: {run} {measure} {topic} {value} "
            "OR ir_measures: {run} {topic} {measure} {value}",
            has_header,
        )

    if typical_cols == 6:
        # Check for Q0 marker
        q0_count = sum(1 for row in data_rows[:5] if len(row) > 1 and row[1] == "Q0")
        if q0_count > 0:
            return FormatHint(
                ["ranking"],
                "6 columns with Q0: ranking {topic} Q0 {doc_id} {rank} {score} {run}",
                has_header,
            )
        return FormatHint(
            ["ranking"],
            "6 columns, possibly ranking format",
            has_header,
        )

    return FormatHint(
        [],
        f"Unusual column count: {typical_cols}",
        has_header,
    )


@dataclass
class FormatGuess:
    """Educated guess about format based on topic matching."""
    likely_format: LeaderboardFormat
    confidence: str  # "high", "medium", "low"
    topic_col_matches: dict[int, int]  # col_index -> match_count
    numeric_cols: List[int]  # columns that appear numeric
    reason: str


def guess_format_by_topics(
    lines: List[str],
    known_topics: Set[str],
    has_header: bool = False,
) -> FormatGuess:
    """
    Make educated guess about tot vs ir_measures based on topic matching.

    For 4-column files:
    - ir_measures: {run} {topic} {measure} {value} → topic is col 1
    - tot: {run} {measure} {topic} {value} → topic is col 2

    Counts matches in each column position against known topics.
    Column with most matches is likely the topic column.
    Column with only numeric values is likely the value column.

    This is a diagnostic hint, not reliable auto-detection.

    Args:
        lines: Lines from the file
        known_topics: Set of expected topic IDs
        has_header: Whether first line is a header (skip it)

    Returns:
        FormatGuess with likely format and diagnostic details
    """
    if not lines or not known_topics:
        return FormatGuess(
            likely_format="unknown",
            confidence="low",
            topic_col_matches={},
            numeric_cols=[],
            reason="Empty file or no known topics provided",
        )

    # Parse data rows
    data_rows = []
    for i, line in enumerate(lines):
        if has_header and i == 0:
            continue
        line = line.strip()
        if line and not line.startswith("#"):
            parts = line.split()
            if parts:
                data_rows.append(parts)

    if not data_rows:
        return FormatGuess(
            likely_format="unknown",
            confidence="low",
            topic_col_matches={},
            numeric_cols=[],
            reason="No data rows found",
        )

    # Get typical column count
    col_counts = [len(row) for row in data_rows]
    num_cols = max(set(col_counts), key=col_counts.count)

    if num_cols != 4:
        # Not a tot/ir_measures ambiguity case
        return FormatGuess(
            likely_format="unknown",
            confidence="low",
            topic_col_matches={},
            numeric_cols=[],
            reason=f"Column count {num_cols} != 4, not tot/ir_measures",
        )

    # Count topic matches per column
    topic_matches: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    numeric_counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

    for row in data_rows:
        if len(row) != 4:
            continue
        for col_idx, val in enumerate(row):
            if val in known_topics:
                topic_matches[col_idx] += 1
            try:
                float(val)
                numeric_counts[col_idx] += 1
            except ValueError:
                pass

    total_rows = len([r for r in data_rows if len(r) == 4])

    # Find which column is likely numeric (value column)
    numeric_cols = [
        col for col, count in numeric_counts.items()
        if total_rows > 0 and count >= total_rows * 0.9  # 90%+ numeric
    ]

    # Build diagnostic parts
    diag_parts = []
    for col_idx in range(4):
        matches = topic_matches[col_idx]
        pct = (matches / len(known_topics) * 100) if known_topics else 0
        diag_parts.append(f"col {col_idx}: {matches}/{len(known_topics)} topics ({pct:.0f}%)")

    if numeric_cols:
        diag_parts.append(f"numeric columns: {numeric_cols}")

    # Determine likely format based on which column has topic matches
    # ir_measures: col 1 is topic
    # tot: col 2 is topic
    col1_matches = topic_matches[1]
    col2_matches = topic_matches[2]

    if col1_matches > col2_matches and col1_matches > 0:
        likely = "ir_measures"
        confidence = "high" if col1_matches > len(known_topics) * 0.5 else "medium"
        reason = f"Column 1 has most topic matches ({col1_matches}), suggesting ir_measures format. " + "; ".join(diag_parts)
    elif col2_matches > col1_matches and col2_matches > 0:
        likely = "tot"
        confidence = "high" if col2_matches > len(known_topics) * 0.5 else "medium"
        reason = f"Column 2 has most topic matches ({col2_matches}), suggesting tot format. " + "; ".join(diag_parts)
    elif col1_matches == col2_matches and col1_matches > 0:
        likely = "unknown"
        confidence = "low"
        reason = f"Columns 1 and 2 have equal topic matches ({col1_matches}), ambiguous. " + "; ".join(diag_parts)
    else:
        likely = "unknown"
        confidence = "low"
        reason = f"No topic matches found in expected columns. " + "; ".join(diag_parts)

    return FormatGuess(
        likely_format=likely,
        confidence=confidence,
        topic_col_matches=topic_matches,
        numeric_cols=numeric_cols,
        reason=reason,
    )


def format_error_with_hint(
    specified_format: str,
    expected_cols: int,
    actual_cols: int,
    line: str,
    lines: List[str],
) -> str:
    """Generate helpful error message when format doesn't match.

    Args:
        specified_format: The format user specified
        expected_cols: Expected column count for that format
        actual_cols: Actual column count found
        line: The problematic line
        lines: All lines (for format detection)

    Returns:
        Error message with suggestion
    """
    hint = detect_format(lines)

    base_msg = (
        f"{specified_format} format expects {expected_cols} fields, "
        f"got {actual_cols}: {line!r}"
    )

    suggestions = []

    # Suggest formats that match the actual column count
    matching_formats = [f for f in hint.possible_formats if f != specified_format]
    if matching_formats:
        format_list = " or ".join(f"--eval-format {f}" for f in matching_formats)
        suggestions.append(f"Try: {format_list}")
        suggestions.append(f"Detected: {hint.reason}")

    if hint.has_header:
        suggestions.append("Note: First line appears to be a header - consider --truth-header or --eval-header")

    if suggestions:
        return base_msg + "\n\n" + "\n".join(suggestions)

    return base_msg


def check_format_mismatch(
    file_path: Path,
    specified_format: str,
    known_topics: Set[str],
    has_header: bool = False,
    sample_lines: int = 50,
) -> Optional[str]:
    """
    Pre-hoc check for format mismatch before parsing.

    Reads sample lines from file and guesses format based on topic matching.
    Returns a warning message if the guessed format differs from specified format.

    Args:
        file_path: Path to the leaderboard file
        specified_format: The format specified by user (e.g., "ir_measures", "tot")
        known_topics: Set of expected topic IDs from truth/requests
        has_header: Whether file has a header row
        sample_lines: Number of lines to sample for detection

    Returns:
        Warning message string if mismatch detected, None otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [f.readline() for _ in range(sample_lines)]
            lines = [line for line in lines if line]  # Remove empty reads
    except (OSError, UnicodeDecodeError) as e:
        return f"Warning: Could not read {file_path.name}: {e}"

    guess = guess_format_by_topics(lines, known_topics, has_header=has_header)

    # Only warn if we have a confident guess that differs from specified
    if guess.likely_format == "unknown":
        return None

    if guess.likely_format != specified_format:
        return (
            f"Warning: {file_path.name} appears to be '{guess.likely_format}' format "
            f"but eval-format '{specified_format}' was specified.\n"
            f"  Hint: {guess.reason}\n"
            f"  Try: --eval-format {guess.likely_format}"
        )

    return None
