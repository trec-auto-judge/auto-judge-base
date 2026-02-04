"""Format detection for leaderboard TSV files.

Provides helpful error messages when format mismatches occur by analyzing
file content to suggest the correct format.
"""

from dataclasses import dataclass
from typing import List, Literal

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
