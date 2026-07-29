"""Emit internal Reports to a TREC RAG-family submission JSONL (spec-driven).

The reusable emit layer beneath report_tool's `convert` command: reshape a Report to a
target track's wire format and write it out. Depends only on autojudge_base.
"""
import json
from pathlib import Path
from typing import Any, Dict

from autojudge_base.report import convert, submission_dict


def report_to_submission_dict(report, spec) -> Dict[str, Any]:
    """Reshape one Report to `spec`'s wire dict: `convert` re-expresses the citation
    representation / references / metadata for the target track, then `submission_dict`
    emits exactly the spec's `sentences_key` and `mandatory_metadata`. `spec` may be a
    TrackSpec or a track-id string."""
    return submission_dict(convert(report, spec), spec)


def write_submission_output(reports, path, spec) -> int:
    """Write `reports` converted to `spec` as a JSONL file (one object per line).
    Creates parent directories. Returns the number of reports written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in reports:
            obj = report_to_submission_dict(r, spec)
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(reports)
