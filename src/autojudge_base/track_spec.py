"""Track-year submission specs (the DATA layer).

A `TrackSpec` is the per-track-year *policy overlay* on top of the permissive
Report/Request shape bindings: which metadata fields are mandatory, which citation
representation(s) are allowed, the docid pattern, and the length/citation limits.
Specs are DATA (`track_specs.yml`), bumped yearly; a custom spec file can be loaded
via `load_spec_file()` to cover a track before its base release.

This module holds only the spec data + registry + small pure helpers, and imports
`report_sentences` (not `report`) so it stays free of any runtime dependency on the
Report class. The behavior lives in sibling modules:
- verification: `autojudge_base.report_spec_verification` (`verify`, `check`)
- conversion:   `autojudge_base.report` (`convert`, and the deprecated `to_rag`/`to_ragtime`)

For backwards compatibility those names are still importable from here (see
`__getattr__` at the bottom): `from autojudge_base.track_spec import verify, convert`.

`sentence_type` is a YAML string tag (`rag24`/`ragtime`/`neuclir`) resolved to a class
here, so specs stay serializable and carry no hard class references.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from autojudge_base.report_sentences import (
    NeuclirReportSentence,
    Rag24ReportSentence,
    RagtimeReportSentence,
)

# --- sentence-type tag <-> class -------------------------------------------------

_TAG_TO_CLASS = {
    "rag24": Rag24ReportSentence,      # citations = zero-indexed int positions into references
    "ragtime": RagtimeReportSentence,  # citations = {doc_id: score}
    "neuclir": NeuclirReportSentence,  # citations = [doc_id, ...]
}
_CLASS_TO_TAG = {cls: tag for tag, cls in _TAG_TO_CLASS.items()}


def sentence_class_for(tag: str) -> type:
    """Resolve a sentence_type tag to its ReportSentence class."""
    try:
        return _TAG_TO_CLASS[tag]
    except KeyError:
        raise KeyError(f"Unknown sentence_type tag {tag!r}; known: {sorted(_TAG_TO_CLASS)}")


def tag_of_sentence(sentence) -> Optional[str]:
    """The sentence_type tag for a sentence instance, or None if unrecognized."""
    return _CLASS_TO_TAG.get(type(sentence))


# --- the spec --------------------------------------------------------------------


# Finding categories treated as WARNINGS (smells), not failures, unless a spec overrides
# `smells`. Category names are defined in report_spec_verification (CAT_*) and the CLI
# coverage checks (e.g. "duplicate_topics").
DEFAULT_SMELLS = ("empty_answer", "blank_sentence", "duplicate_topics", "references_duplicate")


@dataclass(frozen=True)
class TrackSpec:
    """One track-year's report-submission wire spec (see track_specs.yml for field docs)."""

    track: str
    task: str
    sentence_type: Tuple[str, ...]           # accepted tags; [0] is canonical for emit
    sentences_key: str                        # "answer" | "responses"
    topic_id_field: str                       # "narrative_id" | "topic_id"
    mandatory_metadata: Tuple[str, ...]
    forbid_extra_metadata: bool = False
    require_exact_narrative: bool = False
    run_id_max_len: Optional[int] = None
    docid_pattern: Optional[str] = None
    collection_ids: Optional[Tuple[str, ...]] = None
    references_kind: str = "cited_only"       # "cited_only" | "retrieval_list" | "none" | "ignore"
    references_optional: bool = False         # if True, absent/empty references are OK; only a
    references_max: Optional[int] = None      #   non-empty references array is checked (RAGTIME)
    max_citations_per_sentence: Optional[int] = None
    length_unit: str = "words"                # "words" | "chars"
    length_limit: Optional[int] = None
    length_limit_request_field: Optional[str] = None
    smells: Tuple[str, ...] = ()              # check categories treated as WARNINGS, not failures

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrackSpec":
        st = d["sentence_type"]
        if isinstance(st, str):               # scalar shorthand -> 1-element list
            st = [st]
        coll = d.get("collection_ids")
        return cls(
            track=d["track"],
            task=d["task"],
            sentence_type=tuple(st),
            sentences_key=d["sentences_key"],
            topic_id_field=d["topic_id_field"],
            mandatory_metadata=tuple(d["mandatory_metadata"]),
            forbid_extra_metadata=bool(d.get("forbid_extra_metadata", False)),
            require_exact_narrative=bool(d.get("require_exact_narrative", False)),
            run_id_max_len=d.get("run_id_max_len"),
            docid_pattern=d.get("docid_pattern"),
            collection_ids=tuple(coll) if coll else None,
            references_kind=d.get("references_kind", "cited_only"),
            references_optional=bool(d.get("references_optional", False)),
            references_max=d.get("references_max"),
            max_citations_per_sentence=d.get("max_citations_per_sentence"),
            length_unit=d.get("length_unit", "words"),
            length_limit=d.get("length_limit"),
            length_limit_request_field=d.get("length_limit_request_field"),
            # unspecified -> the default smell set; `smells: []` disables all smells
            smells=tuple(DEFAULT_SMELLS if d.get("smells") is None else d["smells"]),
        )

    @property
    def emit_sentence_type(self) -> str:
        """The canonical citation representation to emit (first accepted tag)."""
        return self.sentence_type[0]


# --- registry --------------------------------------------------------------------

_BUNDLED_SPECS_PATH = Path(__file__).parent / "track_specs.yml"


def load_spec_file(path) -> Dict[str, TrackSpec]:
    """Load a track_specs.yml file into {track_id: TrackSpec}."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {k: TrackSpec.from_dict(v) for k, v in (data.get("specs") or {}).items()}


#: Built-in specs (bumped yearly with autojudge-base). Custom specs override via load_spec_file.
SPECS: Dict[str, TrackSpec] = load_spec_file(_BUNDLED_SPECS_PATH)


def get_spec(track: str, task: Optional[str] = None) -> TrackSpec:
    """Look up a built-in spec by track id (e.g. 'rag26'); optionally assert its task."""
    spec = SPECS.get(track)
    if spec is None:
        raise KeyError(f"No track spec for track={track!r}. Known: {sorted(SPECS)}")
    if task is not None and spec.task != task:
        raise KeyError(f"Track {track!r} spec is task={spec.task!r}, not {task!r}")
    return spec


# --- length counting (pure text helpers, shared by verify + chop/export) ---------
# Matches the official RAGTIME validator (hltcoe/rag-run-validator, validate_length):
# the sentence texts are FIRST joined with single spaces, THEN measured -- 'chars' as
# NFKC-normalized Unicode characters, 'words' as whitespace-split tokens. Consecutive
# whitespace is NOT collapsed (the validator counts it as-is).

def _joined(texts) -> str:
    return " ".join(texts)


def _word_count(texts) -> int:
    # whitespace-split tokens of the space-joined text (validator's 'words' mode)
    return len(_joined(texts).split())


def _char_count(texts) -> int:
    # NFKC-normalized character length of the space-joined text (validator's 'characters' mode)
    return len(unicodedata.normalize("NFKC", _joined(texts)))


def length_count(texts, unit: str) -> int:
    """Report length in the spec's unit, matching the official RAGTIME validator: join the
    sentence texts with single spaces, then count NFKC 'chars' or whitespace-split 'words'."""
    return _word_count(texts) if unit == "words" else _char_count(texts)


# --- backwards-compat re-exports -------------------------------------------------
# verify/check moved to report_spec_verification; convert/to_rag/to_ragtime moved to
# report. Lazily forward the old import paths (no import-time cycle).
def __getattr__(name: str):
    if name in ("verify", "check", "findings", "TrackSpecVerification", "TrackSpecVerificationError"):
        from autojudge_base import report_spec_verification as _v
        return getattr(_v, name)
    if name in ("convert", "to_rag", "to_ragtime"):
        from autojudge_base import report as _r
        return getattr(_r, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
 