"""Track-year submission specs: verify and convert Reports against organizer wire formats.

The `Report`/`Request`/`Document` models are permissive *shape bindings* — a superset
that must LOAD any track's submission. A `TrackSpec` is the per-track-year *policy
overlay*: which metadata fields are mandatory, which citation representation(s) are
allowed, the docid pattern, and the length/citation limits. Specs are DATA
(`track_specs.yml`), bumped yearly; a custom spec file can be loaded via
`load_spec_file()` to cover a track before its base release.

Design invariants:
- The spec is always a CALLER-SUPPLIED parameter, never sniffed from the (self-reported,
  unreliable) data. `verify(report, spec=None)` with `spec=None` falls back to
  structural-only checks; passing a spec turns on the strict, collection-aware policy.
- Fully backwards compatible: nothing here changes existing `Report` behavior; the
  `Report.verify_rag/verify_ragtime` methods gain an optional trailing `spec=None`.

`sentence_type` is a YAML string tag (`rag24`/`ragtime`/`neuclir`) resolved to a class
here, so specs stay serializable and carry no hard class references.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from autojudge_base.report import (
    NeuclirReportSentence,
    Rag24ReportSentence,
    RagtimeReportSentence,
    Report,
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
    references_kind: str = "cited_only"       # "cited_only" | "retrieval_list" | "none"
    references_max: Optional[int] = None
    max_citations_per_sentence: Optional[int] = None
    length_unit: str = "words"                # "words" | "chars"
    length_limit: Optional[int] = None
    length_limit_request_field: Optional[str] = None

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
            references_max=d.get("references_max"),
            max_citations_per_sentence=d.get("max_citations_per_sentence"),
            length_unit=d.get("length_unit", "words"),
            length_limit=d.get("length_limit"),
            length_limit_request_field=d.get("length_limit_request_field"),
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


# --- verification ----------------------------------------------------------------


def _sentences(report: Report, use_answer: bool):
    s = report.answer if use_answer else report.responses
    if s is None:
        raise RuntimeError("Report has neither responses nor answer to verify")
    return s


def _cited_doc_ids(sentence, references: List[str]) -> List[str]:
    tag = tag_of_sentence(sentence)
    cits = sentence.citations
    if tag == "rag24":
        return [references[i] for i in (cits or []) if isinstance(i, int) and 0 <= i < len(references)]
    if tag == "neuclir":
        return list(cits or [])
    if tag == "ragtime":
        return list((cits or {}).keys())
    return []


def _citation_count(sentence) -> int:
    c = sentence.citations
    return len(c) if c else 0


def _word_count(texts) -> int:
    l1 = sum(len(unicodedata.normalize("NFKC", t).split()) for t in texts)
    l2 = sum(len(t.split()) for t in texts)
    l3 = sum(len(re.findall(r"\w+", unicodedata.normalize("NFKC", t))) for t in texts)
    return max(l1, l2, l3)


def _char_count(texts) -> int:
    return sum(len(unicodedata.normalize("NFKC", t)) for t in texts)


def _verify_metadata(report: Report, spec: TrackSpec) -> None:
    md = report.metadata
    for key in spec.mandatory_metadata:
        val = getattr(md, key, None)
        if val is None or (isinstance(val, str) and not val.strip()):
            raise RuntimeError(
                f"{spec.track}: metadata.{key} must be present and non-empty, got {val!r}"
            )
    if spec.run_id_max_len is not None:
        rid = md.run_id or ""
        if len(rid) > spec.run_id_max_len:
            raise RuntimeError(
                f"{spec.track}: run_id length {len(rid)} exceeds {spec.run_id_max_len}"
            )
    # NOTE: forbid_extra_metadata is a RAW-WIRE check (the loaded model always
    # materializes optional fields as None and syncs id aliases), so it belongs to
    # file/dict-level validators, not this model-level pass.


def _verify_sentences(report: Report, spec: TrackSpec, sentences) -> None:
    if not sentences:
        raise RuntimeError(f"{spec.track}: report has no sentences")
    refs = report.references or []
    maxc = spec.max_citations_per_sentence
    pat = re.compile(spec.docid_pattern) if spec.docid_pattern else None
    for i, s in enumerate(sentences):
        tag = tag_of_sentence(s)
        if tag not in spec.sentence_type:
            raise RuntimeError(
                f"{spec.track}: answer[{i}] is {tag or type(s).__name__}, "
                f"not an accepted sentence_type {list(spec.sentence_type)}"
            )
        if not isinstance(s.text, str) or not s.text.strip():
            raise RuntimeError(f"{spec.track}: answer[{i}].text must be a non-empty string")
        n = _citation_count(s)
        if maxc is not None and n > maxc:
            raise RuntimeError(f"{spec.track}: answer[{i}] has {n} citations (max {maxc})")
        if tag == "rag24":
            for c in (s.citations or []):
                if not isinstance(c, int) or isinstance(c, bool) or not (0 <= c < len(refs)):
                    raise RuntimeError(
                        f"{spec.track}: answer[{i}] citation {c!r} is not a valid reference index"
                    )
        elif tag == "ragtime":
            for did, score in (s.citations or {}).items():
                if not isinstance(score, (int, float)) or isinstance(score, bool):
                    raise RuntimeError(
                        f"{spec.track}: answer[{i}] citation {did!r} score must be numeric, got {score!r}"
                    )
        if pat is not None:
            for did in _cited_doc_ids(s, refs):
                if not pat.match(did):
                    raise RuntimeError(
                        f"{spec.track}: cited doc-id {did!r} does not match docid pattern {spec.docid_pattern}"
                    )


def _verify_references(report: Report, spec: TrackSpec, sentences) -> None:
    refs = report.references
    cited = set()
    for s in sentences:
        cited.update(_cited_doc_ids(s, refs or []))

    if spec.references_kind == "none":
        if refs:
            raise RuntimeError(
                f"{spec.track}: references must be absent/empty (references_kind=none)"
            )
        return

    if refs is None:
        raise RuntimeError(f"{spec.track}: a references array is required")
    if len(set(refs)) != len(refs):
        raise RuntimeError(f"{spec.track}: references contains duplicate doc-ids")
    if spec.docid_pattern:
        pat = re.compile(spec.docid_pattern)
        for d in refs:
            if not pat.match(d):
                raise RuntimeError(
                    f"{spec.track}: reference doc-id {d!r} does not match docid pattern {spec.docid_pattern}"
                )

    if spec.references_kind == "cited_only":
        if set(refs) != cited:
            missing = sorted(cited - set(refs))
            uncited = sorted(set(refs) - cited)
            raise RuntimeError(
                f"{spec.track}: references must equal the cited set "
                f"(cited-but-absent {missing}, present-but-uncited {uncited})"
            )
    elif spec.references_kind == "retrieval_list":
        if not cited.issubset(set(refs)):
            raise RuntimeError(f"{spec.track}: some cited doc-ids are not in references")
        if spec.references_max is not None and len(refs) > spec.references_max:
            raise RuntimeError(
                f"{spec.track}: references has {len(refs)} entries (max {spec.references_max})"
            )


def _verify_length(report: Report, spec: TrackSpec, sentences, request) -> None:
    texts = [s.text for s in sentences if isinstance(getattr(s, "text", None), str)]
    count = _word_count(texts) if spec.length_unit == "words" else _char_count(texts)
    limit = spec.length_limit
    if limit is None and spec.length_limit_request_field is not None:
        if request is None:
            return  # per-request limit but no Request supplied -> cannot verify; skip silently
        limit = getattr(request, spec.length_limit_request_field, None)
    if limit is not None and count > limit:
        raise RuntimeError(f"{spec.track}: report is {count} {spec.length_unit} (limit {limit})")


def _verify_structural(report: Report, sentences) -> bool:
    """Format-universal checks used when no spec is supplied (spec=None sniff path)."""
    if not sentences:
        raise RuntimeError("report has no sentences")
    refs = report.references or []
    for i, s in enumerate(sentences):
        if not isinstance(getattr(s, "text", None), str) or not s.text.strip():
            raise RuntimeError(f"answer[{i}].text must be a non-empty string")
        if tag_of_sentence(s) == "rag24":
            for c in (s.citations or []):
                if not isinstance(c, int) or isinstance(c, bool) or not (0 <= c < len(refs)):
                    raise RuntimeError(f"answer[{i}] citation index {c!r} is out of range")
    return True


def verify(report: Report, spec: Optional[TrackSpec] = None, *, request=None, use_answer: bool = False) -> bool:
    """Verify a Report against a TrackSpec (or structural-only if spec is None).

    Raises RuntimeError on the first violation; returns True if valid. `request` is
    only needed for tracks whose length limit comes from the Request (RAGTIME).
    """
    sentences = _sentences(report, use_answer)
    if spec is None:
        return _verify_structural(report, sentences)
    _verify_metadata(report, spec)
    _verify_sentences(report, spec, sentences)
    _verify_references(report, spec, sentences)
    _verify_length(report, spec, sentences, request)
    return True


# --- conversion ------------------------------------------------------------------


def _convert(report: Report, spec: TrackSpec) -> Report:
    """Build a NEW Report whose sentences use spec.emit_sentence_type and whose
    references follow spec.references_kind. Source citations are read via the
    format-agnostic resolver, so any input representation converts."""
    tag = spec.emit_sentence_type
    resolved = report.get_sentences_with_citations()  # -> Neuclir sentences (doc-id lists)

    references: List[str] = []
    index: Dict[str, int] = {}
    for s in resolved:
        for d in (s.citations or []):
            if d not in index:
                index[d] = len(references)
                references.append(d)

    new_sentences = []
    for s in resolved:
        dids = s.citations or []
        if tag == "rag24":
            new_sentences.append(Rag24ReportSentence(
                text=s.text, citations=[index[d] for d in dids],
                metadata=s.metadata, evaldata=s.evaldata))
        elif tag == "neuclir":
            new_sentences.append(NeuclirReportSentence(
                text=s.text, citations=list(dids),
                metadata=s.metadata, evaldata=s.evaldata))
        elif tag == "ragtime":
            new_sentences.append(RagtimeReportSentence(
                text=s.text, citations={d: 1.0 for d in dids},
                metadata=s.metadata, evaldata=s.evaldata))
        else:
            raise KeyError(f"Cannot emit unknown sentence_type tag {tag!r}")

    return Report(
        metadata=report.metadata.model_copy(),
        responses=new_sentences,
        references=None if spec.references_kind == "none" else references,
    )


def to_rag(report: Report, spec: Optional[TrackSpec] = None) -> Report:
    """Convert to a RAG report (default: RAG 2025 generation, integer-index citations)."""
    return _convert(report, spec or SPECS["rag25"])


def to_ragtime(report: Report, spec: Optional[TrackSpec] = None) -> Report:
    """Convert to a RAGTIME report (default: RAGTIME 2025 repgen, doc-id->score citations)."""
    return _convert(report, spec or SPECS["ragtime25"])
