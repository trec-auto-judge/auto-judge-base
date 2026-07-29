"""Verify a Report against a track-year submission spec.

Separate module (mirroring qrels/verification.py and leaderboard/verification.py) so
report.py can import it at top level: it references the sentence models via
`track_spec.tag_of_sentence` and never imports Report at runtime (only under
TYPE_CHECKING), so there is no import cycle.

Every check is tagged with a CATEGORY (see the CAT_* constants). A spec may mark some
categories as SMELLS (`TrackSpec.smells`): those findings become WARNINGS rather than
hard errors -- surfaced, but not a failure.

Entry points:
- `verify(report, spec)`  -> bool, RAISING TrackSpecVerificationError on the first hard
  ERROR (smell warnings never raise). Fail-fast emit gate.
- `check(report, spec)`   -> list[str], the hard ERRORS only (back-compat).
- `findings(report, spec)`-> (errors, warnings), both lists (used by the CLI).

`spec` may be a `TrackSpec`, a track-id string (e.g. "rag26", resolved via SPECS), or
None (structural-only checks). The spec is always caller-supplied -- never sniffed.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

from autojudge_base.track_spec import TrackSpec, get_spec, length_count, tag_of_sentence

if TYPE_CHECKING:  # for type hints only; no runtime dependency on report.py
    from autojudge_base.report import Report


class TrackSpecVerificationError(RuntimeError):
    """A report violates its track spec. Subclasses RuntimeError so existing callers
    that catch RuntimeError from `verify` keep working."""


# Finding categories. A spec lists any of these in `smells` to downgrade them to
# warnings. Keep names stable -- they are the configuration surface in track_specs.yml.
CAT_EMPTY_ANSWER = "empty_answer"        # no answer / empty list / all sentences blank
CAT_BLANK_SENTENCE = "blank_sentence"    # one sentence (among others) has empty text
CAT_CITATION_FORMAT = "citation_format"  # sentence uses a citation format the track rejects
CAT_CITATION_COUNT = "citation_count"    # too many citations on one sentence
CAT_REFERENCE_INDEX = "reference_index"  # rag24 citation index out of range
CAT_CITATION_SCORE = "citation_score"    # ragtime citation score not numeric
CAT_DOCID_PATTERN = "docid_pattern"      # a cited/reference doc-id has the wrong id format
CAT_REF_PRESENT = "references_present"       # references given for a track that uses none
CAT_REF_MISSING = "references_missing"       # a required references array is absent
CAT_REF_DUPLICATE = "references_duplicate"   # references lists the same doc-id twice
CAT_REF_UNDECLARED = "references_undeclared"  # a cited doc-id is not listed in references
CAT_REF_UNCITED = "references_uncited"       # a references doc-id is never cited in the answer
CAT_REF_MAX = "references_max"               # references array longer than the allowed max
CAT_METADATA = "metadata"               # a mandatory metadata field is missing/empty
CAT_RUN_ID = "run_id_length"            # run_id too long
CAT_LENGTH = "length"                   # answer text over the length limit
CAT_STRUCTURAL = "structural"           # spec=None fallback checks

# `spec` argument: a resolved TrackSpec, a track-id string, or None.
SpecArg = Union[TrackSpec, str, None]
# add(category, message) -> route a finding to errors or warnings by the spec's smells.
AddFn = Callable[[str, str], None]


def _resolve_spec(spec: SpecArg) -> Optional[TrackSpec]:
    if spec is None or isinstance(spec, TrackSpec):
        return spec
    if isinstance(spec, str):
        return get_spec(spec)
    raise TypeError(f"spec must be a TrackSpec, a track-id str, or None; got {type(spec).__name__}")


# --- per-aspect checks (report a finding via add(category, message)) ------------

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


def _check_metadata(report: "Report", spec: TrackSpec, add: AddFn) -> None:
    md = report.metadata
    for key in spec.mandatory_metadata:
        val = getattr(md, key, None)
        if val is None or (isinstance(val, str) and not val.strip()):
            add(CAT_METADATA,
                f"{spec.track}: metadata.{key} is a required field but is missing or empty (got {val!r})")
    if spec.run_id_max_len is not None:
        rid = md.run_id or ""
        if len(rid) > spec.run_id_max_len:
            add(CAT_RUN_ID,
                f"{spec.track}: run_id is {len(rid)} characters, over the {spec.run_id_max_len}-character limit")
    # NOTE: forbid_extra_metadata is a RAW-WIRE check (the loaded model always
    # materializes optional fields as None and syncs id aliases), so it belongs to
    # the file/dict-level validator, not this model-level pass.


def _check_sentences(report: "Report", spec: TrackSpec, sentences, add: AddFn) -> None:
    refs = report.references or []
    maxc = spec.max_citations_per_sentence
    key = spec.sentences_key  # "answer" (RAG) or "responses" (RAGTIME/DRAGUN)
    pat = re.compile(spec.docid_pattern) if spec.docid_pattern else None
    coll = f" for collection {spec.collection_ids[0]}" if spec.collection_ids else ""
    for i, s in enumerate(sentences):
        tag = tag_of_sentence(s)
        if tag not in spec.sentence_type:
            add(CAT_CITATION_FORMAT,
                f"{spec.track}: {key}[{i}] uses citation format '{tag or type(s).__name__}', which "
                f"{spec.track} does not accept (accepted citation formats: {list(spec.sentence_type)})")
            continue  # citation shape is unknown for an unaccepted type
        if not isinstance(s.text, str) or not s.text.strip():
            add(CAT_BLANK_SENTENCE,
                f"{spec.track}: {key}[{i}].text must be a non-empty string (this sentence has no text)")
        n = _citation_count(s)
        if maxc is not None and n > maxc:
            add(CAT_CITATION_COUNT,
                f"{spec.track}: {key}[{i}] has {n} citations (max {maxc}) -- too many for one sentence")
        if tag == "rag24":
            for c in (s.citations or []):
                if not isinstance(c, int) or isinstance(c, bool) or not (0 <= c < len(refs)):
                    add(CAT_REFERENCE_INDEX,
                        f"{spec.track}: {key}[{i}] cites reference index {c!r}, which is not a valid "
                        f"reference index (the references array has {len(refs)} entries)")
        elif tag == "ragtime":
            for did, score in (s.citations or {}).items():
                if not isinstance(score, (int, float)) or isinstance(score, bool):
                    add(CAT_CITATION_SCORE,
                        f"{spec.track}: {key}[{i}] cites doc {did!r} with a non-numeric relevance "
                        f"score {score!r} (RAGTIME citations map each doc-id to a numeric score)")
        if pat is not None:
            for did in _cited_doc_ids(s, refs):
                if not pat.match(did):
                    add(CAT_DOCID_PATTERN,
                        f"{spec.track}: cited doc-id {did!r} is not a valid document id{coll} "
                        f"(must match docid pattern {spec.docid_pattern})")


def _check_references(report: "Report", spec: TrackSpec, sentences, add: AddFn) -> None:
    if spec.references_kind == "ignore":
        return  # references not checked at all (e.g. auto-judge round-trip formats)

    refs = report.references
    cited = set()
    for s in sentences:
        cited.update(_cited_doc_ids(s, refs or []))

    if spec.references_kind == "none":
        if refs:
            add(CAT_REF_PRESENT, f"{spec.track}: this track uses no references array, but the report provided one")
        return

    # RAGTIME: references are optional -- absent or empty is fine; only a non-empty
    # references array is held to the cited-set / pattern rules below.
    if spec.references_optional and not refs:
        return

    if refs is None:
        add(CAT_REF_MISSING, f"{spec.track}: this track requires a references array, but the report has none")
        return
    if len(set(refs)) != len(refs):
        add(CAT_REF_DUPLICATE, f"{spec.track}: the references array lists the same document id more than once")
    if spec.docid_pattern:
        pat = re.compile(spec.docid_pattern)
        coll = f" for collection {spec.collection_ids[0]}" if spec.collection_ids else ""
        for d in refs:
            if not pat.match(d):
                add(CAT_DOCID_PATTERN,
                    f"{spec.track}: reference doc-id {d!r} is not a valid document id{coll} "
                    f"(must match docid pattern {spec.docid_pattern})")

    if spec.references_kind == "cited_only":
        # cited_only: the references array must list EXACTLY the documents cited in the answer
        missing = sorted(cited - set(refs))    # cited in the answer but absent from references
        uncited = sorted(set(refs) - cited)    # present in references but cited by no sentence
        if missing:
            add(CAT_REF_UNDECLARED,
                f"{spec.track}: {len(missing)} document(s) cited in the answer are not listed in "
                f"references (e.g. {missing[0]!r}); references must list exactly the cited documents")
        if uncited:
            add(CAT_REF_UNCITED,
                f"{spec.track}: {len(uncited)} document(s) in references are never cited in the answer "
                f"(e.g. {uncited[0]!r}); references must list exactly the cited documents")
    elif spec.references_kind == "retrieval_list":
        if not cited.issubset(set(refs)):
            add(CAT_REF_UNDECLARED, f"{spec.track}: some documents cited in the answer are not listed in references")
        if spec.references_max is not None and len(refs) > spec.references_max:
            add(CAT_REF_MAX,
                f"{spec.track}: the references array has {len(refs)} entries (max {spec.references_max})")


def _check_length(report: "Report", spec: TrackSpec, sentences, request, add: AddFn) -> None:
    texts = [s.text for s in sentences if isinstance(getattr(s, "text", None), str)]
    count = length_count(texts, spec.length_unit)
    limit = spec.length_limit
    if limit is None and spec.length_limit_request_field is not None:
        if request is None:
            return  # per-request limit but no Request supplied -> cannot verify; skip silently
        limit = getattr(request, spec.length_limit_request_field, None)
    if limit is not None and count > limit:
        add(CAT_LENGTH,
            f"{spec.track}: the answer text is too long -- {count} {spec.length_unit} (limit {limit})")


def _structural(report: "Report", sentences, add: AddFn) -> None:
    """Format-universal checks used when no spec is supplied (spec=None)."""
    refs = report.references or []
    for i, s in enumerate(sentences):
        if not isinstance(getattr(s, "text", None), str) or not s.text.strip():
            add(CAT_BLANK_SENTENCE, f"sentence[{i}].text must be a non-empty string")
        if tag_of_sentence(s) == "rag24":
            for c in (s.citations or []):
                if not isinstance(c, int) or isinstance(c, bool) or not (0 <= c < len(refs)):
                    add(CAT_REFERENCE_INDEX, f"sentence[{i}] citation index {c!r} is out of range")


# --- verification object (Qrels-style) ------------------------------------------

class TrackSpecVerification:
    """Accumulates a report's findings against a spec, split into `errors` (hard
    failures) and `warnings` (categories the spec lists in `smells`).

    `spec` is already resolved to a TrackSpec (or None for structural-only checks).
    """

    def __init__(self, report: "Report", spec: Optional[TrackSpec], *,
                 request=None, use_answer: bool = False):
        self.report = report
        self.spec = spec
        self.request = request
        self.use_answer = use_answer
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def _add(self, category: str, message: str) -> None:
        smells = self.spec.smells if self.spec is not None else ()
        (self.warnings if category in smells else self.errors).append(message)

    def run(self) -> "TrackSpecVerification":
        prefix = f"{self.spec.track}: " if self.spec is not None else ""
        key = self.spec.sentences_key if self.spec is not None else "response"
        sentences = self.report.answer if self.use_answer else self.report.responses

        # Distinguish an ABSENT answer from an EMPTY one, and stop there -- these are
        # the whole problem; the rest of the checks would just be noise.
        if sentences is None:
            self._add(CAT_EMPTY_ANSWER, f"{prefix}missing {key}")
            return self
        if not sentences:
            self._add(CAT_EMPTY_ANSWER, f"{prefix}empty {key} given (no sentences)")
            return self
        if all(not (isinstance(s.text, str) and s.text.strip()) for s in sentences):
            self._add(CAT_EMPTY_ANSWER, f"{prefix}empty {key} given (all sentences blank)")
            return self

        if self.spec is None:
            _structural(self.report, sentences, self._add)
            return self
        _check_metadata(self.report, self.spec, self._add)
        _check_sentences(self.report, self.spec, sentences, self._add)
        _check_references(self.report, self.spec, sentences, self._add)
        _check_length(self.report, self.spec, sentences, self.request, self._add)
        return self

    def raise_first(self) -> None:
        if self.errors:  # smells (warnings) never raise
            raise TrackSpecVerificationError(self.errors[0])


def _run(report, spec, request, use_answer) -> TrackSpecVerification:
    return TrackSpecVerification(
        report, _resolve_spec(spec), request=request, use_answer=use_answer
    ).run()


def findings(report: "Report", spec: SpecArg = None, *, request=None,
             use_answer: bool = False) -> Tuple[List[str], List[str]]:
    """Validate a Report and return (errors, warnings).

    `errors` are hard failures; `warnings` are findings whose category the spec lists in
    `smells`. An empty `errors` list means the report passes (warnings are informational).
    """
    v = _run(report, spec, request, use_answer)
    return v.errors, v.warnings


def check(report: "Report", spec: SpecArg = None, *, request=None,
          use_answer: bool = False) -> List[str]:
    """Validate a Report against a track spec, COLLECTING all hard-ERROR violations.

    Non-raising counterpart to `verify`: returns the list of hard-failure messages (an
    empty list means valid). Smell categories (see `TrackSpec.smells`) are NOT returned
    here -- use `findings` to get both errors and warnings.

    `spec` may be a TrackSpec, a track-id string, or None (structural-only). `request`
    is only needed for per-request length limits (RAGTIME).
    """
    return _run(report, spec, request, use_answer).errors


def verify(report: "Report", spec: SpecArg = None, *, request=None,
           use_answer: bool = False) -> bool:
    """Validate a Report against a track spec, RAISING on the first hard ERROR.

    Fail-fast gate: returns True if there are no hard errors, otherwise raises
    TrackSpecVerificationError with the first one. Smell warnings never raise. For every
    problem at once without raising, use `check` (errors) or `findings` (errors+warnings).

    `spec` may be a TrackSpec, a track-id string, or None (structural-only). `request`
    is only needed for tracks whose length limit comes from the Request (RAGTIME).
    """
    _run(report, spec, request, use_answer).raise_first()
    return True
