"""RAG-family report models and operations.

`Report` is a permissive binding that loads any TREC RAG-family submission format;
`load_report(path)` reads a submission JSONL. Validate or convert against a *track spec*
(a track-id string like "rag26", or a `TrackSpec`) -- the spec is always caller-supplied,
never inferred from the data:

    from autojudge_base.report import load_report, convert, submission_dict

    report = load_report("run.jsonl")[0]
    report.verify("rag26")          # -> True, or raises TrackSpecVerificationError (first problem)
    errors = report.check("rag26")  # -> [] if valid, else every violation (collect mode)
    obj = submission_dict(convert(report, "rag26"), "rag26")  # -> wire-format dict

Verification lives in `report_spec_verification`, spec data in `track_spec`, and the
sentence models in `report_sentences`. A CLI wraps this: `python -m
autojudge_base.report_tool check|convert`. See README.md.
"""
import argparse
from enum import Enum
import gzip
import re
from typing import Any, Dict, Iterable, List, Optional, Set, TextIO, Tuple, Type, Union, TypeAlias
from io import StringIO
from pathlib import Path
import json
from pydantic import BaseModel, ConfigDict, Field
from autojudge_base.document.document import Document
# Verification lives in its own module (report-free at runtime), imported top-level
# here -- mirroring the Leaderboard/Qrels verification layout.
from autojudge_base import report_spec_verification
from autojudge_base.track_spec import SPECS, sentence_class_for

class TaskType(str, Enum):
    """Ragtime tasks"""
    MULTILINGUAL = "multilingual"
    ENGLISH = "english"
    RAG = "rag"

class ReportMetaData(BaseModel):
    """Report meta data for requested reports"""
    team_id:str
    run_id:str
    topic_id:str = None
    collection_ids:Optional[List[str]] = None
    task:Optional[TaskType] = None
    description:Optional[str] = None
    creator:Dict[str,Any] = None
    extra: Dict[str,Any] = None

            
    # dragun
    use_starter_kit:Optional[int] = None
    type:Optional[str] = None
    
    # ragtime
    request_id:Optional[str]=None
    limit:Optional[int]=None
    
    # rag25
    # include narrative_id (topic id) and narrative (topic text)
    # if the track requires them;
    # collection_ids should be ["msmarco_v2.1_doc_segmented"].
    narrative_id:Optional[str|int] = None  # topic_id
    narrative:Optional[str] = None  # topic text

    # rag26: short description of the submitted system or run
    run_desc:Optional[str] = None


    # AutoJudge
    evaldata: Optional[Dict[str,Any]] = None

    def set_topic_ids(self):
        self.narrative_id = self.topic_id
        self.request_id = self.topic_id

    def set_narrative_text(self,narratives:Dict[str,Any]):
        self.narrative = narratives[self.narrative_id]
        
    def set_msmarco_collection_id(self):
        self.collection_ids = ["msmarco_v2.1_doc_segmented"]


    def model_post_init(self, __context__: dict | None = None) -> None:
        if self.topic_id is not None and self.narrative_id is not None and str(self.topic_id) != str(self.narrative_id):
            raise ValueError(
                f"Inconsistent topic identifiers: "
                f"topic_id={self.topic_id}, narrative_id={self.narrative_id}"
            )            

        if self.topic_id is None:
            # print("metadata topic_id is None, looking at other fields", self)
            # RAG input
            if self.narrative_id is not None and self.topic_id is None:
                if isinstance(self.narrative_id,int):
                    self.topic_id = f"{self.narrative_id}"
                else:
                    self.topic_id = self.narrative_id

        if self.topic_id is None:
            raise RuntimeError(f"ReportMetaData does not contain topic_id or narrative_id: {self}")
        self.set_topic_ids()

        # Expose as RAG format
        if self.narrative_id is None:
            self.narrative_id = self.topic_id

    model_config = ConfigDict(populate_by_name=True)


    
# Sentence models live in their own leaf module so the verifier can import them
# without importing Report; re-exported here for backwards compatibility.
from autojudge_base.report_sentences import (  # noqa: E402
    NeuclirReportSentence,
    Rag24ReportSentence,
    RagtimeReportSentence,
    ReportSentence,
)

def rank_confidences(doc_ids: List[str]) -> Dict[str, float]:
    """Synthesize ragtime confidences from a priority-ordered doc-id list.

    Only for formats that carry priority as *order* (neuclir, rag24) and have no
    real confidences to preserve. Any strictly decreasing function would do; 1/rank
    is used so the ordering survives the round trip. Note that these values occupy
    the bottom of the 0.0-100.0 range ragtime permits -- they encode rank, not a
    model score, and must not be compared against natively produced confidences.

    Duplicates keep their first (highest-priority) position.
    """
    unique = list(dict.fromkeys(doc_ids))
    return {d: 1.0 / rank for rank, d in enumerate(unique, start=1)}


RankedCitations: TypeAlias = List[Tuple[str, float]]
"""Citations as (doc_id, confidence) pairs, highest confidence first."""

EmittedCitations: TypeAlias = Union[Dict[str, float], List[str], List[int]]
"""A sentence's citations in one track's wire shape.

One member per sentence_type tag: {doc_id: confidence} for ragtime, [doc_id] for
neuclir, [position into references] for rag24 (see track_spec._TAG_TO_CLASS).
"""


def rank_citations(citations: Optional[Dict[str, float]]) -> Optional[RankedCitations]:
    """Order a ragtime citation mapping by descending confidence.

    The inverse of `rank_confidences`: that one turns order into numbers, this one
    turns numbers into order while keeping them. Every wire format wants the
    citations in priority order -- ragtime states it as the confidence, neuclir and
    rag24 as list position -- so ranking once and carrying the pairs lets a converter
    serve all three without re-deriving anything.

    Ties keep their submitted order. None (the sentence cited nothing) stays None,
    which is not the same as an empty citation set.
    """
    if citations is None:
        return None
    return sorted(citations.items(), key=lambda kv: kv[1], reverse=True)


class RankedDocument(BaseModel):
    doc_id:str
    rank:int
    doc:Optional[Document]=None
    score:Optional[float]=None

class RetrievedDocuments(BaseModel):
    query_id:str
    test_collection:str
    run_id:Optional[str]=None
    metadata:Optional[Dict[str,Any]] = None
    ranked_docs:List[RankedDocument] = list()


class Report(BaseModel):
    is_ragtime:bool = True
    metadata:ReportMetaData
    evaldata: Optional[Dict[str,Any]] = None
    responses:Optional[List[NeuclirReportSentence]|List[RagtimeReportSentence]|List[Rag24ReportSentence]]=None
    answer:Optional[List[NeuclirReportSentence]|List[RagtimeReportSentence]|List[Rag24ReportSentence]]=None
    path:Optional[Path]=Field(default=None, exclude=True)
    references:Optional[List[str]]=None  # index resolves to document id for `RAG25ReportSentence`
    ranking:Optional[List[RetrievedDocuments]|List[RankedDocument]] = None
    documents:Optional[Dict[str,Document]] = None
    
    
    def model_post_init(self, __context__: dict | None = None) -> None:
        if self.responses is None:
            # RAG
            self.responses = self.answer

        # RAGTIME validation
        if self.responses is None:
            raise RuntimeError(f"Report does not contain responses or answer: {self}")

        # Expose as RAG format
        if self.answer is None:
            self.answer = self.responses

        
    def get_report_text(self):
        return " ".join([sent.text for sent in self.responses])

    def get_text(self) -> str:
        return self.get_report_text()
    
    def get_paragraphs(self) -> List[str]:
        """Split report text into paragraphs on double-newlines."""
        import re
        text = self.get_text()
        normalized = re.sub(r'\n\s*\n+', '\n\n', text)
        return [p.strip() for p in normalized.split('\n\n') if p.strip()]
    
    def get_sentences(self) ->List[str]:
        return [s.text for s in self.responses]

    def get_sentences_with_citations(self) -> List[NeuclirReportSentence]:
        """Get all sentences with citations in unified format.

        Returns NeuclirReportSentence objects where citations is List[str]
        ordered by priority. Does not modify the underlying report data.

        Handles all sentence formats:
        - NeuclirReportSentence: returned as-is
        - RagtimeReportSentence: citations sorted by confidence (descending)
        - Rag24ReportSentence: indices resolved to doc_ids via report.references
        """
        references = self.references or []
        result: List[NeuclirReportSentence] = []

        for r in self.responses:
            if isinstance(r, NeuclirReportSentence):
                result.append(r)
            elif isinstance(r, RagtimeReportSentence):
                citation_confidences = r.citations.items() if r.citations else []
                sorted_ids = [eid for eid, conf in sorted(citation_confidences, key=lambda kv: kv[1], reverse=True)]
                result.append(NeuclirReportSentence(text=r.text, citations=sorted_ids, metadata=r.metadata, evaldata=r.evaldata))
            elif isinstance(r, Rag24ReportSentence):
                doc_ids = [references[i] for i in (r.citations or []) if 0 <= i < len(references)]
                result.append(NeuclirReportSentence(text=r.text, citations=doc_ids, metadata=r.metadata, evaldata=r.evaldata))

        return result
    
    def get_sentences_with_citation_confidences(self) -> List[RagtimeReportSentence]:
        """Get all sentences with citations+confidences in unified format.

        Returns RagtimeReportSentence objects where citations is Dict[str,float]
        where higher float means higher priority. Does not modify the underlying report data.

        Handles all sentence formats:
        - RagtimeReportSentence: returned as-is, real confidences preserved
        - NeuclirReportSentence: confidence by descending order
        - Rag24ReportSentence: indices resolved to doc_ids via report.references, confidence by descending order

        The order-carrying formats have no confidences to preserve, so theirs are
        synthesized from rank (see rank_confidences) -- unlike the ragtime case,
        where the submitted values pass through untouched.
        """
        references: List[str] = self.references or []
        result: List[RagtimeReportSentence] = []

        for r in self.responses:
            citations: Optional[Dict[str, float]]
            if isinstance(r, RagtimeReportSentence):
                result.append(r)
            elif isinstance(r, NeuclirReportSentence):
                citations = rank_confidences(r.citations) if r.citations is not None else None
                result.append(RagtimeReportSentence(text=r.text, citations=citations, metadata=r.metadata, evaldata=r.evaldata))
            elif isinstance(r, Rag24ReportSentence):
                doc_ids: List[str] = [references[i] for i in (r.citations or []) if 0 <= i < len(references)]
                citations = rank_confidences(doc_ids) if r.citations is not None else None
                result.append(RagtimeReportSentence(text=r.text, citations=citations, metadata=r.metadata, evaldata=r.evaldata))
            else:
                raise RuntimeError(f"Unknown sentence format {type(r).__name__}, cannot resolve citations: {r}")

        return result

    def autofill_references(self):
        ragtime_citation_set:Set[str] = {c for r in self.responses \
                            for c in r.citations.keys()   
                            if isinstance(r, RagtimeReportSentence) \
                        }
        neuclir_citation_set:Set[str] = {c for r in self.responses \
                            for c in r.citations   
                            if isinstance(r, NeuclirReportSentence) \
                        }
        self.references = list(ragtime_citation_set.union(neuclir_citation_set))

    def switch_responses_to_answer(self):
        self.answer=self.responses
        self.responses=None

    def switch_to_neuclir_responses(self):
        """Convert all sentence formats to NeuclirReportSentence.

        After calling this method, all sentences in report.responses will be
        NeuclirReportSentence with citations as List[str] ordered by priority.

        Handles:
        - RagtimeReportSentence: sorts citations by confidence (descending)
        - Rag24ReportSentence: resolves indices to doc_ids via report.references
        - NeuclirReportSentence: passes through unchanged
        """
        references = self.references or []

        def convert_sentence(r: ReportSentence) -> NeuclirReportSentence:
            if isinstance(r, NeuclirReportSentence):
                return r
            elif isinstance(r, RagtimeReportSentence):
                # Sort by confidence descending
                citation_confidences = r.citations.items() if r.citations else []
                sorted_ids = [eid for eid, conf in sorted(citation_confidences, key=lambda kv: kv[1], reverse=True)]
                return NeuclirReportSentence(text=r.text, citations=sorted_ids, metadata=r.metadata, evaldata=r.evaldata)
            elif isinstance(r, Rag24ReportSentence):
                # Resolve indices to doc_ids
                doc_ids = [references[i] for i in (r.citations or []) if 0 <= i < len(references)]
                return NeuclirReportSentence(text=r.text, citations=doc_ids, metadata=r.metadata, evaldata=r.evaldata)
            else:
                raise RuntimeError(f"Unknown sentence type: {type(r)}")

        def convert_response_sentences(responses: List[ReportSentence]) -> List[NeuclirReportSentence]:
            return [convert_sentence(r) for r in responses]

        if self.responses is not None:
            self.responses = convert_response_sentences(self.responses)
        elif self.answer is not None:
            self.answer = convert_response_sentences(self.answer)
        else:
            raise RuntimeError(f"Either responses or answer must be set to a non-None value, but received: {self}")



    def verify_ragtime(self, use_answer:bool = False, check_doc_ids:bool = True, spec=None):
        """DEPRECATED: legacy in-place RAGTIME validator, superseded by the track-spec
        framework. Prefer ``report.verify(spec=SPECS["ragtime25"])`` /
        ``verify(spec=SPECS["ragtime26"])`` (or pass ``spec=`` here, which delegates to
        it). Retained only for backwards compatibility.
        """
        if spec is not None:
            return report_spec_verification.verify(self, spec, use_answer=use_answer)

        import warnings
        warnings.warn(
            "Report.verify_ragtime() (no spec) is deprecated; use the track-spec "
            "framework: report.verify(spec=SPECS['ragtime26']).",
            DeprecationWarning, stacklevel=2,
        )

        def verify_citation_reference():
            citation_set:Set[str] = {c for r in self.responses \
                                        for c in r.citations.keys()   
                                        if isinstance(r, RagtimeReportSentence) \
                                    }
            reference_set:Set[str] = set(self.references or [])
            if not citation_set == reference_set:
                raise RuntimeError(f"Ragtime Report format invalid: citations and reference set must match. However, citation set is {citation_set} while reference set is {reference_set}.")

        def verify_citation_confidence_range():
            for r in self.responses:
                for c,v in r.citations.items():
                    if v<0.0 or v>100.0:
                        print(f"Warning: Ragtime Report format invalid confidence: citation confidences must be between 0.0 and 100.0, but found confidence {v} in sentence {r}. Limiting to valid range.")
                    if v<0.0:
                        r.citations[c]=0.0
                    if v>100.0:
                        r.citations[c]=100.0
                
        def verify_citation_given():
            for r in self.responses:
                if r.citations is None or len(r.citations)==0:
                    print(f"WARNING: Ragtime Report format contains empty citations: {r}")
                        
        def verify_citation_doc_id():
            if check_doc_ids:
                pattern = re.compile(r'^[A-Za-z0-9]+(?:-[A-Za-z0-9]+){4}_[A-Za-z0-9]+$')
                for r in self.responses:
                    for c,v in r.citations.items():
                        if not bool(pattern.match(c)):
                            print(f"WARNING: Ragtime Report format invalid docid? Citation contains document_id does not match format, maybe this document is from the wrong collection? document_id: {c}, but should look like this: \"47601789-65d8-4706-9bde-fc89fccfdf14_159897\"")
        
        
        def verify_task():
            if self.metadata.task is None or not (self.metadata.task == TaskType.ENGLISH or self.metadata.task == TaskType.MULTILINGUAL):
                raise RuntimeError(f"Ragtime Report requires `metadata.task` to be set to either {TaskType.MULTILINGUAL} or {TaskType.ENGLISH}, but is set to {self.metadata.task}")
        

        verify_task()
        verify_citation_reference()
        verify_citation_confidence_range()
        if self.is_ragtime:
            verify_citation_doc_id()
        verify_citation_given()
        
        return True
     
    def verify(self, spec=None, *, request=None, use_answer: bool = False) -> bool:
        """Validate this report against a track spec, RAISING on the first violation.

        Fail-fast gate: returns True if the report satisfies `spec`, otherwise raises
        RuntimeError describing the first problem. Use where an invalid report must not
        proceed -- an emit/submission gate or a pipeline assertion.

        To instead get EVERY problem back without raising, use `check` (they share the
        same rules; `verify` just raises the first thing `check` would return).

        Args:
            spec: the TrackSpec to validate against; None runs structural-only checks
                (citations resolve, text non-empty) with no track policy.
            request: only needed for tracks whose length limit lives on the Request
                (RAGTIME per-topic character budget); ignored otherwise.
            use_answer: validate `self.answer` instead of `self.responses`.

        `spec` may be a TrackSpec or a track-id string (e.g. "rag26").
        Delegates to autojudge_base.report_spec_verification.verify.
        """
        return report_spec_verification.verify(self, spec, request=request, use_answer=use_answer)

    def check(self, spec=None, *, request=None, use_answer: bool = False):
        """Validate this report against a track spec, COLLECTING all violations.

        Non-raising counterpart to `verify`: returns a list of human-readable violation
        messages (an empty list means valid) instead of stopping at the first problem.
        Use to triage/ingest data where you want to surface everything at once -- a
        submission validator, or a non-fatal pipeline warning.

        Same arguments and rules as `verify` (`spec` may be a TrackSpec or track-id
        string). Returns only hard-ERROR messages; smell categories (TrackSpec.smells)
        are omitted -- use `check_findings` for those. Delegates to
        autojudge_base.report_spec_verification.check.
        """
        return report_spec_verification.check(self, spec, request=request, use_answer=use_answer)

    def check_findings(self, spec=None, *, request=None, use_answer: bool = False):
        """Validate this report and return (errors, warnings).

        Like `check`, but also returns the SMELL warnings (findings whose category the
        spec lists in `smells`) as a second list. `errors` empty means the report passes.
        Delegates to autojudge_base.report_spec_verification.findings.
        """
        return report_spec_verification.findings(self, spec, request=request, use_answer=use_answer)

    def verify_rag(self, use_answer: bool = False, spec=None) -> bool:
        """Verify against a RAG spec (default: the latest RAG track, rag26).

        Convenience wrapper over `verify(spec)`.
        """
        return report_spec_verification.verify(self, spec or SPECS["rag26"], use_answer=use_answer)

    def convert(self, spec) -> "Report":
        """Convert this report into another track's compliant shape (explicit spec).

        Returns a NEW Report whose sentence representation, references and metadata all
        match `spec` (a TrackSpec or track-id string), so serializing it yields spec-valid
        output. See the module-level `convert`.
        """
        return convert(self, spec)

    def to_rag(self, spec=None) -> "Report":
        """DEPRECATED: use `convert(spec)` with an explicit RAG spec (e.g. "rag26")."""
        return to_rag(self, spec)

    def to_ragtime(self, spec=None) -> "Report":
        """DEPRECATED: use `convert(spec)` with an explicit RAGTIME spec (e.g. "ragtime26")."""
        return to_ragtime(self, spec)


# --- conversion (build a spec-compliant Report from any input format) -----------
# Lives here (not in track_spec) because it constructs Report/ReportMetaData; that
# keeps track_spec free of any runtime dependency on report.py.

def _map_metadata(src: "ReportMetaData", spec) -> "ReportMetaData":
    """Rebuild metadata in the target spec's shape.

    Carries team_id/run_id, puts the canonical topic id in the spec's topic_id_field,
    and fills the spec's remaining mandatory keys from the source where present
    (required-but-absent stay None -> verify flags them). Fields the target does not
    list are dropped (the metadata imposition).
    """
    fields = {
        "team_id": src.team_id,
        "run_id": src.run_id,
        spec.topic_id_field: src.topic_id,  # canonical id (topic_id/narrative_id synced)
    }
    for key in (*spec.mandatory_metadata, *spec.recommended_metadata):
        if key in fields:
            continue
        fields[key] = src.narrative if key == "narrative" else getattr(src, key, None)
    return ReportMetaData(**fields)


def convert(report: Report, spec) -> Report:
    """Build a NEW Report in the target spec's wire shape.

    `spec` may be a TrackSpec or a track-id string (e.g. "rag26"). Sentences use
    spec.emit_sentence_type; metadata is remapped to the spec (see _map_metadata).

    Source citations are read via `get_sentences_with_citation_confidences`, which
    normalizes any input format to the *ragtime* shape -- the most general of the
    three, being doc-id keyed like neuclir while also carrying priority as a number
    rather than as list position. Those are then ranked once into (doc_id, confidence)
    pairs (`rank_citations`) and carried in that shape through truncation and
    reference building, so each target format is served from the same ordered data:

        ragtime -> dict(ranked)                       confidences verbatim
        neuclir -> [doc_id ...]                       priority becomes list order
        rag24   -> [position in references ...]       priority becomes list order

    Lossy in one direction only: emitting rag24 or neuclir drops the confidences
    those formats cannot express (the ordering preserves the priority). A ragtime
    source converted to ragtime keeps its confidences exactly; the rank-derived
    stand-ins for order-only sources are minted once in the resolver
    (`rank_confidences`), never here. A sentence that cited nothing (citations is
    None) stays None rather than becoming an empty citation set.

    Conversion never enforces a rule the spec marks as a smell: citations are
    truncated to spec.max_citations_per_sentence only when "citation_count" is NOT
    in spec.smells (a smell means the data is known to violate the rule and must
    survive the conversion so the violation stays visible). Truncation keeps the
    top-ranked citations.

    References follow spec.references_kind: "cited_only" rebuilds the list from the
    citations alone, anything else keeps the source list (deduplicated) and appends
    any cited doc-id missing from it. A reference entry is a doc-id and nothing else
    -- no format gives it a confidence -- so citations index into it while their
    confidences stay on the sentence.
    """
    from autojudge_base.track_spec import get_spec
    if isinstance(spec, str):
        spec = get_spec(spec)
    tag: str = spec.emit_sentence_type
    cls: Type[ReportSentence] = sentence_class_for(tag)
    maxc: Optional[int] = spec.max_citations_per_sentence

    # One shape from here on: ragtime sentences, citations as Dict[str, float].
    resolved:List[RagtimeReportSentence] = report.get_sentences_with_citation_confidences()

    # Rank once, then carry (doc_id, confidence) pairs the rest of the way. Truncating
    # the ranked pairs is what makes "keeps the top-ranked citations" true.
    chop_cites: bool = (maxc is not None) and ("citation_count" not in spec.smells)
    per_sentence: List[Tuple[RagtimeReportSentence, Optional[RankedCitations]]] = []
    for s in resolved:
        ranked: Optional[RankedCitations] = rank_citations(s.citations)
        if ranked is not None and chop_cites:
            ranked = ranked[:maxc]
        per_sentence.append((s, ranked))

    # Build up the reference set, ensuring all cited documents are in it.
    references: List[str] = []
    if spec.references_kind != "cited_only":
        references = list(dict.fromkeys(report.references or []))  # deduplicate the list

    # Position of each doc-id in (what will become) references; rag24 cites by position.
    index: Dict[str, int] = {doc_id: pos for pos, doc_id in enumerate(references)}
    for _s, sentence_citations in per_sentence:
        for doc_id, _confidence in sentence_citations or []:
            if doc_id not in index:
                index[doc_id] = len(references)
                references.append(doc_id)  # on "cited_only" this builds the whole list,
                                           # in citation-priority order as encountered

    def emit_citations(ranked: Optional[RankedCitations]) -> Optional[EmittedCitations]:
        """Project the ranked pairs onto the target format's citation representation."""
        if ranked is None:
            return None
        if tag == "ragtime":
            ragtime_citations: Dict[str, float] = dict(ranked)
            return ragtime_citations
        if tag == "neuclir":
            neuclir_citations: List[str] = [doc_id for doc_id, _confidence in ranked]
            return neuclir_citations
        if tag == "rag24":
            rag24_citations: List[int] = [index[doc_id] for doc_id, _confidence in ranked]
            return rag24_citations
        raise KeyError(f"Cannot emit citations for unknown sentence_type tag {tag!r}")

    new_sentences: List[ReportSentence] = [
        cls(text=s.text, citations=emit_citations(ranked), metadata=s.metadata, evaldata=s.evaldata)
        for s, ranked in per_sentence
    ]

    return Report(
        is_ragtime=(tag == "ragtime"),
        metadata=_map_metadata(report.metadata, spec),
        responses=new_sentences,
        references=None if spec.references_kind == "none" else references,
    )


def to_rag(report: Report, spec=None) -> Report:
    """DEPRECATED: use `convert(report, spec)` with an explicit RAG spec.

    Defaulting a spec is the implicit spec-selection the framework avoids; call
    `convert(report, "rag26")` (or another RAG spec) instead.
    """
    import warnings
    warnings.warn(
        "to_rag is deprecated; use convert(report, spec) with an explicit spec, "
        "e.g. convert(report, 'rag26').",
        DeprecationWarning, stacklevel=2,
    )
    return convert(report, spec or SPECS["rag25"])


def to_ragtime(report: Report, spec=None) -> Report:
    """DEPRECATED: use `convert(report, spec)` with an explicit RAGTIME spec.

    Call `convert(report, "ragtime26")` (or another RAGTIME spec) instead.
    """
    import warnings
    warnings.warn(
        "to_ragtime is deprecated; use convert(report, spec) with an explicit spec, "
        "e.g. convert(report, 'ragtime26').",
        DeprecationWarning, stacklevel=2,
    )
    return convert(report, spec or SPECS["ragtime25"])


def submission_dict(report: Report, spec) -> Dict[str, Any]:
    """Serialize a Report to a track's wire-format JSON object per `spec`.

    Assumes the report already matches the spec's sentence representation -- call
    `convert(report, spec)` first if it might not. Metadata is reduced to the spec's
    mandatory keys (the topic-id field stringified); sentences go under
    spec.sentences_key; `references` is included unless references_kind == "none".
    `spec` may be a TrackSpec or a track-id string.
    """
    from autojudge_base.track_spec import get_spec
    if isinstance(spec, str):
        spec = get_spec(spec)
    md = report.metadata
    metadata = {
        key: (str(getattr(md, key)) if key == spec.topic_id_field else getattr(md, key))
        for key in (*spec.mandatory_metadata, *spec.recommended_metadata)
        if getattr(md, key, None) is not None or key in spec.mandatory_metadata
    }
    obj: Dict[str, Any] = {"metadata": metadata}
    if spec.references_kind != "none":
        obj["references"] = report.references
    obj[spec.sentences_key] = [
        {"text": s.text, "citations": s.citations} for s in (report.responses or [])
    ]
    return obj


def extract_doc_ids_from_report(report: Report, cited_only: bool = False) -> Set[str]:
    """Extract all unique document IDs from a Report's citations.

    Args:
        report: The report to extract doc IDs from.
        cited_only: If True, only include documents actually cited in sentences.
            If False, also include documents in report.references that aren't cited.

    Returns:
        Set of document IDs as strings.
    """
    doc_ids: Set[str] = set()

    for sentence in report.responses or []:
        if sentence.citations is None:
            continue

        if isinstance(sentence, RagtimeReportSentence):
            # Dict[str, float] - keys are doc IDs
            doc_ids.update(sentence.citations.keys())

        elif isinstance(sentence, NeuclirReportSentence):
            # List[str] - direct doc IDs (may be parsed as int from some corpora, hence str())
            doc_ids.update(str(c) for c in sentence.citations)

        elif isinstance(sentence, Rag24ReportSentence):
            # List[int] - indices into report.references
            if report.references:
                for idx in sentence.citations:
                    if 0 <= idx < len(report.references):
                        doc_ids.add(report.references[idx])

    # Add any references not already in doc_ids
    if not cited_only and report.references:
        for ref in report.references:
            ref_str = str(ref)
            if ref_str not in doc_ids:
                doc_ids.add(ref_str)

    return doc_ids


def remove_unavailable_citations(report: Report, unavailable_doc_ids: Set[str]) -> int:
    """Remove citations for unavailable doc_ids from report in place.

    Handles all sentence types:
    - Rag24ReportSentence: Removes indices pointing to unavailable references,
      then remaps remaining indices after references are pruned.
    - NeuclirReportSentence: Removes doc_ids from citations list.
    - RagtimeReportSentence: Removes doc_ids from citations dict.

    Also removes unavailable doc_ids from report.references.

    Args:
        report: The report to modify in place.
        unavailable_doc_ids: Set of doc_ids to remove.

    Returns:
        Count of citations removed.
    """
    if not unavailable_doc_ids:
        return 0

    removed_count = 0

    # For Rag24: build index remapping before modifying references
    # old_index -> new_index for references that will be kept
    old_to_new_index: Dict[int, int] = {}
    if report.references:
        new_idx = 0
        for old_idx, ref in enumerate(report.references):
            if ref not in unavailable_doc_ids:
                old_to_new_index[old_idx] = new_idx
                new_idx += 1

    # Clean sentences
    for sentence in report.responses or []:
        if sentence.citations is None:
            continue

        if isinstance(sentence, RagtimeReportSentence):
            # Dict[str, float] - filter by key
            original_len = len(sentence.citations)
            sentence.citations = {
                doc_id: conf
                for doc_id, conf in sentence.citations.items()
                if doc_id not in unavailable_doc_ids
            }
            removed_count += original_len - len(sentence.citations)

        elif isinstance(sentence, NeuclirReportSentence):
            # List[str] - filter by value
            original_len = len(sentence.citations)
            sentence.citations = [
                doc_id for doc_id in sentence.citations
                if str(doc_id) not in unavailable_doc_ids
            ]
            removed_count += original_len - len(sentence.citations)

        elif isinstance(sentence, Rag24ReportSentence):
            # List[int] - remap indices to new positions
            # Indices not in old_to_new_index are filtered out (not set to None)
            original_len = len(sentence.citations)
            sentence.citations = [
                old_to_new_index[idx]
                for idx in sentence.citations
                if idx in old_to_new_index
            ]
            removed_count += original_len - len(sentence.citations)

    # Clean references
    if report.references:
        report.references = [
            ref for ref in report.references
            if ref not in unavailable_doc_ids
        ]
        if not report.references:
            report.references = None

    # Also clean report.answer if it differs from responses
    if report.answer is not None and report.answer is not report.responses:
        for sentence in report.answer:
            if sentence.citations is None:
                continue

            if isinstance(sentence, RagtimeReportSentence):
                original_len = len(sentence.citations)
                sentence.citations = {
                    doc_id: conf
                    for doc_id, conf in sentence.citations.items()
                    if doc_id not in unavailable_doc_ids
                }
                removed_count += original_len - len(sentence.citations)

            elif isinstance(sentence, NeuclirReportSentence):
                original_len = len(sentence.citations)
                sentence.citations = [
                    doc_id for doc_id in sentence.citations
                    if str(doc_id) not in unavailable_doc_ids
                ]
                removed_count += original_len - len(sentence.citations)

            elif isinstance(sentence, Rag24ReportSentence):
                original_len = len(sentence.citations)
                sentence.citations = [
                    old_to_new_index[idx]
                    for idx in sentence.citations
                    if idx in old_to_new_index
                ]
                removed_count += original_len - len(sentence.citations)

    return removed_count


def load_report(reports_path:Path)->List[Report]:
    reports = list()
    with open(file=reports_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue    # tolerate blank/trailing lines in submission files
            data = json.load(fp=StringIO(line))
            report = Report.model_validate(data)
            report.path = reports_path.absolute()
            reports.append(report)
    return reports



def write_pydantic_json_list(objs: List[BaseModel], out: Union[str, Path, TextIO]) -> None:

    """
    Serialize a Pydantic model to JSON and write it to a file or stream, omitting fields with `None` values.

    This function supports writing to:
      - A plain text file path (as `str` or `Path`)
      - A gzip-compressed file path ending in `.gz`
      - An open `TextIO` stream (e.g., `open(..., 'w')`, `gzip.open(..., 'wt')`, or `io.StringIO`)

    Args:
        model (BaseModel): The Pydantic model to serialize.
        out (Union[str, Path, TextIO]): The output destination. If a string or Path is provided,
            it will be opened in write-text mode (`"wt"`). If the path ends in `.gz`, it will be
            automatically gzip-compressed.

    Example:
        >>> write_model_json(my_model, "out.json")
        >>> write_model_json(my_model, "out.json.gz")
        >>> with open("out.json", "w") as f:
        >>>     write_model_json(my_model, f)
        >>> with gzip.open("out.json.gz", "wt") as f:
        >>>     write_model_json(my_model, f)
    """
    if isinstance(out, (str, Path)):
        open_fn = gzip.open if str(out).endswith(".gz") else open
        with open_fn(out, mode="wt", encoding="utf-8") as f:
            for obj in objs:
                line = json.dumps(obj.model_dump(mode="json", exclude_none=True), separators=(",", ":"), indent=None)
                f.write(line + "\n")
    else:
        # Assume it's already a valid TextIO stream
        for obj in objs:
            line = json.dumps(obj.model_dump(mode="json", exclude_none=True), separators=(",", ":"), indent=None)
            out.write(line + "\n")



class JsonlWriter:
    """
    Stream JSON-Lines to a file or TextIO.  Usage:

        with JsonlWriter("out.jsonl") as w:
            w.write(obj1)
            w.write(obj2)

        # or keep it open elsewhere
        writer = JsonlWriter("out.jsonl.gz")
        for obj in objects:
            writer.write(obj)
        writer.close()
    """

    def __init__(self,
                 out: Union[str, Path, TextIO],
                 *,
                 auto_flush: bool = True) -> None:
        self._auto_flush = auto_flush
        self._owns_stream = isinstance(out, (str, Path))

        if self._owns_stream:
            out_path = Path(out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            open_fn = gzip.open if str(out).endswith(".gz") else open
            # "at": append-text so you can resume writing if the file exists
            self._f: TextIO = open_fn(out, mode="wt", encoding="utf-8", buffering=1)
        else:
            # Already a TextIO (caller must close it)
            self._f = out  # type: ignore

    # --------------------------------------------------------
    # context-manager sugar so you can `with JsonlWriter(...)`
    # --------------------------------------------------------
    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # --------------------------------------------------------
    # public API
    # --------------------------------------------------------
    def write(self, obj: BaseModel) -> None:
        """Write ONE Pydantic object as a JSONL line and flush."""
        line: str = json.dumps(
            obj.model_dump(mode="json", exclude_none=True),
            separators=(",", ":")
        )
        self._f.write(line + "\n")
        if self._auto_flush:
            self._f.flush()

    def write_many(self, objs: Iterable[BaseModel]) -> None:
        """Convenience helper for a batch."""
        for o in objs:
            self.write(o)

    def close(self) -> None:
        if self._owns_stream and not self._f.closed:
            self._f.close()




def make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]
    elif isinstance(obj, set):
        return sorted(make_json_serializable(x) for x in obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)  # fallback
