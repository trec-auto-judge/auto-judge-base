import argparse
from enum import Enum
import gzip
import re
from typing import Any, Dict, Iterable, List, Optional, Set, TextIO, Union, TypeAlias
from io import StringIO
from pathlib import Path
import json
from pydantic import BaseModel, ConfigDict, Field
from autojudge_base.document.document import Document

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


    
class NeuclirReportSentence(BaseModel):
    citations: Optional[List[str]] = None    
    text:str
    metadata: Optional[Dict[str,Any]] = None
    evaldata: Optional[Dict[str,Any]] = None


class RagtimeReportSentence(BaseModel):
    citations: Optional[Dict[str,float]] = None
    text:str
    metadata: Optional[Dict[str,Any]] = None
    evaldata: Optional[Dict[str,Any]] = None

class Rag24ReportSentence(BaseModel):
    citations: Optional[List[int]] = None    
    text:str
    metadata: Optional[Dict[str,Any]] = None
    evaldata: Optional[Dict[str,Any]] = None


ReportSentence: TypeAlias = RagtimeReportSentence | NeuclirReportSentence | Rag24ReportSentence

class Report(BaseModel):
    is_ragtime:bool = True
    metadata:ReportMetaData
    evaldata: Optional[Dict[str,Any]] = None
    responses:Optional[List[NeuclirReportSentence]|List[RagtimeReportSentence]|List[Rag24ReportSentence]]=None
    answer:Optional[List[NeuclirReportSentence]|List[RagtimeReportSentence]|List[Rag24ReportSentence]]=None
    path:Optional[Path]=Field(default=None, exclude=True)
    references:Optional[List[str]]=None  # index resolves to document id for `RAG25ReportSentence`
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
    
    def get_paragraphs(self) ->List[str]:
        from nuggety.text_chunker import  get_paragraph_chunks  # Todo this next
        return get_paragraph_chunks(self.get_text())
    
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



    def verify_ragtime(self, use_answer:bool = False):
        
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
    with open(file=reports_path) as f:
        for line in f.readlines():
            data = json.load(fp=StringIO(line))
            report = Report.validate(data)
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
