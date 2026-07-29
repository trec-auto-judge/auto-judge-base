"""Report sentence models, one per citation representation.

Kept in their own leaf module (no dependency on report.py) so the track-spec
verifier can reference them at runtime without importing the Report class -- which
lets report.py import the verifier at top level (see track_spec_verification.py),
mirroring the Leaderboard/Qrels verification layout.

The three representations are organizer-defined wire formats:
- NeuclirReportSentence: citations are a plain list of doc-id strings
- RagtimeReportSentence: citations are a {doc_id: confidence} mapping
- Rag24ReportSentence:   citations are zero-based integer indices into references
"""
from typing import Any, Dict, List, Optional, TypeAlias

from pydantic import BaseModel


class NeuclirReportSentence(BaseModel):
    citations: Optional[List[str]] = None
    text: str
    metadata: Optional[Dict[str, Any]] = None
    evaldata: Optional[Dict[str, Any]] = None


class RagtimeReportSentence(BaseModel):
    citations: Optional[Dict[str, float]] = None
    text: str
    metadata: Optional[Dict[str, Any]] = None
    evaldata: Optional[Dict[str, Any]] = None


class Rag24ReportSentence(BaseModel):
    citations: Optional[List[int]] = None
    text: str
    metadata: Optional[Dict[str, Any]] = None
    evaldata: Optional[Dict[str, Any]] = None


ReportSentence: TypeAlias = RagtimeReportSentence | NeuclirReportSentence | Rag24ReportSentence
