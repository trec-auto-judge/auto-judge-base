import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, TypeVar, Generic, Callable, Iterable, Sequence, Union

from .verification import QrelsVerification, QrelsVerificationError

# Type for on_duplicate behavior in QrelsSpec
OnDuplicate = Literal["error", "keep_max", "keep_last"]


def doc_id_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


R = TypeVar("R")

@dataclass(frozen=True)
class QrelRow:
    topic_id: str
    doc_id: str
    grade: int

@dataclass(frozen=True)
class QrelsSpec(Generic[R]):
    topic_id: Callable[[R], str]
    doc_id: Callable[[R], str]
    grade: Callable[[R], int]
    on_duplicate: OnDuplicate = "error"

@dataclass(frozen=True)
class Qrels:
    """
    Collection of relevance judgments.

    Qrels are intentionally policy-free:
      - doc_id is opaque
      - no assumptions about corpus vs generated content
      - no assumptions about how grades are produced
    """
    rows: Sequence[QrelRow]
    
    def verify(self, expected_topic_ids: Optional[Sequence[str]], warn:Optional[bool]=False):
        QrelsVerification(self, expected_topic_ids=expected_topic_ids, warn=warn).all()
        
        
# === Qrel builder ===

def build_qrels(*, records: Iterable[R], spec: QrelsSpec[R]) -> list[QrelRow]:
    seen: dict[tuple[str, str], int] = {}
    for r in records:
        tid = spec.topic_id(r)
        did = spec.doc_id(r)
        g = int(spec.grade(r))

        key = (tid, did)
        if key in seen:
            if spec.on_duplicate == "error":
                raise ValueError(f"Duplicate qrel for {key}: old={seen[key]} new={g}")
            elif spec.on_duplicate == "keep_max":
                seen[key] = max(seen[key], g)
            elif spec.on_duplicate == "keep_last":
                seen[key] = g
            else:
                raise ValueError(f"Unknown on_duplicate: {spec.on_duplicate}")
        else:
            seen[key] = g

    return Qrels(rows=[QrelRow(topic_id=tid, doc_id=did, grade=g) for (tid, did), g in seen.items()])


# === serialization ===


def read_qrel_file(qrel_file: Union[str, Path]) -> Qrels:
    """
    Read qrels from standard TREC format:

        topic_id  iteration  doc_id  grade

    The iteration field (second column) is ignored.

    Args:
        qrel_file: Path to qrels file

    Returns:
        Qrels instance with loaded rows
    """
    path = Path(qrel_file)
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            topic_id, _ignored, doc_id, grade = parts[0], parts[1], parts[2], parts[3]
            rows.append(QrelRow(topic_id=topic_id, doc_id=doc_id, grade=int(grade)))

    return Qrels(rows=rows)


def write_qrel_file(
    *,
    qrel_out_file: Union[str, Path],
    qrels: Qrels,
    _ignored: str = "0",
) -> None:
    """
    Write qrels in standard TREC format:

        topic_id  iteration  doc_id  grade

    Notes:
    - The iteration field is always '0' (historical artifact, ignored by TREC tools).
    - Ordering is deterministic (sorted by topic_id, then doc_id).
    """
    path = Path(qrel_out_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Sort for reproducibility
    rows = sorted(
        qrels.rows,
        key=lambda r: (r.topic_id, r.doc_id),
    )

    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r.topic_id} {_ignored} {r.doc_id} {r.grade}\n")
