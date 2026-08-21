"""RAGTIME 2026 nugget bank format.

One JSON object per topic, per line:

```json
{"metadata":    {"team_id": "example-team", "topic_id": "topic-1",
                 "run_id": "example-run", "run_desc": "example run description"},
 "nugget_bank": [{"question": "First example question?",
                  "aggregator_type": "AND",
                  "answers": [{"answer": "example answer",
                               "references": ["doc-1"]}]}]}
```

Per the organizers' spec:

- ``metadata`` carries ``team_id``, ``topic_id``, ``run_id`` (at most 25 characters)
  and ``run_desc``. Other metadata fields may be present; the scorer ignores them, so
  they are preserved here (``extra="allow"``) rather than dropped.
- ``aggregator_type`` is ``AND`` (every answer must appear) or ``OR`` (any one does).
- ``references`` are document ids, typed ``str | Reference`` so a submission can grow
  passage offsets later without a format change. Nothing is normalized, so a nugget
  bank round-trips in the shape it arrived in.
- Nuggets and answers are unordered ("treated as a set"), but the list is kept as a
  list so a round-trip preserves the submitted order. Only ``to_nugget_bank()``
  collapses them into the ragtime25 keyed dict.

Questions, answers and references each carry an optional ``metadata`` dict, as the
ragtime25 models do; questions additionally carry an optional ``importance``.

```python
from autojudge_base.nugget_data import load_ragtime26_nugget_banks_from_file

nugget_banks = load_ragtime26_nugget_banks_from_file("nugget_banks.jsonl")
nugget_bank = nugget_banks.banks["topic-1"]
ragtime25 = nugget_bank.to_nugget_bank()
```
"""

from typing import Any, Dict, List, Literal, Optional, Sequence, Type

from pydantic import BaseModel, ConfigDict

from ..nugget_data import Answer, NuggetBank, NuggetQuestion, Reference

RAGTIME26_FORMAT_VERSION = "RAGTIME26"

# The spec allows exactly these two; the shared AggregatorType also has NOT/SUM/Default,
# which a ragtime26 nugget bank must not use.
Ragtime26AggregatorType = Literal["AND", "OR"]

# A document id, or a Reference once passage offsets are wanted.
Ragtime26Reference = str | Reference


class Ragtime26NuggetMetadata(BaseModel):
    """Run metadata identifying the submission a nugget bank came from.

    ``extra="allow"`` because the spec says other fields may be present: the scorer
    ignores them, but they are a participant's data and must survive a round-trip.
    """

    model_config = ConfigDict(extra="allow")

    topic_id: str
    team_id: Optional[str] = None
    # The spec caps run_id at 25 characters. Not enforced here on purpose: an
    # over-long run_id should be reported at submission time, not make an already
    # submitted nugget bank impossible to load (and so impossible to anonymize).
    run_id: Optional[str] = None
    run_desc: Optional[str] = None


class Ragtime26Answer(BaseModel):
    """An answer span with the documents that support it."""

    answer: str
    references: Optional[List[Ragtime26Reference]] = None

    metadata: Optional[Dict[str, Any]] = None

    def reference_doc_ids(self) -> List[str]:
        """Document ids of this answer's references, whichever shape they are in."""
        return [r if isinstance(r, str) else r.doc_id for r in (self.references or [])]


class Ragtime26Nugget(BaseModel):
    """One question a good answer should address, with its acceptable answers."""

    question: str
    aggregator_type: Optional[Ragtime26AggregatorType] = None
    answers: Optional[List[Ragtime26Answer]] = None

    importance: Optional[int | str] = None
    metadata: Optional[Dict[str, Any]] = None


class Ragtime26NuggetBank(BaseModel):
    """One topic's nugget bank, in the ragtime26 submission format."""

    format_version: str = RAGTIME26_FORMAT_VERSION
    metadata: Ragtime26NuggetMetadata
    nugget_bank: Optional[List[Ragtime26Nugget]] = None

    @property
    def query_id(self) -> str:
        """Canonical identifier for the topic (alias for metadata.topic_id)."""
        return self.metadata.topic_id

    def nuggets_as_list(self) -> List[Ragtime26Nugget]:
        """All nuggets in this nugget bank (matches NuggetBank.nuggets_as_list)."""
        return list(self.nugget_bank or [])

    def to_nugget_bank(self) -> NuggetBank:
        """Convert into the ragtime25 NuggetBank (v3).

        Nuggets become NuggetQuestions keyed by question text and answers become the
        keyed Answer dict -- lossless under the spec's "treated as a set", though the
        submitted order is not preserved on the ragtime25 side.
        """
        questions: List[NuggetQuestion] = []
        for nugget in self.nuggets_as_list():
            answers: List[Answer] = []
            for src in nugget.answers or []:
                answer = Answer(answer=src.answer, metadata=src.metadata)
                answer.add_references(src.references)
                answers.append(answer)

            question = NuggetQuestion(
                question=nugget.question,
                query_id=self.query_id,
                aggregator_type=nugget.aggregator_type,
                importance=nugget.importance,
                metadata=nugget.metadata,
            )
            if answers:
                question.add_answers(answers)
            questions.append(question)

        nugget_bank = NuggetBank(
            query_id=self.query_id,
            metadata=self.metadata.model_dump(exclude_none=True),
        )
        return nugget_bank.add_nuggets(questions)


class Ragtime26NuggetBanks(BaseModel):
    """Container for multiple ragtime26 nugget banks, keyed by topic id."""

    format_version: str = RAGTIME26_FORMAT_VERSION
    banks: Dict[str, Ragtime26NuggetBank] = {}

    @classmethod
    def get_bank_model(cls) -> Type[Ragtime26NuggetBank]:
        """Return the nugget bank model class for this container type."""
        return Ragtime26NuggetBank

    @classmethod
    def from_banks_list(
        cls, banks: List[Ragtime26NuggetBank], overwrite: bool = False
    ) -> "Ragtime26NuggetBanks":
        """Create from a list of nugget banks, keyed by topic id.

        Args:
            banks: List of Ragtime26NuggetBank instances
            overwrite: If False (default), raise on a duplicate topic id
        """
        result: Dict[str, Ragtime26NuggetBank] = {}
        for nugget_bank in banks:
            qid = nugget_bank.query_id
            if qid is None:
                raise ValueError("Ragtime26NuggetBank must have a metadata.topic_id")
            if qid in result and not overwrite:
                raise ValueError(f"Duplicate topic_id: {qid}")
            result[qid] = nugget_bank
        return cls(banks=result)

    def verify(self, expected_topic_ids: Sequence[str], warn: bool = False) -> None:
        """Verify nugget banks against expected topic IDs."""
        from ..verification import NuggetBanksVerification  # Local import avoids cycle

        NuggetBanksVerification(self, expected_topic_ids=expected_topic_ids, warn=warn).all()


# Curried I/O functions for Ragtime26NuggetBanks
from ..io import make_io_functions  # noqa: E402  (matches the nuggetizer layout)

load_ragtime26_nugget_banks_from_file, \
    load_ragtime26_nugget_banks_from_directory, \
    write_ragtime26_nugget_banks = \
    make_io_functions(Ragtime26NuggetBank, Ragtime26NuggetBanks)
