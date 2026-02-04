import gzip
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, TextIO, Type, TypeVar, Union
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime, timezone


"""
# NuggetBank Library

A structured, Pydantic-based representation of query-focused nugget questions and answers.

## Features

- Recursive question hierarchy via `NuggetQuestion` and `sub_nuggets`
- Optional metadata and provenance tracking (`Reference`)
- Simple construction from strings via `from_lazy` constructors
- Flat indexing support with `NuggetBank.index_nuggets()`

## Usage

```python
from autojudge_base.nugget_data import NuggetBank, NuggetQuestion, NuggetClaim, Answer

# Create simple nuggets with gold answers
nugget = NuggetQuestion.from_lazy(
    query_id="1",
    question="Where is Machu Picchu located?",
    gold_answers=["in Peru", "in South America"]
)

# Add a claim-style nugget
claim = NuggetClaim.from_lazy(
    query_id="1",
    claim="Machu Picchu was built in the 15th century"
)

# Create a NuggetBank and populate it
bank = NuggetBank(query_id="1", title_query="Machu Picchu facts")
bank.add_nuggets([nugget, claim])
bank.index_nuggets()

# Print flat index of all nuggets
print(bank.all_nuggets_view)
```
"""


def merge_metadata(
    left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Merge two metadata dictionaries. Keys from `right` override `left`. Adds a 'merged' key to retain both inputs."""

    if left is None and right is None:
        return None

    left = left or {}
    right = right or {}

    merged = {**left, **right}
    merged["merged"] = [left, right]

    return merged


def none_as_empty(x: Optional[Dict[Any, Any]]) -> Dict[Any, Any]:
    if x is None:
        return {}
    else:
        return x


def none_as(x: Optional[Any], default_val):
    if x is None:
        return default_val
    else:
        return x


def merge_metadata_list(
    items: Iterable[Optional[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Merge a sequence of metadata dictionaries. Ignores `None` values."""
    result: Optional[Dict[str, Any]] = None
    for item in items:
        if item is not None:
            result = merge_metadata(result, item)
    return result


def opt_list_as_empty(x: Optional[List[Any]]) -> List[Any]:
    """Convert None to an empty list; return the list unchanged otherwise."""
    if x is None:
        return []
    else:
        return x


def opt_int_as_zero(x: Optional[int]) -> int:
    """Convert None to 0; return the integer unchanged otherwise."""

    if x is None:
        return 0
    else:
        return x


T = TypeVar("T", bound=int | str)

agg_take_right: Callable[[Union[int, str], Union[int, str]], Union[int, str]] = (
    lambda x, y: y
)


def aggregate_unless_none(
    x: Optional[T],
    y: Optional[T],
    default_val: Optional[T],
    aggregate_fun: Callable[[T, T], T],
) -> Optional[T]:
    """Aggregate two optional values using a custom function, falling back to defaults if needed."""

    if x is None:
        return y if y is not None else default_val
    elif y is None:
        return x
    else:
        return aggregate_fun(x, y)


def aggregate_unless_none_list(
    anss: List[Optional[T]],
    aggregate_fun: Callable[[T, T], T],
    default_val: Optional[T] = None,
) -> Optional[T]:
    """Aggregate a list of optional values using a custom function, skipping None entries."""

    result: Optional[T] = None
    for ans in anss:
        result = (
            ans
            if result is None
            else aggregate_fun(result, ans) if ans is not None else result
        )
    return result if result is not None else default_val


I = TypeVar("I")  # input type (e.g., str or Reference)
O = TypeVar("O")  # output type (e.g., Reference)
InputList = Optional[Union[I, str, List[Union[I, str]]]]


def normalize_list_input(
    x: Optional[Union[I, str, List[I], List[str]]],  # ← accepts all four shapes
    coerce_fun: Callable[[str], I],  # how to turn a str into I
) -> Optional[List[I]]:
    """Normalize a single item or list of items into a list, coercing any strings via a given constructor."""

    if x is None:
        return None
    if isinstance(x, list):
        return [coerce_fun(v) if isinstance(v, str) else v for v in x]
    return [coerce_fun(x) if isinstance(x, str) else x]


def coerce_type(x: Union[str, I], constr: Callable[[str], I]) -> I:
    """Apply constructor to a string, return item unchanged otherwise."""
    return constr(x) if isinstance(x, str) else x


def never(_: str) -> I:  # raises if a stray str appears
    """Raise an error when a string is unexpectedly encountered."""

    raise ValueError("Cannot coerce str")


class AggregatorType(str, Enum):
    """Enumeration of logical operators for combining answers or sub-questions."""

    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    SUM = "SUM"
    # AT_LEAST_THREE = "AT LEAST THREE"
    # AT_LEAST_HALF = "AT LEAST HALF"
    Default = "Default"


class Offsets(BaseModel):
    """Start and end character offsets with optional encoding metadata."""

    start_offset: int
    end_offset: int
    encoding: str
    metadata: Optional[Dict[str, Any]] = None


C = TypeVar("C", bound="Creator")


class Creator(BaseModel):
    """Describes the origin of an answer, whether human or generated by an LLM."""

    is_human: bool
    assessor_id: Optional[str] = None
    llm_model: Optional[str] = None
    llm_backend: Optional[str] = None
    llm_prompt: Optional[List[str]] = None
    llm_prompt_strategy: Optional[str] = None
    source_data: Optional[List["Reference"]] = None
    creation_date: Optional[str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    contact: Optional[List[str]] = None
    format: Optional[str] = None

    model_config = {"recursive": True, "extra": "ignore"}  # Required in pydantic v2 to support recursion

    def set_creation_date(self, dt: datetime) -> None:
        """Set creation_date from a datetime object as an ISO string."""

        self.creation_date = dt.isoformat()

    def get_creation_datetime(self) -> Optional[datetime]:
        """Convert creation_date string back into a datetime object, if set."""

        if self.creation_date is None:
            return None
        return datetime.fromisoformat(self.creation_date)

    @classmethod
    def merge_creators(
        cls: Type[C],
        left: Optional[Union[C, List[C]]],
        right: Optional[Union[C, List[C]]],
    ) -> Optional[List[C]]:
        """Merge two Creator instances or lists into a flat list."""

        def to_list(val: Optional[Union[C, List[C]]]) -> List[C]:
            if val is None:
                return []
            if isinstance(val, list):
                return val
            return [val]

        merged = to_list(left) + to_list(right)
        return merged or None

    @classmethod
    def merge_creator_list(
        cls: Type[C], items: Iterable[Optional[Union[C, List[C]]]]
    ) -> Optional[List[C]]:
        """Merge multiple Creator instances or lists into a flat list."""

        result: List[C] = []
        for item in items:
            result = cls.merge_creators(result, item) or []
        return result or None


def normalize_creators(
    c: Optional[Union["Creator", List["Creator"]]],
) -> Optional[List["Creator"]]:
    """Normalize one or more Creator instances into a flat list. Raises if input includes a string."""

    return normalize_list_input(c, coerce_fun=never)


class Reference(BaseModel):
    """Represents document-level provenance for an answer, including optional metadata."""

    doc_id: str
    collection: Optional[str] = None
    text: Optional[str] = None
    offsets: Optional[Offsets] = None

    quality: Optional[int | str] = None
    importance: Optional[int | str] = None
    creator: Optional[List[Creator]] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = {"recursive": True, "extra": "ignore"}  # Required in pydantic v2 to support recursion

    def add_creator(self, creator: Optional[Union[List[Creator], Creator]]):
        """Normalize and attach creator(s) to the nugget."""
        self.creator = normalize_creators(creator)
        return self


def normalize_references(
    r: Optional[Union[str, "Reference", List[str], List["Reference"]]],
) -> Optional[List["Reference"]]:
    """Normalize Reference instances or document ID strings into a list of Reference objects."""

    return normalize_list_input(r, coerce_fun=lambda doc: Reference(doc_id=doc))


class Answer(BaseModel):
    """A basic answer with optional provenance, quality, and metadata.\

    ```
    # Minimal answer
    a1 = Answer(answer="Mount Everest is the highest mountain.")

    # Extended answer: add references and creator
    # It is recommended to add references and creator via these functions
    a2 = Answer(answer="Mount Everest is located in the Himalayas.")
    a2.add_references(["doc123"])
    a2.add_creator(Creator(is_human=True, assessor_id="annotator-7", contact=["<your name>"]))
    ```
    """

    answer: str
    references: Optional[List[Reference]] = None

    quality: Optional[int | str] = None
    importance: Optional[int | str] = None
    creator: Optional[List[Creator]] = None
    metadata: Optional[Dict[str, Any]] = None

    def add_references(
        self,
        references: Optional[Union[str, Reference, List[str], List[Reference]]],
    ):
        """Normalize and attach reference(s) to the answer."""

        self.references = normalize_references(references)

    def add_creator(self, creator: Optional[Union[List[Creator], Creator]]):
        """Normalize and attach creator(s) to the answer."""

        self.creator = normalize_creators(creator)
        return self

    @classmethod
    def from_lazy(
        cls,
        answer: str,
        references: Optional[Union[str, Reference, List[str], List[Reference]]] = None,
        creator: Optional[Union[List[Creator], Creator]] = None,
        **kwargs,
    ):
        """Construct an Answer from optional references and creators, accepting raw input."""
        return cls(
            answer=answer,
            references=normalize_references(references),
            creator=normalize_creators(creator),
            **kwargs,
        )

    model_config = {"recursive": True, "extra": "ignore"}  # Required in pydantic v2 to support recursion

    @classmethod
    def merge_answers(
        cls,
        ans1: "Answer",
        ans2: "Answer",
        aggregate_quality: Callable[
            [Optional[int | str], Optional[int | str]], Optional[int | str]
        ] = lambda x, y: x,
        aggregate_importance: Callable[
            [Optional[int | str], Optional[int | str]], Optional[int | str]
        ] = lambda x, y: x,
    ) -> "Answer":
        """Merge two answers, combining references, metadata, and choosing quality/importance."""

        return cls(
            answer=ans1.answer,
            references=opt_list_as_empty(ans1.references)
            + opt_list_as_empty(ans2.references),
            quality=aggregate_unless_none(
                ans1.quality,
                ans2.quality,
                default_val=None,
                aggregate_fun=aggregate_quality,
            ),
            importance=aggregate_unless_none(
                ans1.importance,
                ans2.importance,
                default_val=None,
                aggregate_fun=aggregate_importance,
            ),
            creator=Creator.merge_creators(ans1.creator, ans2.creator),
            metadata=merge_metadata(ans1.metadata, ans2.metadata),
        )

    @classmethod
    def merge_answer_dicts(
        cls, answers: List[Dict[str, "Answer"]]
    ) -> Dict[str, "Answer"]:
        """Merge a list of answer dictionaries, resolving duplicate keys via merge_answers."""
        merged: Dict[str, "Answer"] = dict()
        for answer_dict in answers:
            for key, answer in answer_dict.items():
                if key in merged:
                    # we need to merge answers
                    old_answer = merged[key]
                    merged[key] = cls.merge_answers(old_answer, answer)
                else:
                    merged[key] = answer
        return merged


def filter_answers_with_no_references(answers: Dict[str, Answer]) -> Dict[str, Answer]:
    """Filter out answers that have no references attached."""
    return {k: v for k, v in answers.items() if v.references and len(v.references) > 0}


def normalize_answers(gold_answers: str | Answer | List[str | Answer]) -> List[Answer]:
    """Normalize raw strings or Answer objects into a list of Answer instances."""

    if isinstance(gold_answers, Answer):
        return [gold_answers]
    elif isinstance(gold_answers, str):
        return [Answer(answer=gold_answers)]
    elif isinstance(gold_answers, list):
        return [
            Answer(answer=elem) if isinstance(elem, str) else elem
            for elem in gold_answers
        ]
    else:
        raise TypeError(f"Invalid type for answer input: {type(gold_answers)}")


def hash_answers(answers: List[Answer]) -> dict[str, Answer]:
    """Convert a list of Answer objects into a dict keyed by answer text, merging duplicates."""

    result: dict[str, Answer] = {}
    for a in answers:
        key = a.answer
        if key in result:
            result[key] = Answer.merge_answers(result[key], a)
        else:
            result[key] = a
    return result


class NuggetQuestion(BaseModel):
    """Represents a question, its answers, and nested sub-nuggets with optional metadata.
    ```

    # Basic nugget question with query ID and question text
    nq = NuggetQuestion.from_lazy(
        query_id="q42",
        question="What is the tallest mountain on Earth?"
    )

    # Add answers after construction
    nq.add_answers([
        "Mount Everest",
        "Everest"
    ])

    # Add a human creator
    nq.add_creator(Creator(is_human=True, assessor_id="expert-3", contact=["<your name>"]))
    ```
    """

    question: str
    question_markup: Optional[Any] = None
    answers: Optional[Dict[str, Answer]] = None
    sub_nuggets: Optional[List[Union["NuggetQuestion", "NuggetClaim"]]] = None
    references: Optional[List[Reference]] = None

    question_id: Optional[str] = None
    aggregator_type: Optional[AggregatorType] = None

    query_id: str
    test_collection: Optional[str] = None

    quality: Optional[int | str] = None
    importance: Optional[int | str] = None
    creator: Optional[List[Creator]] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = {"recursive": True, "extra": "ignore"}  # Required in pydantic v2 to support recursion

    def model_post_init(self, __context):
        """Assign a hash-based ID if none is given, based on the question text."""

        if self.question_id is None:
            self.question_id = hashlib.md5(self.question.encode()).hexdigest()

    def add_answers(self, gold_answers: List[str | Answer] | str | Answer):
        """Normalize and store gold answer strings or objects."""
        self.answers = hash_answers(normalize_answers(gold_answers))
        return self

    def add_creator(self, creator: Optional[Union[List[Creator], Creator]]):
        """Normalize and attach creator(s) to the nugget."""
        self.creator = normalize_creators(creator)
        return self

    def add_references(
        self,
        references: Optional[Union[str, Reference, List[str], List[Reference]]],
    ):
        """Normalize and attach reference(s) to the answer."""
        self.references = normalize_references(references)
        return self

    @classmethod
    def from_lazy(
        cls,
        query_id: str,
        question: str,
        gold_answers: Optional[List[Answer | str] | str | Answer] = None,
        references: Optional[Union[str, Reference, List[str], List[Reference]]] = None,
        creator: Optional[Union[List[Creator], Creator]] = None,
        **kwargs,
    ) -> "NuggetQuestion":
        """Convenient constructor that accepts raw strings or Answers and performs normalization."""

        gold_answers_norm = (
            hash_answers(normalize_answers(gold_answers))
            if gold_answers is not None
            else None
        )

        return cls(
            query_id=query_id,
            question=question,
            answers=gold_answers_norm,
            references=normalize_references(references),
            creator=normalize_creators(creator),
            **kwargs,
        )


def merge_nugget_questions(nuggets: List[NuggetQuestion]) -> NuggetQuestion:
    """Merge multiple NuggetQuestion instances into one, combining answers, sub-nuggets, and metadata."""

    if len(nuggets) == 0:
        raise ValueError("list of nuggets can't be empty")

    base = nuggets[0]

    return NuggetQuestion(
        question=base.question,
        question_id=base.question_id,
        question_markup=base.question_markup,
        answers=Answer.merge_answer_dicts(
            [n.answers for n in nuggets if n.answers is not None]
        ),
        sub_nuggets=[
            sub for n in nuggets if n.sub_nuggets is not None for sub in n.sub_nuggets
        ],
        aggregator_type=base.aggregator_type,
        query_id=base.query_id,
        test_collection=base.test_collection,
        quality=aggregate_unless_none_list(
            anss=[n.quality for n in nuggets],
            aggregate_fun=agg_take_right,
            default_val=None,
        ),  # type: Optional[int | str]
        importance=aggregate_unless_none_list(
            [n.importance for n in nuggets if n.importance is not None],
            aggregate_fun=agg_take_right,
        ),  # type: Optional[int | str],
        creator=Creator.merge_creator_list(
            [n.creator for n in nuggets if n.creator is not None],
        ),
        metadata=merge_metadata_list(
            [n.metadata for n in nuggets if n.metadata is not None]
        ),
    )


class NuggetClaim(BaseModel):
    """Represents a fact-like claim with optional references and provenance."""

    claim: str
    claim_markup: Optional[Any] = None
    references: Optional[List[Reference]] = None

    sub_nuggets: Optional[List[Union[NuggetQuestion, "NuggetClaim"]]] = None
    aggregator_type: Optional[AggregatorType] = None

    claim_id: Optional[str] = None
    query_id: str
    test_collection: Optional[str] = None

    quality: Optional[int | str] = None
    importance: Optional[int | str] = None
    creator: Optional[List[Creator]] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = {"recursive": True, "extra": "ignore"}

    def model_post_init(self, __context):
        """Assign a hash-based ID if none is given, based on the claim text."""
        if self.claim_id is None:
            self.claim_id = hashlib.md5(self.claim.encode()).hexdigest()

    def add_creator(self, creator: Optional[Union[List[Creator], Creator]]):
        """Normalize and attach creator(s) to the nugget."""
        self.creator = normalize_creators(creator)
        return self

    def add_references(
        self,
        references: Optional[Union[str, Reference, List[str], List[Reference]]],
    ):
        """Normalize and attach reference(s) to the claim."""
        self.references = normalize_references(references)
        return self

    @classmethod
    def from_lazy(
        cls,
        query_id: str,
        claim: str,
        references: Optional[Union[str, Reference, List[str], List[Reference]]] = None,
        creator: Optional[Union[List[Creator], Creator]] = None,
        **kwargs,
    ) -> "NuggetClaim":
        """Convenient constructor that accepts raw strings and performs normalization."""
        return cls(
            query_id=query_id,
            claim=claim,
            references=normalize_references(references),
            creator=normalize_creators(creator),
            **kwargs,
        )


def merge_nugget_claims(claims: List[NuggetClaim]) -> NuggetClaim:
    """Merge multiple NuggetClaim instances into one, combining sub-nuggets, references, and metadata."""

    if len(claims) == 0:
        raise ValueError("list of claims can't be empty")

    base = claims[0]

    return NuggetClaim(
        claim=base.claim,
        claim_id=base.claim_id,
        claim_markup=base.claim_markup,
        references=opt_list_as_empty(base.references) + [
            ref
            for c in claims[1:]
            if c.references is not None
            for ref in c.references
        ],
        sub_nuggets=[
            sub for c in claims if c.sub_nuggets is not None for sub in c.sub_nuggets
        ],
        aggregator_type=base.aggregator_type,
        query_id=base.query_id,
        test_collection=base.test_collection,
        quality=aggregate_unless_none_list(
            anss=[c.quality for c in claims],
            aggregate_fun=agg_take_right,
            default_val=None,
        ),
        importance=aggregate_unless_none_list(
            [c.importance for c in claims if c.importance is not None],
            aggregate_fun=agg_take_right,
        ),
        creator=Creator.merge_creator_list(
            [c.creator for c in claims if c.creator is not None],
        ),
        metadata=merge_metadata_list(
            [c.metadata for c in claims if c.metadata is not None]
        ),
    )


# Ensure recursive model references are resolved (required in Pydantic v2).
Creator.model_rebuild()
Reference.model_rebuild()
Answer.model_rebuild()
NuggetQuestion.model_rebuild()
NuggetClaim.model_rebuild()


class NuggetBank(BaseModel):
    """Container for a set of nugget questions or claims, tied to a query and optionally including metadata."""

    query_id: Optional[str]
    title_query: str
    full_query: Any = None
    test_collection: Optional[str] = None
    format_version: str = "v3"

    nugget_bank: Optional[Dict[str, NuggetQuestion]] = None
    claim_bank: Optional[Dict[str, NuggetClaim]] = None
    # Read-only view of all nuggets (questions + claims) flattened including sub_nuggets
    # Excluded from serialization - rebuilt on load via index_nuggets()
    all_nuggets_view: Optional[Mapping[str, NuggetQuestion | NuggetClaim]] = Field(
        default=None, exclude=True
    )

    quality: Optional[int | str] = None
    importance: Optional[int | str] = None
    creator: Optional[List[Creator]] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = {"recursive": True, "extra": "ignore"}

    def nuggets_as_list(self) -> List[NuggetQuestion | NuggetClaim]:
        """Return combined list of all questions and claims."""
        result: List[NuggetQuestion | NuggetClaim] = []
        if self.nugget_bank is not None:
            result.extend(self.nugget_bank.values())
        if self.claim_bank is not None:
            result.extend(self.claim_bank.values())
        return result

    def _put_question(self, key: str, nugget: NuggetQuestion):
        """Insert a question into nugget_bank, merging if key exists."""
        if self.nugget_bank is None:
            self.nugget_bank = dict()

        if key not in self.nugget_bank:
            self.nugget_bank[key] = nugget
        else:
            self.nugget_bank[key] = merge_nugget_questions([self.nugget_bank[key], nugget])

    def _put_claim(self, key: str, claim: NuggetClaim):
        """Insert a claim into claim_bank, merging if key exists."""
        if self.claim_bank is None:
            self.claim_bank = dict()

        if key not in self.claim_bank:
            self.claim_bank[key] = claim
        else:
            self.claim_bank[key] = merge_nugget_claims([self.claim_bank[key], claim])

    def add_nuggets(
        self,
        nuggets: Optional[
            List[Union[NuggetQuestion, NuggetClaim]] | NuggetQuestion | NuggetClaim
        ],
    ) -> "NuggetBank":
        """Add one or more nuggets to the bank, dispatching by type and performing merge if needed."""

        if nuggets is None:
            return self

        if isinstance(nuggets, NuggetQuestion):
            self._put_question(key=nuggets.question, nugget=nuggets)
        elif isinstance(nuggets, NuggetClaim):
            self._put_claim(key=nuggets.claim, claim=nuggets)
        elif isinstance(nuggets, list):
            for elem in nuggets:
                if isinstance(elem, NuggetQuestion):
                    self._put_question(key=elem.question, nugget=elem)
                elif isinstance(elem, NuggetClaim):
                    self._put_claim(key=elem.claim, claim=elem)
                else:
                    raise ValueError(f"Nuggets of type {type(elem)} are not supported.")
        else:
            raise ValueError(f"Nuggets of type {type(nuggets)} are not supported.")

        self.index_nuggets()
        return self

    def add_creator(self, creator: Optional[Union[List[Creator], Creator]]):
        """Normalize and attach creator(s) to the nugget."""
        self.creator = normalize_creators(creator)
        return self

    @staticmethod
    def index_nuggets_internal(
        nugget_dictionary: Dict[str, NuggetQuestion], nugget_bank: List[NuggetQuestion]
    ):
        """Recursive utility to build a flat index of nuggets and sub-nuggets by text."""

        for entry in nugget_bank:
            # print(f"entry, {entry}")
            if isinstance(entry, NuggetQuestion):
                nugget_dictionary[entry.question] = entry
            elif isinstance(entry, NuggetClaim):
                nugget_dictionary[entry.claim] = entry
            else:
                raise ValueError(
                    f"index_nuggets_internal does not support nugget type {type(entry)}."
                )
            if entry.sub_nuggets is not None and len(entry.sub_nuggets) > 0:
                nugget_dictionary = NuggetBank.index_nuggets_internal(
                    nugget_dictionary, entry.sub_nuggets
                )  # type: ignore
        return nugget_dictionary

    def index_nuggets(self):
        """Compute a read-only flat index view of all nuggets (questions + claims) in the bank."""
        internal_dict: Dict[str, NuggetQuestion | NuggetClaim] = {}
        if self.nugget_bank is not None:
            internal_dict = NuggetBank.index_nuggets_internal(
                internal_dict, list(self.nugget_bank.values())
            )
        if self.claim_bank is not None:
            internal_dict = NuggetBank.index_nuggets_internal(
                internal_dict, list(self.claim_bank.values())
            )
        self.all_nuggets_view = MappingProxyType(internal_dict)

    def model_post_init(self, __context):
        """Assign a hash-based query_id from the title_query if none is provided."""
        if self.query_id is None:
            self.query_id = hashlib.md5(self.title_query.encode()).hexdigest()
        self.index_nuggets()


def nugget_to_dict(nugget: NuggetQuestion | NuggetClaim) -> Dict[str, Any]:
    """Convert a nugget (question or claim) to legacy v1 dict format."""
    if isinstance(nugget, NuggetQuestion):
        result: Dict[str, Any] = {
            "query_id": nugget.query_id,
            "question_text": nugget.question,
            "question_id": nugget.question_id,
            "info": {"aggregator_type": nugget.aggregator_type or AggregatorType.OR},
            "gold_answers": [
                {
                    "answer": answer.answer,
                    "citations": [ref.doc_id for ref in (answer.references or [])],
                }
                for answer in (nugget.answers or {}).values()
            ],
        }
        if nugget.references:
            result["references"] = [ref.doc_id for ref in nugget.references]
        return result
    elif isinstance(nugget, NuggetClaim):
        result = {
            "query_id": nugget.query_id,
            "claim_text": nugget.claim,
            "claim_id": nugget.claim_id,
            "info": {"aggregator_type": nugget.aggregator_type or AggregatorType.OR},
        }
        if nugget.references:
            result["references"] = [ref.doc_id for ref in nugget.references]
        return result
    else:
        raise RuntimeError(f"Don't know nugget type {type(nugget)}. Received: {nugget}")


# Backwards compatibility alias
question_nugget_to_dict = nugget_to_dict


# def get_doc_id_to_nuggets_mapping(nuggets:List[NuggetQuestion])->Dict[str, List[NuggetQuestion]]:
#     """Obtain a mapping from documents to nuggets answered by those documents

#     Replacement forc
#     def get_doc_id_to_nuggets_mapping(
#     nuggets: List[Dict[str, Any]],
# ) -> Dict[str, List[Dict[str, Any]]]:

#     doc_to_nugget_mapping = {}
#     for nugget in nuggets:
#         for gold_answer in nugget["gold_answers"]:
#             for doc_id in gold_answer["citations"]:
#                 if doc_id not in doc_to_nugget_mapping:
#                     doc_to_nugget_mapping[doc_id] = []
#                 doc_to_nugget_mapping[doc_id].append(nugget)
#     return doc_to_nugget_mapping
#     """
#     nugget:NuggetQuestion
#     answer:Answer
#     res:Dict[str, List[NuggetQuestion]] = collections.defaultdict(list)
#     for nugget in nuggets:
#         for answer in (nugget.answers or {}).values():
#             for ref in answer.references or []:


def str_nugget_json(d: BaseModel) -> str:
    """String of Pretty-printed a NuggetBank as JSON, omitting fields with None values."""

    return d.model_dump_json(indent=2, exclude_none=True)


def print_nugget_json(d: BaseModel):
    """Pretty-prints a NuggetBank as JSON to stdout, omitting fields with None values."""

    print(str_nugget_json(d))


def write_nugget_json(nugget_bank: NuggetBank, out: TextIO):
    """Writes a NuggetBank to a file-like object as indented JSON, excluding None fields."""

    json.dump(nugget_bank.model_dump(exclude_none=True), out, indent=2)


def write_nugget_bank_json(
    nugget_bank: NuggetBank, out: Union[str, Path, TextIO]
) -> None:
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
            json.dump(nugget_bank.model_dump(exclude_none=True), f, indent=2)
    else:
        # Assume it's already a valid TextIO stream
        json.dump(nugget_bank.model_dump(exclude_none=True), out, indent=2)


M = TypeVar("M", bound=BaseModel)


def load_model_json(cls: Type[M], source: Union[str, Path, TextIO]) -> M:
    """
    Load any Pydantic model instance from JSON, supporting plain files, gzip-compressed files, and text streams.

    This function supports reading from:
      - A plain text file path (as `str` or `Path`)
      - A gzip-compressed file path ending in `.gz`
      - An open `TextIO` stream (e.g., `open(..., 'r')`, `gzip.open(..., 'rt')`, or `io.StringIO`)

    Args:
        cls (Type[T]): The Pydantic model class to load (e.g., `MyModel`).
        source (Union[str, Path, TextIO]): The source JSON input. If a string or Path is provided,
            it will be opened in read-text mode (`"rt"`). If the path ends in `.gz`, it will be
            automatically decompressed using gzip.

    Returns:
        T: An instance of the specified Pydantic model.

    Example:
        >>> model = load_model_json(MyModel, "input.json")
        >>> model = load_model_json(MyModel, "input.json.gz")
        >>> with open("input.json", "r") as f:
        >>>     model = load_model_json(MyModel, f)
        >>> with gzip.open("input.json.gz", "rt") as f:
        >>>     model = load_model_json(MyModel, f)
    """
    if isinstance(source, (str, Path)):
        open_fn = gzip.open if str(source).endswith(".gz") else open
        with open_fn(source, mode="rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        # Assume it's an already open text stream
        data = json.load(source)

    return cls.model_validate(data)


def load_nugget_bank_json(source: Union[str, Path, TextIO]) -> NuggetBank:
    """
    Load a NuggetBank Pydantic model instance from JSON, supporting plain files, gzip-compressed files, and text streams.

    This function supports reading from:
      - A plain text file path (as `str` or `Path`)
      - A gzip-compressed file path ending in `.gz`
      - An open `TextIO` stream (e.g., `open(..., 'r')`, `gzip.open(..., 'rt')`, or `io.StringIO`)

    """
    return load_model_json(cls=NuggetBank, source=source)


def filter_questions_with_no_answers(
    questions: Dict[str, NuggetQuestion | NuggetClaim],
) -> Dict[str, NuggetQuestion | NuggetClaim]:
    """Filter out questions that have no answers. Claims always pass (they're fact assertions)."""
    return {
        k: v
        for k, v in questions.items()
        if isinstance(v, NuggetClaim)  # Claims always pass
        or (isinstance(v, NuggetQuestion) and v.answers and len(v.answers) > 0)
    }


def filter_empty_citations(nuggetbank: NuggetBank, filter_citations=True) -> NuggetBank:
    """Remove answers with no references from the NuggetBank."""
    if nuggetbank.nugget_bank is None:
        return nuggetbank

    for nugget in nuggetbank.nugget_bank.values():
        if isinstance(nugget, NuggetQuestion):
            nugget.answers = (
                filter_answers_with_no_references(nugget.answers)
                if filter_citations
                else nugget.answers
            )
    # drop questions with no answers
    nuggetbank.nugget_bank = (
        filter_questions_with_no_answers(nuggetbank.nugget_bank)
        if filter_citations
        else nuggetbank.nugget_bank
    )
    return nuggetbank
