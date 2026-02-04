# Qrels Module

This module provides safe way to construct, verify, and serialize *qrels* (relevance judgments) for IR / RAG evaluation.

It is designed to:

* avoid silent mismatches and partial evaluations,
* keep task-specific logic (how doc_ids or grades are produced) outside the core,
* make verification explicit and opt-in,
* integrate cleanly with TREC-style tooling.

---

## Core Concepts

### `QrelRow`

```python
@dataclass(frozen=True)
class QrelRow:
    topic_id: str
    doc_id: str
    grade: int
```

A single relevance judgment.

* `topic_id`: query or topic identifier
* `doc_id`: document identifier (string), can be md5 generated
* `grade`: non-negative relevance grade (e.g., 0/1 or 0–3)

The module deliberately does **not** impose semantics on `doc_id`.
Whether it refers to a corpus document, a generated passage, or a hash is up to the calling code.

---

### `Qrels`

```python
@dataclass(frozen=True)
class Qrels:
    rows: Sequence[QrelRow]
```

A container for qrels.

`Qrels` as a dataclass:

* no assumptions about document provenance,
* no assumptions about grading scale,
* no built-in policy.

This keeps the abstraction boundary clean and reusable.

---

## Building Qrels

### `QrelsSpec`

```python
@dataclass(frozen=True)
class QrelsSpec(Generic[R]):
    topic_id: Callable[[R], str]
    doc_id: Callable[[R], str]
    grade: Callable[[R], int]
    on_duplicate: str = "error"
```

A `QrelsSpec` defines **how to extract qrels from records**.

Each field is a function:

* `topic_id(record) -> str`
* `doc_id(record) -> str` (already resolved)
* `grade(record) -> int`

`on_duplicate` controls what happens if the same `(topic_id, doc_id)` appears multiple times:

* `"error"` (default): fail fast
* `"keep_max"`: keep the maximum grade
* `"keep_last"`: overwrite with the last seen grade

---

### `build_qrels`

```python
qrels = build_qrels(records=records, spec=spec)
```

Builds a `Qrels` object from an iterable of records.

Properties:

* enforces uniqueness of `(topic_id, doc_id)` pairs,
* centralizes duplicate handling,

Example:

```python
spec = QrelsSpec[MyRecord](
    topic_id=lambda r: r.query_id,
    doc_id=lambda r: r.doc_id,
    grade=lambda r: int(r.is_relevant),
)

qrels = build_qrels(records=my_records, spec=spec)
```

---

## Generated Document IDs (Helper)

### `doc_id_md5`

```python
def doc_id_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()
```

A small helper for generating stable document identifiers from text.

**Important**:

* This function is not used by the Qrels module itself.
* It is intended for *implementations* (e.g., Umbrela) that need consistent IDs for generated or extracted text.

Recommended usage pattern:

```python
def resolve_doc_id(record):
    if record.corpus_doc_id is not None:
        return f"corpus:{record.corpus_doc_id}"
    return f"md5:{doc_id_md5(record.generated_text)}"
```

---

## Verification

Qrels verification is separate from construction.

### Quick Verification via `.verify()`

The simplest way to verify qrels:

```python
topic_ids = [t.request_id for t in topics]
qrels.verify(
    expected_topic_ids=topic_ids,
    warn=False  # raise exceptions on failure
)
```

Parameters:
- `expected_topic_ids`: List of topic IDs that should be present
- `warn`: If `True`, print warnings instead of raising exceptions

### Detailed Verification via `QrelsVerification`

For more granular control or test cases, use the fluent `QrelsVerification` class:

```python
from trec_auto_judge.qrels.qrels import QrelsVerification

# Chain specific checks
QrelsVerification(
    qrels,
    expected_topic_ids=topic_ids,
    warn=False
).complete_topics().no_extra_topics().no_duplicates()

# Or run all checks
QrelsVerification(qrels, expected_topic_ids=topic_ids).all()
```

Available verification methods:

| Method | Description |
|--------|-------------|
| `complete_topics()` | Every expected topic has at least one qrel row |
| `no_extra_topics()` | No qrels for unexpected topics |
| `no_duplicates()` | No duplicate `(topic_id, doc_id)` pairs |
| `all()` | Run all checks |

This prevents:

* silent topic drop-out,
* evaluating on a subset of intended queries,
* accidental topic mixing across datasets.

---

## Serialization

### `write_qrel_file`

```python
write_qrel_file(
    qrel_out_file=Path("run.qrels"),
    qrels=qrels,
)
```

Writes qrels in **standard TREC format**:

```
topic_id  system_id  doc_id  grade
```

Notes:

* `system_id` defaults to `"0"` (conventional for human or synthetic judgments).
* Output is **deterministically sorted** by `(topic_id, doc_id)` for reproducibility.
* Compatible with `trec_eval`, `pytrec_eval`, `ir_measures`, etc.

---

## Design Philosophy (Why it looks like this)

* **No stringly-typed coupling**: qrels are built from a single `QrelsSpec`, not from parallel dicts.
* **No hidden policy**: doc_id semantics, hashing, and corpora live outside this module.
* **Fail fast**: duplicates and missing topics raise immediately.
* **Composable**: fits naturally alongside leaderboards, judges, and generated-text corpora.

If you already made your Leaderboard safer using verification and spec-based construction, this module is intended to feel familiar.

---

## Typical Usage Pattern

```python
# 1. Build qrels
qrels = build_qrels(records=annotations, spec=spec)

# 2. Verify coverage
verify_qrels_topics(
    expected_topic_ids=[t.query_id for t in topics],
    qrels=qrels,
    require_no_extras=True,
)

# 3. Serialize
write_qrel_file(qrel_out_file=output.with_suffix(".qrels"), qrels=qrels)
```


