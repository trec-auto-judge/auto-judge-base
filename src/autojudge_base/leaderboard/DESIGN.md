# Leaderboard System Design

> **Note for Claude:** This DESIGN.md documents the leaderboard module's architecture and invariants.
> **Read this document before modifying any leaderboard code.** The design has specific invariants
> that are easy to accidentally violate. Check the Key Invariants and Anti-Patterns sections first.

## Key Invariants (Read First)

1. **All construction goes through LeaderboardBuilder** - Never create `Leaderboard` directly except in `build()`
2. **`_handle_aggregates()` is the ONLY handler for `drop_aggregate`** - Don't filter "all" entries elsewhere
3. **Validation happens in `add()`** - Unknown/missing measures fail immediately via `KeyError`
4. **Casting happens in `add()`** - Values are always cast before storage, ensuring type consistency
5. **`for_leaderboard()` handles spec filtering** - When loading with `drop_aggregate=True`, use this factory to exclude "all"-only measures

## Anti-Patterns

❌ **Don't skip "all" entries before calling `add()`** - Let `_handle_aggregates()` decide what to keep/drop

❌ **Don't create `LeaderboardSpec` manually after loading** - Use `for_leaderboard()` factory which derives the correct spec

❌ **Don't check `drop_aggregate` in multiple places** - It should only affect behavior in `_handle_aggregates()` and `for_leaderboard()`

❌ **Don't bypass the builder** - Even for simple transformations, go through `LeaderboardBuilder` to ensure casting and validation

## Overview

The leaderboard system provides type-safe, schema-driven storage and manipulation of evaluation results. It separates concerns between data storage (`Leaderboard`), construction logic (`LeaderboardBuilder`), and schema definition (`LeaderboardSpec`/`MeasureSpec`).

## Core Components

### Leaderboard (Data Container)

**Responsibility:** Immutable vessel for leaderboard data. No construction logic.

```python
@dataclass(frozen=True)
class Leaderboard:
    measures: Tuple[MeasureName, ...]      # Measure names in schema order
    entries: Tuple[LeaderboardEntry, ...]  # Per-topic + "all" rows
    all_topic_id: str = "all"              # Aggregate row identifier
    spec: Optional[LeaderboardSpec]        # Schema (always set after load/build)
```

**What it does:**
- Stores entries (per-topic rows and aggregate "all" rows)
- Provides read-only access: `run_ids`, `topic_ids`, `get_aggregate_ranking()`
- Serialization: `write()` to various formats
- Delegates to builder for transformations: `filter_and_recompute()`

**What it does NOT do:**
- No entry validation
- No aggregate computation
- No dtype casting

### LeaderboardBuilder (Construction Logic)

**Responsibility:** All leaderboard construction happens here.

```python
class LeaderboardBuilder:
    def __init__(self, spec: LeaderboardSpec): ...

    # Factory methods
    @classmethod
    def for_leaderboard(cls, lb, drop_aggregate=True): ...  # Create builder from existing leaderboard

    # Adding entries
    def add(*, run_id, topic_id, values): ...      # Single entry
    def add_records(records, ...): ...              # From arbitrary objects
    def add_entries(entries, skip_all_rows): ...   # From LeaderboardEntry objects
    def add_from_all(leaderboards, ...): ...       # Merge multiple leaderboards

    # Building
    def build(expected_topic_ids, on_missing, drop_aggregate) -> Leaderboard: ...

    # Internal
    def _handle_aggregates(entries, drop_aggregate, phantom_defaults): ...
```

**Responsibilities:**
1. **Validation** - Unknown/missing measures raise `KeyError`
2. **Casting** - Values cast via `MeasureSpec.get_cast()`
3. **Aggregate handling** - Drop and/or recompute "all" rows via `_handle_aggregates()`
4. **Missing entry handling** - Fill defaults or phantom defaults per `on_missing`

**Design principle:** Any operation that creates or transforms a leaderboard goes through the builder:
- Judge creates results → `builder.add()` → `builder.build()`
- Load from file → `_load_file()` → `_build_from_entries()` → `builder.add()` → `builder.build()`
- Merge files (directory) → `_load_directory()` → `builder.add_from_all()` → `builder.build()`
- Filter → `filter_and_recompute()` → `builder.add()` → `builder.build()`

### MeasureSpec (Measure Schema)

**Responsibility:** Define a single measure's name and processing behavior.

```python
@dataclass(frozen=True)
class MeasureSpec:
    name: MeasureName
    dtype: type = float  # Only float or str allowed

    def get_cast(self) -> CastFn: ...
    def get_aggregate(self) -> AggFn: ...
    def get_default(self) -> Any: ...
```

**dtype → behavior mapping:**

| dtype | cast | aggregate | default |
|-------|------|-----------|---------|
| `float` | `float(x)` | mean | 0.0 |
| `str` | `str(x)` | first_value | "" |

**Only `float` and `str` are allowed.** This ensures dtype survives save/load round-trips:
- Numeric values → `float`
- Non-numeric values → `str`

If you have boolean data, store it as `1.0`/`0.0` with `dtype=float`. The aggregation (mean) gives you the fraction of `True` values, which is typically what you want.

### LeaderboardSpec (Leaderboard Schema)

**Responsibility:** Collection of MeasureSpecs defining the full schema.

```python
@dataclass(frozen=True)
class LeaderboardSpec:
    measures: Tuple[MeasureSpec, ...]
    all_topic_id: str = "all"

    @property
    def names(self) -> Tuple[str, ...]: ...
    @property
    def name_set(self) -> Set[str]: ...
    def cast_values(self, values) -> Dict[str, Any]: ...
```

## MeasureSpec Creation: Two Paths

### Path 1: Explicit Creation (Judge Implementation)

When implementing a judge, the developer explicitly defines the spec:

```python
MY_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("GRADE"),              # dtype=float (default)
    MeasureSpec("IS_MATCH"),           # dtype=float - use 1.0/0.0 for boolean
    MeasureSpec("CATEGORY", str),      # dtype=str
))

class MyJudge(AutoJudge):
    def judge(self, ...) -> Leaderboard:
        builder = LeaderboardBuilder(MY_SPEC)
        builder.add(run_id="run1", topic_id="topic1", GRADE=0.8, IS_MATCH=1.0, CATEGORY="A")
        return builder.build()
```

**Characteristics:**
- Developer knows the semantics
- dtype chosen intentionally
- Validation catches typos at build time

### Path 2: Inference from Values (load() only)

When loading from files, dtype is inferred from string values:

```python
def _infer_dtype_from_values(values: Sequence[Any]) -> type:
    # All parse as float → float
    # Otherwise → str
```

**Only `_load_file()` uses inference.** The inference is conservative:
- If all values parse as numbers → `float`
- Otherwise → `str`

**`bool` and `int` are not allowed as dtypes** because they can't survive save/load round-trips. Use `float` with `1.0`/`0.0` for boolean data, and `float` for integer counts (no precision loss for typical values).

After loading, the spec is explicit and routed through all subsequent operations.

## The `drop_aggregate` Flag

### Purpose

Leaderboard files may contain pre-computed "all" aggregate rows. The `drop_aggregate` flag controls whether to:
- **Keep them** (`drop_aggregate=False`): Use original aggregates as-is
- **Recompute them** (`drop_aggregate=True`): Drop original, compute fresh from per-topic data

### Why Recompute?

1. **Subset filtering**: When evaluating on a subset of topics, aggregates must be recomputed
2. **Consistency**: Aggregates may have been computed differently (different formula, missing topics)
3. **Measures only in "all" rows**: Some files have measures that don't exist per-topic; these can't be recomputed

### Flag Flow

```
User code / CLI
      │
      ▼
Leaderboard.load(path, drop_aggregate=True/False)
      │
      ▼
_load_file() / _load_directory()  ──► Returns raw Leaderboard with spec
      │
      ▼
_build_from_entries(lb, drop_aggregate, on_missing)
      │
      ▼
LeaderboardBuilder.for_leaderboard(lb, drop_aggregate)  ──► Factory creates builder with appropriate spec
      │
      ├─► drop_aggregate=True: spec from per-topic measures only
      └─► drop_aggregate=False: full spec from leaderboard
      │
      ▼
builder.add() for each entry  ──► All entries added (casting happens here)
      │
      ▼
builder.build(drop_aggregate=?)
      │
      ▼
builder._handle_aggregates(entries, drop_aggregate, phantom_defaults)  ◄── SINGLE HANDLER
      │
      ├─► drop_aggregate=True:
      │     • Filter out existing "all" entries
      │     • Compute fresh aggregates from per-topic entries
      │
      └─► drop_aggregate=False:
            • Keep original "all" entries (already cast)
            • Skip aggregate computation
```

**Single responsibility:** `LeaderboardBuilder._handle_aggregates()` is the only place that handles `drop_aggregate` logic. This ensures casting always happens (via the builder) regardless of the flag.

### Responsibility Split for "all" Handling

Handling "all" topics involves three aspects:

| Aspect | Handler | Purpose |
|--------|---------|---------|
| **3a. Spec filtering** | `for_leaderboard(drop_aggregate=True)` | Exclude "all"-only measures from spec |
| **3b. Value filtering** | `_build_from_entries()` | Filter entry values to match filtered spec |
| **3c. Entry filtering** | `_handle_aggregates()` | Drop/keep/recompute "all" entries |

**Which code paths use which:**

| Code Path | 3a | 3b | 3c | Notes |
|-----------|----|----|----|----|
| Judge implementation | ✗ | ✗ | ✓ | Developer controls spec, no "all"-only measures |
| `Leaderboard.load()` | ✓ | ✓ | ✓ | External data may have "all"-only measures |
| `filter_and_recompute()` | ✗ | ✗ | ✓ | Already-processed data, manually skips "all" |

**Why this split is intentional:**

- **3a and 3b are load-specific**: Only needed when reading external files where "all" rows may have different measures than per-topic rows. The `for_leaderboard()` factory and `_build_from_entries()` handle this.

- **3c is universal**: All code paths go through `build()` → `_handle_aggregates()`, which is the single handler for deciding whether to drop/keep/recompute "all" entries.

### Related: filter_and_recompute()

When filtering a leaderboard (subset of topics/runs), aggregates must always be recomputed:

```python
def filter_and_recompute(self, run_ids, topic_ids) -> Leaderboard:
    builder = LeaderboardBuilder(self.spec)
    # Add only matching per-topic entries (skip "all" rows)
    for e in self.entries:
        if e.topic_id == self.all_topic_id:
            continue
        if run_ids and e.run_id not in run_ids:
            continue
        if topic_ids and e.topic_id not in topic_ids:
            continue
        builder.add(...)
    return builder.build(expected_topic_ids=..., on_missing="fix_aggregate")
```

This always recomputes aggregates because the subset changes what should be aggregated.

## Connection to LeaderboardEvaluator

`LeaderboardEvaluator` computes correlation between judge leaderboards and ground truth:

```python
class LeaderboardEvaluator:
    def __init__(self, ground_truth: Leaderboard, ...): ...

    def evaluate(self, judge_leaderboard: Leaderboard) -> CorrelationResult:
        # Filter both to common topics/runs
        # Extract rankings from "all" rows
        # Compute Kendall's tau, Kendall@k, etc.
```

**Key interaction:**
1. Ground truth loaded with `drop_aggregate=True` (recompute for subset)
2. Judge leaderboard produced by `AutoJudge.judge()` via builder
3. Both filtered to common topics via `filter_and_recompute()`
4. Rankings extracted from "all" rows for correlation

## Connection to AutoJudgeProtocol

```python
class AutoJudge(Protocol):
    def judge(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        **kwargs
    ) -> Leaderboard: ...
```

**Judge implementation pattern:**

```python
class MyJudge(AutoJudge):
    def judge(self, rag_responses, rag_topics, llm_config, **kwargs) -> Leaderboard:
        builder = LeaderboardBuilder(MY_SPEC)

        for response in rag_responses:
            # Compute measures for this response
            grade = compute_grade(response, llm_config)

            builder.add(
                run_id=response.metadata.run_id,
                topic_id=response.metadata.topic_id,
                GRADE=grade,
            )

        expected_topics = [t.request_id for t in rag_topics]
        return builder.build(expected_topic_ids=expected_topics, on_missing="fix_aggregate")
```

**The contract:**
- Judge receives responses and topics
- Judge produces a `Leaderboard` via `LeaderboardBuilder`
- Aggregates are computed by the builder
- Missing entries handled per `on_missing` policy

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                        JUDGE IMPLEMENTATION                          │
│                                                                      │
│  LeaderboardSpec ──► LeaderboardBuilder ──► Leaderboard             │
│  (explicit dtype)         │                                          │
│                     add() / add_records()                            │
│                           │                                          │
│                     build() ──► validate, cast, compute aggregates   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        LOADING FROM FILES                            │
│                                                                      │
│  File ──► _load_file() ──► Leaderboard (with inferred spec)         │
│                │                                                     │
│          _infer_dtype_from_values()                                  │
│                │                                                     │
│                ▼                                                     │
│  _build_from_entries(drop_aggregate=?)                              │
│                │                                                     │
│                ▼                                                     │
│  LeaderboardBuilder.for_leaderboard(lb, drop_aggregate)             │
│                │                                                     │
│                ├─► drop_aggregate=True: spec from per-topic only     │
│                └─► drop_aggregate=False: full spec                   │
│                │                                                     │
│                ▼                                                     │
│  builder.add() for each entry ──► casting happens here              │
│                │                                                     │
│                ▼                                                     │
│  builder.build(drop_aggregate=?)                                    │
│                │                                                     │
│                ▼                                                     │
│  _handle_aggregates() ──► drop or keep "all" rows                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        MERGING LEADERBOARDS                          │
│                                                                      │
│  [lb1, lb2, ...] ──► LeaderboardBuilder(lb1.spec)                   │
│                           │                                          │
│                     add_from_all([lb1, lb2, ...])                    │
│                           │                                          │
│                     build() ──► merged with fresh aggregates         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        FILTERING                                     │
│                                                                      │
│  Leaderboard ──► filter_and_recompute(topic_ids=subset)             │
│                           │                                          │
│                     LeaderboardBuilder(self.spec)                    │
│                           │                                          │
│                     add_entries() (filtered)                         │
│                           │                                          │
│                     build() ──► filtered with fresh aggregates       │
└─────────────────────────────────────────────────────────────────────┘
```

## Design Principles

1. **Single point of construction**: All leaderboard creation goes through `LeaderboardBuilder`
2. **Immutable containers**: `Leaderboard`, `LeaderboardSpec`, `MeasureSpec` are frozen dataclasses
3. **Fail-fast validation**: Unknown/missing measures raise immediately
4. **Explicit over implicit**: Judges define specs explicitly; inference only for file loading
5. **Spec routing**: Once a spec exists, it's routed through all operations (never recreated)
6. **Single handler for flags**: `drop_aggregate` handled in one place (`LeaderboardBuilder._handle_aggregates`)
