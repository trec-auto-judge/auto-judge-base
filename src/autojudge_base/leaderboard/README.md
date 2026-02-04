# Leaderboard + Builder

This module defines a simple, safe way to construct a leaderboard from per-topic scores while computing
per-run aggregate ("all") rows automatically.

The design avoids the common failure mode where measure names are repeated in multiple places as raw
strings (and drift silently). Instead, `LeaderboardSpec` defines:

- which measures exist
- how to cast/normalize measure values
- how to aggregate per-topic values into an `"all"` row

`Leaderboard` itself is a thin, serializable dataclass (plus a `write()` method).


## Concepts

### `Leaderboard` (dataclass)
A `Leaderboard` contains:

- `measures`: an ordered tuple of measure names (schema)
- `entries`: all rows, including synthetic `"all"` rows
- `all_topic_id`: the topic id used for aggregated rows (default `"all"`)

It intentionally does *not* know how to aggregate. It is safe to serialize and write.

### `LeaderboardSpec` (build-time schema)
A `LeaderboardSpec` defines the leaderboard schema and build-time behavior:

- `MeasureSpec.name`: the measure key
- `MeasureSpec.cast`: converts/normalizes input values when rows are added
- `MeasureSpec.aggregate`: computes the `"all"` value from per-topic values

### `LeaderboardBuilder` (assembler)
A `LeaderboardBuilder` is the only place where:

- per-topic rows are added
- measure keys are validated (fail-fast on typos / missing keys)
- values are cast
- synthetic `"all"` rows are computed

---

## Quickstart

### Define aggregators for `all'

or use from leaderboard module.

Example

```python
from statistics import mean

def mean_of_bools(values):
    return mean(1.0 if bool(v) else 0.0 for v in values)
```

## Define a Leaderboard spec


```python

MY_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("GRADE", aggregate=mean_of_floats, cast=float),
    MeasureSpec("IS_MATCH", aggregate=mean_of_bools, cast=bool),
))
```

## Build a Leaderboard

This is the simplest interface to build a leaderboard.

```python

from pathlib import Path

b = LeaderboardBuilder(MY_SPEC)

b.add(run_id="runA", topic_id="t1", GRADE=0.9, IS_MATCH=True)
b.add(run_id="runA", topic_id="t2", GRADE=0.4, IS_MATCH=False)

lb = b.build()
lb.write(Path("leaderboard.tsv"))

```


## Advanced: record-builder for leaderboard

If you already have a sequence of objects from which different measures are derived, you can use `add_records` of the leaderboard builder.

```python
    b = LeaderboardBuilder(MY_SPEC)
    b.add_records(
        prompt_output,
        run_id=lambda r: r.run_id,
        topic_id=lambda r: r.query_id,
        get_values=lambda r: {c
            "GRADE": r.match_score,
            "IS_MATCH": r.is_match,
        },
    )
    return b.build()
```

This is safe because measure names ("GRADE", "IS_MATCH") live in the spec. If get_values contains a typo key (e.g., "ISMATCH"), or are missing, then builder.add(...) raises an error.


## Output format of Leaderboard.write

`Leaderboard.write()` produces white-space separated lines akin to `trec_evals -q`. 

Format `tot`:


```
run_id measure topic_id value
```

runA    GRADE    t1     0.9
runA    IS_MATCH t1     True
runA    GRADE    all    0.65
runA    IS_MATCH all    0.5


Format `ir_measures`:


```
run_id topic_id measure value
```
runA     1001    nugget_coverage_weighted        0.50
runA     1001    f1      0.54
runA     all     nugget_coverage_weighted_macro  0.47
runA     all     f1_macro        0.51



The leaderboard will automatically include an `all` topic based on the aggregator



## Verification

The leaderboard module includes verification utilities to catch *silent* data issues early.

### Quick Verification via `.verify()`

The simplest way to verify a leaderboard:

```python
topic_ids = [t.request_id for t in topics]
leaderboard.verify(
    on_missing="error",
    expected_topic_ids=topic_ids,
    warn=False  # raise exceptions on failure
)
```

Parameters:
- `on_missing`: How to handle missing topics - `"error"` (raise), `"warn"` (print warning), `"default"` (fill defaults), `"fix_aggregate"` (only fix aggregation)
- `expected_topic_ids`: List of topic IDs that should be present
- `warn`: If `True`, print warnings instead of raising exceptions

### Detailed Verification via `LeaderboardVerification`

For more granular control or test cases, use the fluent `LeaderboardVerification` class:

```python
from trec_auto_judge.leaderboard.leaderboard import LeaderboardVerification

# Chain specific checks
LeaderboardVerification(
    leaderboard,
    on_missing="error",
    expected_topic_ids=topic_ids,
    warn=False
).complete_measures().complete_topics().no_extra_topics()

# Or run all checks
LeaderboardVerification(
    leaderboard,
    on_missing="error",
    expected_topic_ids=topic_ids
).all()
```

Available verification methods:

| Method | Description |
|--------|-------------|
| `complete_measures(include_all_row=True)` | Every entry has all expected measure keys |
| `same_topics_per_run(include_all_row=False)` | All runs have the same set of topics |
| `complete_topics(include_all_row=False)` | Every expected topic has entries |
| `no_extra_topics(include_all_row=False)` | No entries for unexpected topics |
| `all(include_all_row=True)` | Run all applicable checks |

These checks are designed to raise failures immediately rather than silently propagating into incorrect aggregate scores or wrong/empty leaderboards.
