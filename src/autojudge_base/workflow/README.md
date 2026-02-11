# AutoJudge Workflow Guide

## Quick Start

A judge is any class with a `judge()` method that returns a `Leaderboard`:

```python
from autojudge_base import Leaderboard, LeaderboardBuilder, LeaderboardSpec, MeasureSpec

MY_SPEC = LeaderboardSpec(measures=(MeasureSpec("MY_SCORE"),))

class MyJudge:
    def judge(self, rag_responses, rag_topics, llm_config, **kwargs):
        builder = LeaderboardBuilder(MY_SPEC)
        for response in rag_responses:
            score = evaluate(response)
            builder.add(run_id=response.metadata.run_id,
                        topic_id=response.metadata.topic_id,
                        values={"MY_SCORE": score})
        topic_ids = [t.request_id for t in rag_topics]
        return builder.build(expected_topic_ids=topic_ids, on_missing="fix_aggregate")

    def create_nuggets(self, *args, **kwargs):
        return None

    def create_qrels(self, *args, **kwargs):
        return None
```

Register in `workflow.yml` and run:

```yaml
judge_class: "judges.myjudge:MyJudge"
```

```bash
auto-judge run --workflow workflow.yml \
    --rag-responses runs/ --rag-topics topics.jsonl --out-dir ./output/
```

## workflow.yml

The workflow file controls the judge pipeline. At minimum it declares which class to use:

```yaml
judge_class: "judges.myjudge:MyJudge"
```

### Execution Phases

The framework runs up to three phases in order: (1) nuggets, (2) qrels, (3) leaderboard.

| Configuration | create_nuggets | create_qrels | judge | Description |
|--------------|----------------|--------------|-------|-------------|
| Judge only | `false` | `false` | `true` | Score responses directly |
| Nuggets + judge | `true` | `false` | `true` | Create nuggets, then judge with them |
| Full pipeline | `true` | `true` | `true` | Nuggets, then qrels, then judge |
| Nuggets only | `true` | `false` | `false` | Create nugget banks without judging |

If `create_nuggets` / `create_qrels` are omitted, they are derived from the lifecycle flags: `create_nuggets` defaults to `judge_uses_nuggets or qrels_uses_nuggets`, and `create_qrels` defaults to `judge_uses_qrels`.

### Settings

Pass hyperparameters via settings dicts. Phase-specific settings override shared ones:

```yaml
settings:
  filebase: "{_name}"
  top_k: 20

nugget_settings:
  extraction_style: "thorough"

judge_settings:
  threshold: 0.5

qrels_settings:
  grade_range: [0, 3]
```

Settings are passed as `**kwargs` to the corresponding method:
- `judge()` receives `judge_settings` merged over `settings`
- `create_nuggets()` receives `nugget_settings` merged over `settings`
- `create_qrels()` receives `qrels_settings` merged over `settings`

**Template variables** use `{variable}` syntax in string values:

| Variable | Description |
|----------|-------------|
| `{_name}` | Configuration name ("default", variant name, or sweep name) |
| `{_nugget_filebase}` | Resolved nugget output path (available in judge/qrels settings) |
| `{_qrels_filebase}` | Resolved qrels output path (available in judge settings) |

User-defined parameters are also available: `{top_k}` expands to the value of `top_k`. Parameters starting with `_` are reserved for built-ins.

**Framework-consumed settings** are processed by the framework and not passed to judge methods:

| Setting | Description |
|---------|-------------|
| `llm_model` | Override `llm_config.model` for this run |

### Variants

Named configurations that override base settings:

```yaml
settings:
  threshold: 0.5
  filebase: "{_name}"

variants:
  strict:
    threshold: 0.8

  lenient:
    threshold: 0.3
    judge_settings:
      use_citations: true
```

Variants can override shared settings and also include `nugget_settings`, `judge_settings`, or `qrels_settings` blocks.

```bash
auto-judge run --workflow workflow.yml --variant strict ...
auto-judge run --workflow workflow.yml --all-variants ...
```

### Parameter Sweeps

Grid search over parameter combinations:

```yaml
sweeps:
  threshold-grid:
    top_k: [10, 20]
    threshold: [0.3, 0.5, 0.8]
```

```bash
auto-judge run --workflow workflow.yml --sweep threshold-grid ...
```

This produces 6 configurations (2 x 3 cartesian product), each with a unique `{_name}`.

## Adding Nuggets

Set `create_nuggets: true` and declare a `nugget_banks_type`:

```yaml
judge_class: "judges.myjudge:MyJudge"
create_nuggets: true
judge: true
```

```python
from autojudge_base import NuggetBanks, NuggetBanksProtocol

class MyJudge:
    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def create_nuggets(self, rag_responses, rag_topics, llm_config,
                       nugget_banks=None, **kwargs):
        banks = []
        for topic in rag_topics:
            bank = NuggetBank(query_id=topic.request_id, title_query=topic.title)
            bank.add_nuggets(generate_questions(topic, llm_config))
            banks.append(bank)
        return NuggetBanks.from_banks_list(banks)

    def judge(self, rag_responses, rag_topics, llm_config,
              nugget_banks=None, **kwargs):
        # Use nugget_banks.banks[topic_id] to score responses
        ...
```

The framework saves nuggets to `{filebase}.nuggets.jsonl` and passes them to `judge()`. The nugget format is determined by `nugget_banks_type` on the judge class (or `nugget_banks_type` in workflow.yml for workflow-level override).

Available formats:
- `NuggetBanks` — questions, claims, answers, references (default)
- `NuggetizerNuggetBanks` — simpler text-based nuggets
- Custom formats implementing `NuggetBanksProtocol`

### Nugget Bank Input

Use `--nugget-banks <path>` to supply a pre-existing nugget bank:

```bash
# Judge with externally-provided nuggets
auto-judge run --workflow workflow.yml \
    --nugget-banks my_nuggets.nuggets.jsonl ...

# Refine existing nuggets, then judge
auto-judge run --workflow workflow.yml \
    --nugget-banks seed.nuggets.jsonl --store-nuggets refined.nuggets.jsonl ...
```

**Resolution order:** (1) `--nugget-banks` CLI path, (2) auto-loaded from `{filebase}.nuggets.jsonl` if it exists from a previous run, (3) created fresh by `create_nuggets()`, (4) `None`.

When a nugget file already exists and `create_nuggets: true`, the framework loads it instead of recreating (saving LLM calls). Use `--force-recreate-nuggets` to override. Same applies to qrels with `--force-recreate-qrels`.

### Nugget/Qrels Path Overrides

Override default paths in workflow.yml:

```yaml
nugget_input: "shared_nuggets.nuggets.jsonl"     # Load existing nuggets
nugget_output: "my_nuggets.nuggets.jsonl"         # Override output path
qrels_input: "existing.qrels"                     # Load existing qrels
qrels_output: "my_output.qrels"                   # Override output path
```

These support template variables (e.g., `nugget_output: "{_name}.nuggets.jsonl"`).

## Adding Qrels

Enable a separate relevance-judgment phase between nuggets and leaderboard:

```yaml
create_nuggets: true
create_qrels: true
judge: true

qrels_settings:
  grade_range: [0, 3]
```

```python
def create_qrels(self, rag_responses, rag_topics, llm_config,
                 nugget_banks=None, **kwargs):
    # Return Qrels object with per-document relevance grades
    ...
```

Execution order: `create_nuggets()` -> `create_qrels(nuggets)` -> `judge(nuggets, qrels)`

## Lifecycle Flags

Fine-grained control over what data flows between phases:

| Flag | Default | Effect |
|------|---------|--------|
| `nugget_depends_on_responses` | `true` | If false, `create_nuggets()` receives `rag_responses=None` |
| `judge_uses_nuggets` | `true` | If false, `judge()` receives `nugget_banks=None` |
| `judge_uses_qrels` | `true` | If false, `judge()` receives `qrels=None` |
| `qrels_uses_nuggets` | `true` | If false, `create_qrels()` receives `nugget_banks=None` |
| `force_recreate_nuggets` | `false` | Recreate nuggets even if output file exists |
| `force_recreate_qrels` | `false` | Recreate qrels even if output file exists |
| `augment_report` | `false` | Save modified `Report.evaldata` to `{filebase}.responses.jsonl` |

All lifecycle flags can be overridden via CLI (e.g., `--no-judge-uses-nuggets`, `--augment-report`).

## Modular Protocol Classes

By default, `judge_class` handles all three phases. For separate implementations, use:

```yaml
nugget_class: "judges.my_nuggets:NuggetCreator"
qrels_class: "judges.my_qrels:QrelsCreator"
judge_class: "judges.my_judge:LeaderboardJudge"
```

Each class only needs to implement its respective protocol method. If `nugget_class` or `qrels_class` is omitted, `judge_class` is used as fallback (assuming it implements all protocols).

## CLI Reference

### Subcommands

```bash
auto-judge run      # Execute workflow.yml (default)
auto-judge judge    # Judge with existing nuggets (--nugget-banks required)
auto-judge nuggify  # Create nuggets only (--store-nuggets required)
```

### Key Options

**Data input:**
- `--rag-responses PATH` — responses file or directory
- `--rag-topics PATH` — topics JSONL file
- `--nugget-banks PATH` — input nugget banks (file or directory)
- `--llm-config PATH` — LLM configuration YAML

**Output:**
- `--out-dir PATH` — parent directory for all output files
- `--filebase STR` — override output filename template
- `--store-nuggets PATH` — override nugget output path

**Workflow overrides:**
- `--variant NAME` / `--all-variants` / `--sweep NAME`
- `--create-nuggets` / `--no-create-nuggets`
- `--create-qrels` / `--no-create-qrels`
- `--judge` / `--no-judge`
- `--force-recreate-nuggets` / `--force-recreate-qrels`
- `--augment-report` / `--no-augment-report`
- `--nugget-judge` / `--no-nugget-judge` (alias for `judge_uses_nuggets`)
- `--nugget-depends-on-responses` / `--no-nugget-depends-on-responses`
- `--judge-uses-qrels` / `--no-judge-uses-qrels`
- `--qrels-uses-nuggets` / `--no-qrels-uses-nuggets`

**Settings overrides (repeatable):**
- `-S KEY=VALUE` — shared settings
- `-N KEY=VALUE` — nugget settings
- `-J KEY=VALUE` — judge settings
- `-Q KEY=VALUE` — qrels settings

**Data filtering (for development):**
- `--topic ID` — specific topics (repeatable)
- `--run ID` — specific runs (repeatable)
- `--limit-topics N` / `--limit-runs N`

## Output Files

Given `filebase: "rubric"`, the framework produces:

| File | Condition |
|------|-----------|
| `rubric.nuggets.jsonl` | `create_nuggets: true` |
| `rubric.qrels` | `create_qrels: true` |
| `rubric.judgment.json` | `judge: true` |
| `rubric.config.yml` | always (captures reproducibility info) |
| `rubric.responses.jsonl` | `augment_report: true` |

The `.config.yml` records the full configuration (settings, git commit, timestamp, model) for reproducibility.

## Custom Nugget Formats

To create a custom format, implement `NuggetBanksProtocol`:

```python
from typing import ClassVar, Dict, List, Type
from pydantic import BaseModel

class MyNuggetBank(BaseModel):
    topic_id: str
    facts: List[str]

    @property
    def query_id(self) -> str:
        return self.topic_id

class MyNuggetBanks(BaseModel):
    _bank_model: ClassVar[Type[MyNuggetBank]] = MyNuggetBank
    banks: Dict[str, MyNuggetBank] = {}

    @classmethod
    def from_banks_list(cls, banks: List[MyNuggetBank], overwrite: bool = False):
        result = {}
        for bank in banks:
            if bank.query_id in result and not overwrite:
                raise ValueError(f"Duplicate: {bank.query_id}")
            result[bank.query_id] = bank
        return cls(banks=result)
```

Requirements:
- `NuggetBank` must have a `query_id` property
- `NuggetBanks` must have `_bank_model: ClassVar` pointing to the bank class
- `NuggetBanks` must have `banks: Dict[str, NuggetBank]` field
- `NuggetBanks` must have `from_banks_list(banks, overwrite=False)` classmethod

The framework handles serialization/deserialization automatically based on the declared type.