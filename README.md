# AutoJudge Base

Core infrastructure for implementing TREC AutoJudge systems.

## Installation

```bash
pip install autojudge-base
```

## Quick Start

```python
from autojudge_base import (
    AutoJudge,
    Report,
    Request,
    Leaderboard,
    LeaderboardBuilder,
    LlmConfigBase,
)

# Define your judge
class MyJudge:
    nugget_banks_type = None  # Or your NuggetBanks class

    def judge(self, rag_responses, rag_topics, llm_config, **kwargs):
        # Your judging logic
        builder = LeaderboardBuilder(...)
        return builder.build()

    def create_nuggets(self, rag_responses, rag_topics, llm_config, **kwargs):
        return None  # Or create nuggets

    def create_qrels(self, rag_responses, rag_topics, llm_config, **kwargs):
        return None  # Or create qrels
```

## Components

### Protocols

- `AutoJudge` - Combined protocol for all three phases (see [Quick Start](#quick-start))
  - Example: `NaiveJudge`, `RetrievalJudge` in starterkit
- `LeaderboardJudgeProtocol` - Produces leaderboard scores (see [Writing Leaderboards](#writing-leaderboards))
  - Example: `TinyJudge` (minimal LLM judge)
- `QrelsCreatorProtocol` - Creates relevance judgments (see [Writing Qrels](#writing-qrels))
- `NuggetCreatorProtocol` - Creates nugget banks (see [Writing NuggetBanks](#writing-nuggetbanks))

For modular composition (separate classes per protocol), see `CompleteExampleJudge` in starterkit.

### Input Data Models

- `Report` - RAG system output (see [Loading Reports](#loading-reports-rag-system-outputs))
- `Request` - Evaluation topic/query (see [Loading Requests](#loading-requests-topicsqueries))
- `Document` - Document with content (see [Loading Documents](#loading-documents-background-corpus))

### Output Containers

- `Leaderboard` - System rankings (see [Writing Leaderboards](#writing-leaderboards))
- `Qrels` - Relevance judgments (see [Writing Qrels](#writing-qrels))
- `NuggetBanks`, `NuggetizerNuggetBanks` - Nugget collections (see [Writing NuggetBanks](#writing-nuggetbanks))

### Configuration

- `LlmConfigProtocol`, `LlmConfigBase` - LLM configuration
- `load_llm_config()` - Load config from env/yaml/cli

See the [auto-judge-starterkit](https://github.com/trec-auto-judge/auto-judge-starterkit) README for LLM configuration examples.

## CLI

The `trec-auto-judge` CLI (provided by the `trec_auto_judge` package) uses autojudge-base:

```bash
# Run a judge workflow
trec-auto-judge run --workflow workflow.yml --rag-responses responses.jsonl

# Export corpus
trec-auto-judge export-corpus --output corpus.tar.gz

# List available models
trec-auto-judge list-models
```

See the [Workflow Guide](src/autojudge_base/workflow/README.md) for details.

## Data Loading Utilities

### Loading Reports (RAG System Outputs)

A `Report` contains sentences with text and citations. Three sentence formats are supported:

| Format | Citations Field | Description |
|--------|----------------|-------------|
| `NeuclirReportSentence` | `List[str]` | Doc IDs ordered by priority |
| `RagtimeReportSentence` | `Dict[str, float]` | Doc ID → confidence score (0-100) |
| `Rag24ReportSentence` | `List[int]` | Indices into `report.references` |

**Getting text and citations from sentences:**

```python
from autojudge_base.report import Report, load_report

reports: list[Report] = load_report(Path("responses.jsonl"))

for report in reports:
    # Get sentences with citations in unified format (does not modify report)
    for sentence in report.get_sentences_with_citations():
        text = sentence.text                    # The sentence text
        citations = sentence.citations or []    # Doc IDs ordered by priority

        # Get the cited document content
        for doc_id in citations:
            if report.documents and doc_id in report.documents:
                doc = report.documents[doc_id]
                print(f"Citation: {doc.title} - {doc.text[:100]}...")
```

**Convenience methods:**

```python
# Text only (no citations)
texts: list[str] = report.get_sentences()

# Text with citations (unified format, non-mutating)
sentences: List[NeuclirReportSentence] = report.get_sentences_with_citations()  

# Full response as single string
test: str = report.get_report_text()  

# Full text of cited documents (keyed by doc_id)
documents: Dict[str, Document] = report.documents
```

**Report metadata and convenience methods:**

```python
for report in reports:
    print(report.metadata.run_id)      # Which system produced this
    print(report.metadata.topic_id)    # Which topic/query this answers
```

Note that formats of different TREC tasks differ slightly. This module will automatically load any and expose it as this one format. Task specific fields, such as `narrative_id` vs `request_id` are also available.

### Loading Requests (Topics/Queries)

A `Request` represents an evaluation topic with the query and context:

```python
from autojudge_base.request import Request, load_requests_from_file, load_requests_from_irds

# Load from JSONL file
requests: list[Request] = load_requests_from_file(Path("topics.jsonl"))

# Load from ir_datasets
requests: list[Request] = load_requests_from_irds("trec-rag-2025")

# Access request fields
for req in requests:
    print(req.request_id)         # Unique topic identifier
    print(req.title)              # The query/question (required)
    print(req.problem_statement)  # Detailed description of the information need 
    print(req.background)         # User background/context for personalization
```

RAG narratives were converted to this format, exposing narratives in the `problem_statement` field.

### Loading Documents (Background Corpus)

Use this when you need to fetch additional documents from a background corpus beyond what's included in reports.

```python
from autojudge_base.document import Document, load_documents, RetrievedDocuments, load_retrieved_docs

# Load corpus documents from JSONL
docs: list[Document] = load_documents(Path("corpus.jsonl"))

# Access document content
for doc in docs:
    print(doc.id)          # Document identifier
    print(doc.title)       # Document title (optional)
    print(doc.text)        # Document content
    print(doc.get_text())  # Title + text combined

# Load pre-retrieved documents with rankings
retrieved: list[RetrievedDocuments] = load_retrieved_docs(Path("retrieved.jsonl"))
for result in retrieved:
    print(result.query_id)
    for ranked_doc in result.ranked_docs:
        print(f"  Rank {ranked_doc.rank}: {ranked_doc.doc.id} (score: {ranked_doc.score})")
```

### Writing Leaderboards

Use `LeaderboardBuilder` with a `LeaderboardSpec` to create type-safe leaderboards:

```python
from autojudge_base import LeaderboardBuilder, LeaderboardSpec, MeasureSpec

# Define your measures
spec = LeaderboardSpec(measures=(
    MeasureSpec("RELEVANCE"),           # Default: mean aggregation, float values
    MeasureSpec("FLUENCY"),
    MeasureSpec("CITATION_QUALITY", aggregate=sum),  # Custom aggregation
))

# Build the leaderboard
builder = LeaderboardBuilder(spec)

for report in reports:
    builder.add(
        run_id=report.metadata.run_id,
        topic_id=report.metadata.topic_id,
        values={
            "RELEVANCE": 0.85,
            "FLUENCY": 0.92,
            "CITATION_QUALITY": 3,
        }
    )

# Finalize with expected topics (handles missing data)
leaderboard = builder.build(
    expected_topic_ids=["topic1", "topic2", "topic3"],
    on_missing="fix_aggregate"  # or "error", "warn", "ignore"
)

```

**In your judge:**

```python
def judge(self, rag_responses, rag_topics, llm_config, **kwargs) -> Leaderboard:
    # ... build leaderboard as above ...
    return leaderboard
```

**Manual file I/O (FYI):** Judge implementations return objects; the framework handles persistence.

```python
# Write (formats: "trec_eval", "tot", "ir_measures", "jsonl")
leaderboard.write(Path("output.eval"), format="trec_eval")

# Load
leaderboard = Leaderboard.load(Path("output.eval"), format="trec_eval")
```

### Writing NuggetBanks

NuggetBanks store evaluation nuggets (questions/claims) per topic. Two formats are supported:

| Format | Configure in workflow.yml | Description |
|--------|--------------------------|-------------|
| `NuggetBanks` | `nugget_banks_type: "autojudge_base.nugget_data.NuggetBanks"` | AutoArgue format with questions and claims |
| `NuggetizerNuggetBanks` | `nugget_banks_type: "autojudge_base.nugget_data.NuggetizerNuggetBanks"` | Nuggetizer format |

```python
from autojudge_base.nugget_data import (
    NuggetBanks,
    NuggetBank,
    NuggetQuestion,
    load_nugget_banks_from_file,
    write_nugget_banks,
)

# Load existing nuggets
nuggets: NuggetBanks = load_nugget_banks_from_file(Path("nuggets.jsonl"))

# Access by topic
bank = nuggets.banks["topic-1"]
for question in bank.nuggets_as_list():
    print(question.question)
    print(question.gold_answers)

# Create new nuggets
bank = NuggetBank(query_id="topic-1")
bank.add_nuggets([
    NuggetQuestion.from_lazy(
        query_id="topic-1",
        question="What is the capital of France?",
        gold_answers=["Paris"],
        references=["doc123"],
        creator="my-judge",
    )
])

nuggets = NuggetBanks.from_banks_list([bank])
```

**In your judge:**

```python
nugget_banks_type = NuggetBanks  # Required class attribute

def create_nuggets(self, rag_responses, rag_topics, llm_config, **kwargs) -> NuggetBanks:
    # ... create nugget banks as above ...
    return nuggets
```

**Manual file I/O (FYI):** Judge implementations return objects; the framework handles persistence.

```python
# Write
write_nugget_banks(nuggets, Path("nuggets.jsonl"))

# Load
nuggets = load_nugget_banks_from_file(Path("nuggets.jsonl"))
```

### Writing Qrels

Qrels store relevance judgments as (topic_id, doc_id, grade) tuples.

```python
from autojudge_base.qrels import Qrels, QrelRow, QrelsSpec, build_qrels, write_qrel_file

# Option 1: Build directly from QrelRow objects
rows = [
    QrelRow(topic_id="topic1", doc_id="doc123", grade=2),
    QrelRow(topic_id="topic1", doc_id="doc456", grade=1),
    QrelRow(topic_id="topic2", doc_id="doc789", grade=3),
]
qrels = Qrels(rows=rows)

# Option 2: Build from arbitrary records using QrelsSpec
@dataclass
class MyJudgment:
    query: str
    document: str
    relevance: int

judgments = [
    MyJudgment("topic1", "doc123", 2),
    MyJudgment("topic1", "doc456", 1),
]

spec = QrelsSpec(
    topic_id=lambda j: j.query,
    doc_id=lambda j: j.document,
    grade=lambda j: j.relevance,
    on_duplicate="keep_max",  # or "error", "keep_last"
)
qrels = build_qrels(records=judgments, spec=spec)

```

**In your judge:**

```python
def create_qrels(self, rag_responses, rag_topics, llm_config, **kwargs) -> Qrels:
    # ... build qrels as above ...
    return qrels
```

**Manual file I/O (FYI):** Judge implementations return objects; the framework handles persistence.

```python
# Write to TREC format (topic_id  0  doc_id  grade)
write_qrel_file(qrel_out_file=Path("output.qrels"), qrels=qrels)

# Load from TREC format
qrels = read_qrel_file(Path("input.qrels"))
```

## License

MIT
