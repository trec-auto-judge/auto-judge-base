# AutoJudge Base

Core infrastructure for implementing TREC AutoJudge systems.

## Installation

```bash
pip install autojudge-base
```

For full LLM features (batching, advanced transport):
```bash
pip install autojudge-base[minima-llm]
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

- `AutoJudge` - Combined protocol for all three phases
- `LeaderboardJudgeProtocol` - Produces leaderboard scores
- `QrelsCreatorProtocol` - Creates relevance judgments
- `NuggetCreatorProtocol` - Creates nugget banks

### Data Models

- `Report` - RAG system output
- `Request` - Evaluation topic/query
- `Document` - Document with content

### Output Containers

- `Leaderboard` - System rankings with scores
- `Qrels` - Relevance judgments
- `NuggetBanks` - Nugget collections per topic

### Configuration

- `LlmConfigBase` - Minimal LLM configuration
- `LlmConfigProtocol` - Protocol for LLM configs
- `load_llm_config()` - Load config from env/yaml/cli

## CLI

```bash
# Run a judge workflow
auto-judge run --workflow workflow.yml --rag-responses responses.jsonl

# Export corpus
auto-judge export-corpus --output corpus.tar.gz

# List available models
auto-judge list-models
```

## Data Loading Utilities

### Loading Reports (RAG System Outputs)

```python
from autojudge_base.report import Report, load_report

# Load all reports from a directory or JSONL file
reports: list[Report] = load_report(Path("responses/"))

# Access report data
for report in reports:
    print(report.metadata.run_id)
    print(report.metadata.topic_id)
    print(report.get_report_text())
```

### Loading Requests (Topics/Queries)

```python
from autojudge_base.request import Request, load_requests_from_file, load_requests_from_irds

# Load from JSONL file
requests: list[Request] = load_requests_from_file(Path("topics.jsonl"))

# Load from ir_datasets
requests: list[Request] = load_requests_from_irds("trec-rag-2025")

# Access request data
for req in requests:
    print(req.request_id)
    print(req.title)
```

### Loading Documents

```python
from autojudge_base.document import Document, load_documents, load_retrieved_docs

# Load corpus documents
docs: list[Document] = load_documents(Path("corpus.jsonl"))

# Load retrieved documents (with rankings)
retrieved: list[RetrievedDocuments] = load_retrieved_docs(Path("retrieved.jsonl"))
```

### Loading Nugget Banks

```python
from autojudge_base.nugget_data import (
    NuggetBanks,
    load_nugget_banks_from_file,
)

# Load nugget banks from JSONL
nuggets: NuggetBanks = load_nugget_banks_from_file(Path("nuggets.jsonl"))

# Access by topic
bank = nuggets.banks["topic-1"]
for question in bank.nuggets_as_list():
    print(question.question)
```

### Writing Qrels

```python
from autojudge_base.qrels import Qrels, write_qrel_file

# Write qrels to TREC format file
write_qrel_file(qrels, Path("output.qrels"))
```

## License

MIT
