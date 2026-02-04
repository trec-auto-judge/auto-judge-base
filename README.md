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

## License

MIT
