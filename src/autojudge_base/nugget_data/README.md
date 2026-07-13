# NuggetBank (v3) — Getting Started

- contact: Laura Dietz <dietz@cs.unh.edu>

A **nugget bank** holds, for one topic, the things a good answer should contain. Each *nugget* is either a **question** the answer ought to address (optionally with gold answers) or a **claim** the answer ought to state. A judge builds one nugget bank per topic, then grades each RAG response by how many of the topic's nuggets it covers.

This page builds up a nugget bank for one topic — *"Rising Demand for Avocado"* — starting from the smallest useful form and adding detail step by step.

## 1. The smallest useful bank: a topic and some questions

The minimal nugget is just a question. Create a `NuggetBank` for the topic and add a few:

```python
from autojudge_base.nugget_data import NuggetBank, NuggetQuestion

bank = NuggetBank(query_id="1053", title_query="Rising Demand for Avocado")

bank.add_nuggets([
    NuggetQuestion.from_lazy(query_id="1053",
        question="What percentage of U.S. avocado imports is supplied by Mexico?"),
    NuggetQuestion.from_lazy(query_id="1053",
        question="How many liters of water are required to produce one kilogram of avocado?"),
    NuggetQuestion.from_lazy(query_id="1053",
        question="What are the main environmental impacts of avocado farming?"),
])
```

`from_lazy` is the convenient constructor — hand it plain strings and it fills in the rest (the `question_id` is a hash of the question text, so the same question always gets the same id). Serialized, the nugget bank looks like this:

```json
{
  "query_id": "1053",
  "title_query": "Rising Demand for Avocado",
  "format_version": "v3",
  "nugget_bank": {
    "What percentage of U.S. avocado imports is supplied by Mexico?": {
      "question": "What percentage of U.S. avocado imports is supplied by Mexico?",
      "question_id": "c9391413e0dbae07eb36b484dcbcab2a",
      "query_id": "1053"
    },
    "How many liters of water are required to produce one kilogram of avocado?": {
      "question": "How many liters of water are required to produce one kilogram of avocado?",
      "question_id": "0aa940f1a0e1eec06b67f46719e1201f",
      "query_id": "1053"
    }
  }
}
```

Inspect a nugget bank at any point with `print_nugget_json(bank)`.

## 2. Add gold answers

A question becomes far more useful for grading when it carries the answer a correct response should give. Pass `gold_answers` — one string, or several acceptable phrasings:

```python
NuggetQuestion.from_lazy(query_id="1053",
    question="What percentage of U.S. avocado imports is supplied by Mexico?",
    gold_answers=["about 90%", "roughly 90 percent"])
```

## 3. Point at the documents that support an answer

`references` records which corpus documents back a nugget — a doc id is enough to start:

```python
NuggetQuestion.from_lazy(query_id="1053",
    question="How many liters of water are required to produce one kilogram of avocado?",
    gold_answers="around 2000 liters",
    references=["doc-042"])
```

For a precise citation, use a `Reference` with character offsets instead of a bare id:

```python
from autojudge_base.nugget_data import Reference, Offsets

Reference(doc_id="doc-042", collection="avocado-corpus",
          text="Producing one kilogram of avocados takes roughly 2,000 litres of water.",
          offsets=Offsets(start_offset=0, end_offset=70, encoding="utf-8"))
```

## 4. Claims — fact-style nuggets

When the thing a good answer should contain is a statement rather than a question, use a `NuggetClaim`. Claims live in the same bank, alongside questions:

```python
from autojudge_base.nugget_data import NuggetClaim

bank.add_nuggets(
    NuggetClaim.from_lazy(query_id="1053",
        claim="Mexico is the world's largest avocado exporter.",
        references=["doc-108"]))
```

## 5. Record who created the nuggets

`Creator` documents the provenance of a nugget bank — human assessment or LLM generation — which matters for meta-evaluation:

```python
from autojudge_base.nugget_data import Creator

# Human-authored
bank.add_creator(Creator(is_human=True, contact=["NIST", "assessor-7"]))

# LLM-generated
bank.add_creator(Creator(is_human=False, llm_model="gpt-4o",
                         llm_prompt_strategy="query-only"))
```

## 6. From one topic to many

Your judge produces a nugget bank per topic and returns them together as a `NuggetBanks`, keyed by topic id:

```python
from autojudge_base.nugget_data import NuggetBanks

banks = NuggetBanks.from_banks_list([bank])   # add one nugget bank per topic
```

For reading and writing nugget files, use the I/O helpers (`make_io_functions`, `load_nugget_banks_from_file`); the [develop-an-autojudge](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/03-develop-an-autojudge.md#creating-nuggets) howto shows where this fits in a judge.

---

## Verification

Verify a `NuggetBanks` against the topics you expected to cover — this turns silent gaps (a missing topic, an empty bank) into a loud error instead of a wrong-but-plausible evaluation. The workflow runner does this automatically before judging; call it yourself in tests and scripts.

### Quick check with `.verify()`

```python
topic_ids = [t.request_id for t in topics]
banks.verify(expected_topic_ids=topic_ids, warn=False)  # warn=True prints instead of raising
```

### Granular checks with `NuggetBanksVerification`

```python
from autojudge_base.nugget_data import NuggetBanksVerification

NuggetBanksVerification(banks, expected_topic_ids=topic_ids, warn=False) \
    .complete_topics().no_extra_topics().non_empty_banks()

# or run every check at once:
NuggetBanksVerification(banks, expected_topic_ids=topic_ids).all()
```

| Method | Checks that |
|--------|-------------|
| `complete_topics()` | every expected topic has a nugget bank |
| `no_extra_topics()` | no bank exists for an unexpected topic |
| `non_empty_banks()` | each nugget bank has at least one nugget |
| `all()` | all of the above |
