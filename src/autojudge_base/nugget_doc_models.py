"""Nugget-document models for export and evaluation.

Provides:
- NuggetDocEntry: A nugget question with its relevant document IDs
- TopicNuggetDocs: All nugget-document mappings for a single topic
- write_nugget_docs_collaborator: Write collaborator-format JSON files
"""

import json
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel


class NuggetDocEntry(BaseModel):
    """A nugget question with its relevant document IDs."""

    question: str
    doc_ids: List[str]
    aggregator: str = "OR"
    answer_type: str = "OPEN_ENDED_ANSWER"

    def to_collaborator_value(self) -> list:
        """Serialize to collaborator format: ["OR", {"OPEN_ENDED_ANSWER": [...]}]"""
        return [self.aggregator, {self.answer_type: self.doc_ids}]


class TopicNuggetDocs(BaseModel):
    """All nugget-document mappings for a single topic."""

    topic_id: str
    entries: List[NuggetDocEntry]

    def to_collaborator_dict(self) -> dict:
        """Serialize to collaborator format: {question: ["OR", {...}], ...}"""
        return {e.question: e.to_collaborator_value() for e in self.entries}


def write_nugget_docs_collaborator(
    topics: Dict[str, TopicNuggetDocs],
    output_dir: Path,
) -> None:
    """Write one nuggets_{topic_id}.json per topic in collaborator format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for topic_id, topic in topics.items():
        path = output_dir / f"nuggets_{topic_id}.json"
        path.write_text(json.dumps(topic.to_collaborator_dict(), indent=4))
    print(f"Wrote {len(topics)} collaborator nugget files to {output_dir}")
