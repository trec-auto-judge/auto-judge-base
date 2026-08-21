from enum import Enum
import gzip
from typing import Any, Dict, Iterable, List, Optional, Set, TextIO, Union, TypeAlias
from io import StringIO
from pathlib import Path
import json
from pydantic import BaseModel


class Request(BaseModel):
    """A topic/request. `request_id` is the canonical identifier.

    Topics files must supply `request_id` -- to be compatible with exisiting auto-judge implementations

    `topic_id` and `narrative_id` are mirroring request_id
    """
    request_id:str
    title:str
    collection_ids:Optional[List[str]]= None
    background:Optional[str] = None
    original_background:Optional[str] = None
    problem_statement:Optional[str] = None
    limit:Optional[int] = None
    word_limit:Optional[int] = None

    # Mirrors of request_id, set in model_post_init (never read as input)
    topic_id:Optional[str] = None
    narrative_id:Optional[str] = None

    def model_post_init(self, __context__: dict | None = None) -> None:
        for name in ("topic_id", "narrative_id"):
            given = getattr(self, name)
            if given is not None and str(given) != str(self.request_id):
                raise ValueError(
                    f"Inconsistent topic identifiers: "
                    f"request_id={self.request_id}, {name}={given}"
                )
            setattr(self, name, self.request_id)


def load_requests_from_irds(ir_dataset)->List[Request]:
    ret = list()

    collection_id = ir_dataset.dataset_id()
    for topic in ir_dataset.queries_iter():
        # ToDo better mapping
        r = {"title": topic.default_text(), "request_id": topic.query_id, "collection_ids": [collection_id]}
        for optional_field in ["background", "original_background", "problem_statement", "limit", "word_limit"]:
            if hasattr(r, optional_field):
                r[optional_field] = getattr(topic, optional_field)
        r_parsed = Request.model_validate(r)
        ret.append(r_parsed)

    return ret

def load_requests_from_file(file: Path)->List[Request]:
    ret = list()
    with open(file, encoding="utf-8") as f:
        for l in f:
            parsed = json.loads(l)
            request = Request.model_validate(parsed)
            ret.append(request)
    return ret