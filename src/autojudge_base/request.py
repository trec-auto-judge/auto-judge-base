from enum import Enum
import gzip
from typing import Any, Dict, Iterable, List, Optional, Set, TextIO, Union, TypeAlias
from io import StringIO
from pathlib import Path
import json
from pydantic import BaseModel


class Request(BaseModel):
    request_id:str
    collection_ids:Optional[List[str]]= None
    background:Optional[str] = None
    original_background:Optional[str] = None
    problem_statement:Optional[str] = None
    limit:Optional[int] = None
    word_limit:Optional[int] = None
    title:str

# Todo @mam10eks:  check bug @26

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
    with open(file) as f:
        for l in f:
            parsed = json.loads(l)
            request = Request.model_validate(parsed)
            ret.append(request)
    return ret