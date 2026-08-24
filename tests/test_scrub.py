"""Tests for the ``scrub`` content-blind transform.

Run: PYTHONPATH=src pytest tests/test_scrub.py
"""
import dataclasses
import json

import pytest

from autojudge_base.scrub import (TIER1_STRING, ScrubStats, is_schema_key,
                                  scrub_json_line, scrub_model, scrub_string,
                                  scrub_value)


def test_tier1_keeps_structure_and_destroys_text_and_length():
    out = scrub_value({"a": "secret", "n": [1, 2], "score": 4.5, "ok": True,
                       "none": None}, chars=False)
    assert out == {"a": TIER1_STRING, "n": [1, 2], "score": 4.5, "ok": True,
                   "none": None}
    assert scrub_string("x", False) == scrub_string("x" * 5000, False)


@pytest.mark.parametrize("text", [
    "café", "第二段落です。", "한국어 텍스트", "ที่นี่", "مرحبا 42", "Привет",
    "msmarco_v2.1_doc_51_8", "€4.50 — [1]", "emoji 🙂 stays", "  \t ws \n",
])
def test_tier2_preserves_byte_and_character_length(text):
    """Length is what tier 2 exists to preserve.

    Regression: decomposing Hangul explodes syllables into conjoining Jamo, and
    Thai combining marks never recompose, so an earlier version tripled Korean
    and shrank Thai.
    """
    out = scrub_string(text, chars=True)
    assert len(out) == len(text)
    assert len(out.encode("utf-8")) == len(text.encode("utf-8"))


def test_tier2_substitutes_alphanumerics_and_keeps_everything_else():
    assert scrub_string('He said: "no" — [1]\n\t.', chars=True) == \
        'aa aaaa: "aa" — [1]\n\t.'
    assert scrub_string("café", chars=True) == "aaaá"      # accent survives


def test_schema_keys_survive_content_keys_and_identifiers_do_not():
    assert is_schema_key("run_id") and not is_schema_key("Paris is the capital")
    out = scrub_value({"run_id": "plum", "Paris is the capital": 1}, chars=False)
    assert out["run_id"] == TIER1_STRING          # identifiers are values, scrubbed
    assert "Paris is the capital" not in out


def test_malformed_json_is_preserved_not_repaired():
    tier1 = json.loads(scrub_json_line('{"a": "secret", "b": }', chars=False))
    assert tier1["__scrub_parse_error__"]["lineno"] == 1
    assert "secret" not in json.dumps(tier1)
    # tier 2 keeps the delimiters and quoting that caused the failure
    assert scrub_json_line('{"a": "secret", "b": }', chars=True) == \
        '{"a": "aaaaaa", "a": }'
    # a wrong type stays a wrong type: that is the reproducer
    assert isinstance(scrub_value({"citations": "not-a-list"}, False)["citations"], str)


def test_stats_are_counts_never_values():
    stats = ScrubStats()
    scrub_json_line('{"a": "xx", "Some sentence key": ["y"]}', True, stats)
    d = stats.as_dict()
    assert d["records"] == 1 and d["keys_scrubbed"] == 1
    assert all(isinstance(v, int) for v in d.values())


def test_scrub_model_round_trips_pydantic_and_dataclasses():
    from autojudge_base.request import Request
    req = scrub_model(Request(request_id="t1", title="Vaping levy"), chars=False)
    assert isinstance(req, Request) and req.title == TIER1_STRING

    @dataclasses.dataclass(frozen=True)
    class Row:
        run_id: str
        score: float

    row = scrub_model(Row("plum", 0.5), chars=False)
    assert isinstance(row, Row)                    # frozen dataclass survives
    assert row.run_id == TIER1_STRING and row.score == 0.5
