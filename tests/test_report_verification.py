"""Tests for the collection-level verifier (autojudge_base.report_verification) and the
emit layer (autojudge_base.report_export). ReportVerification mirrors LeaderboardVerification:
fluent per-report + cross-report checks, findings split into errors vs smell-warnings.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autojudge_base.report_export import report_to_submission_dict, write_submission_output
from autojudge_base.report import Rag24ReportSentence, Report, ReportMetaData, load_report
from autojudge_base.report_verification import (
    ReportVerification, ReportVerificationError, compress_ids, fmt_ids,
)
from autojudge_base.track_spec import SPECS

SHARD, SHARD2 = "shard_00459_61697", "shard_01234_5678"


def _rag26_report(topic="1", refs=None, sents=None):
    md = ReportMetaData(team_id="T", narrative_id=topic, narrative="q text",
                        run_id="run", run_desc="desc")
    if sents is None:
        sents = [Rag24ReportSentence(text="First.", citations=[0]),
                 Rag24ReportSentence(text="Second.", citations=[1])]
    return Report(metadata=md, references=refs if refs is not None else [SHARD, SHARD2],
                  responses=sents)


# --- pure helpers ---------------------------------------------------------------

def test_id_helpers():
    assert compress_ids(["1001", "1002", "1003", "1005"]) == "1001-1003, 1005"
    assert fmt_ids(["a", "b"]) == "a, b"       # non-integer ids -> plain list


# --- spec_compliant (per-report rules) ------------------------------------------

def test_spec_compliant_clean():
    v = ReportVerification([_rag26_report()], "rag26").spec_compliant()
    assert v.errors == [] and v.ok == 1


def test_spec_compliant_errors_and_smells():
    err = _rag26_report(refs=["not_a_shard_id"],
                        sents=[Rag24ReportSentence(text="Only.", citations=[0])])  # bad docid -> hard
    smell = _rag26_report(topic="2", refs=[], sents=[])                            # empty answer
    v = ReportVerification([err, smell], "rag26").spec_compliant()
    assert any("docid" in f.message for f in v.errors)
    assert any("empty answer" in f.message for f in v.warnings)
    assert v.ok == 1                          # only the empty (smell-only) report has no hard error


def test_strict_folds_smells_into_errors():
    smell = _rag26_report(refs=[], sents=[])
    v = ReportVerification([smell], "rag26", strict=True).spec_compliant()
    assert v.warnings == [] and any("empty answer" in f.message for f in v.errors)


def test_finding_records_offending_report(tmp_path: Path):
    obj = {"metadata": {"team_id": "TA", "narrative_id": "9", "narrative": "n",
                        "run_id": "RA", "run_desc": "d"},
           "references": [SHARD], "answer": [{"text": "x", "citations": [5]}]}  # bad index
    p = tmp_path / "runA"
    p.write_text(json.dumps(obj) + "\n", encoding="utf-8")
    v = ReportVerification(load_report(p), "rag26").spec_compliant()
    f = v.errors[0]
    assert f.category == "spec" and f.topic == "9"
    path, team, run = f.origin
    assert team == "TA" and run == "RA" and "runA" in path


# --- cross-report coverage ------------------------------------------------------

def test_coverage_missing_new_dup():
    reports = [_rag26_report("1"), _rag26_report("2"), _rag26_report("2"), _rag26_report("9")]
    # titles match the report narrative ("q text") so the exact-narrative check stays quiet
    requests = {t: SimpleNamespace(request_id=t, title="q text") for t in ("1", "2", "3")}
    v = ReportVerification(reports, "rag26", requests=requests).coverage()
    by_cat = {f.category: f.severity for f in v.findings}
    assert by_cat["missing_topics"] == "warning"      # rag26: absent topics accepted -> smell
    assert by_cat["new_topics"] == "error"            # topic 9 extra
    assert by_cat["duplicate_topics"] == "error"      # exactly one report per narrative


def test_coverage_all_present_no_findings():
    reports = [_rag26_report("1"), _rag26_report("2")]
    requests = {t: SimpleNamespace(request_id=t, title="q text") for t in ("1", "2")}
    v = ReportVerification(reports, "rag26", requests=requests).coverage()
    assert v.findings == []


# --- fluent all() + fail-fast raise_first() -------------------------------------

def test_all_then_raise_first_raises_on_error():
    err = _rag26_report(refs=["not_a_shard_id"],
                        sents=[Rag24ReportSentence(text="Only.", citations=[0])])
    with pytest.raises(ReportVerificationError, match="docid"):
        ReportVerification([err], "rag26").all().raise_first()


def test_raise_first_ignores_smells():
    smell = _rag26_report(refs=[], sents=[])            # empty answer -> smell only
    v = ReportVerification([smell], "rag26").all()
    assert v.raise_first() is v                          # no hard error -> no raise


# --- export ---------------------------------------------------------------------

def test_report_to_submission_dict_shape():
    obj = report_to_submission_dict(_rag26_report(), "rag26")
    assert set(obj.keys()) == {"metadata", "references", "answer"}
    assert obj["answer"][0]["citations"] == [0]


def test_report_to_submission_dict_converts_to_ragtime():
    obj = report_to_submission_dict(_rag26_report(), "ragtime26")
    assert "responses" in obj and "answer" not in obj
    assert isinstance(obj["responses"][0]["citations"], dict)   # doc_id -> score


def test_write_submission_output(tmp_path: Path):
    out = tmp_path / "sub.jsonl"
    n = write_submission_output([_rag26_report("1"), _rag26_report("2")], out, "rag26")
    assert n == 2
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["metadata"]["narrative_id"] == "1"
