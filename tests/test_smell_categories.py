"""The smell-category matrix (#3): every finding category fires, and every one demotes.

For EACH category in report_spec_verification (per-report) and report_verification
(cross-report), one minimal offending case proves:

  1. the category fires as a hard ERROR when the spec's ``smells`` does not list it;
  2. listing the category in ``smells`` demotes exactly that finding to a WARNING.

This is the coverage contract for the ``smells:`` configuration surface in
track_specs.yml -- a category passes through this matrix or it does not exist.
``CASES`` maps category -> (spec, offending report); the two parametrized tests below
run every entry both ways.
"""
from dataclasses import replace

import pytest

from autojudge_base.report import (
    NeuclirReportSentence,
    Rag24ReportSentence,
    RagtimeReportSentence,
    Report,
    ReportMetaData,
)
from autojudge_base.report_spec_verification import findings
from autojudge_base.report_verification import ReportVerification
from autojudge_base.track_spec import get_spec

SHARD, SHARD2 = "shard_00459_61697", "shard_01234_5678"
UUID = "b6a21af8-9cc4-462d-9c70-00bb9f009401_56341480"
UUID2 = "042ce256-aaa2-4944-8725-7deb68b8b43f_125182681"

RAG26 = get_spec("rag26")
RAGTIME = get_spec("ragtime25")
DRAGUN = get_spec("dragun25")


def _rag26(sents=None, refs=None, **md_over):
    md = dict(team_id="T", narrative_id="1", narrative="q text",
              run_id="run", run_desc="desc")
    md.update(md_over)
    if sents is None:
        sents = [Rag24ReportSentence(text="A sentence.", citations=[0])]
    if refs is None:
        refs = [SHARD]
    return Report(metadata=ReportMetaData(**md), responses=sents, references=refs)


def _ragtime(sents, refs):
    md = ReportMetaData(team_id="T", topic_id="300", run_id="run", task="multilingual")
    return Report(metadata=md, responses=sents, references=refs)


#: category -> (spec, offending report). One minimal trigger per category.
CASES = {
    "empty_answer": (RAG26, _rag26(sents=[])),
    "blank_sentence": (RAG26, _rag26(
        sents=[Rag24ReportSentence(text="ok.", citations=[0]),
               Rag24ReportSentence(text="   ", citations=[0])])),
    "citation_format": (RAG26, _rag26(
        sents=[RagtimeReportSentence(text="x.", citations={SHARD: 1.0})])),
    "citation_count": (replace(RAG26, max_citations_per_sentence=1), _rag26(
        sents=[Rag24ReportSentence(text="x.", citations=[0, 0])])),
    "reference_index": (RAG26, _rag26(
        sents=[Rag24ReportSentence(text="x.", citations=[0, 7])])),
    "citation_score": (RAGTIME, _ragtime(
        [RagtimeReportSentence.model_construct(text="x.", citations={UUID: "high"})],
        [UUID])),
    "docid_pattern": (RAG26, _rag26(
        sents=[Rag24ReportSentence(text="x.", citations=[0])], refs=["not_a_shard"])),
    "references_present": (DRAGUN, Report(
        metadata=ReportMetaData(team_id="T", run_id="r", topic_id="t",
                                type="automatic", use_starter_kit=0),
        responses=[NeuclirReportSentence(text="x.", citations=[])],
        references=[SHARD])),
    "references_missing": (RAG26, _rag26(
        sents=[Rag24ReportSentence(text="x.", citations=[])])
        .model_copy(update={"references": None})),
    "references_duplicate": (RAG26, _rag26(
        sents=[Rag24ReportSentence(text="x.", citations=[0])], refs=[SHARD, SHARD])),
    "references_undeclared": (RAGTIME, _ragtime(
        [RagtimeReportSentence(text="x.", citations={UUID: 1.0, UUID2: 1.0})], [UUID])),
    "references_uncited": (RAGTIME, _ragtime(
        [RagtimeReportSentence(text="x.", citations={UUID: 1.0})], [UUID, UUID2])),
    "references_max": (replace(RAG26, references_max=1), _rag26(
        sents=[Rag24ReportSentence(text="x.", citations=[0]),
               Rag24ReportSentence(text="y.", citations=[1])],
        refs=[SHARD, SHARD2])),
    "metadata": (RAG26, _rag26(team_id="")),
    "metadata_recommended": (replace(RAGTIME, recommended_metadata=("run_desc",)),
                             _ragtime([RagtimeReportSentence(text="x.",
                                                             citations={UUID: 1.0})],
                                      [UUID])),
    "run_id_length": (replace(RAG26, run_id_max_len=2), _rag26()),
    "length": (replace(RAG26, length_limit=1), _rag26(
        sents=[Rag24ReportSentence(text="five words are too many.", citations=[0])])),
}


@pytest.mark.parametrize("category", sorted(CASES))
def test_category_fires_as_error_by_default(category):
    spec, report = CASES[category]
    errors, warnings = findings(report, replace(spec, smells=()))
    assert errors, f"{category}: expected a hard error, got none"
    assert not warnings


@pytest.mark.parametrize("category", sorted(CASES))
def test_category_demotes_to_warning_when_listed_as_smell(category):
    spec, report = CASES[category]
    errors, warnings = findings(report, replace(spec, smells=(category,)))
    assert warnings, f"{category}: expected the finding as a warning, got none"
    assert not errors, f"{category}: smell listing did not demote: {errors}"


# --- cross-report categories (report_verification.ReportVerification) -----------

def _submission(topics):
    return [_rag26(narrative_id=t, narrative=f"topic {t}") for t in topics]


def _requests(topics):
    from types import SimpleNamespace

    return {t: SimpleNamespace(title=f"topic {t}", limit=None) for t in topics}


COVERAGE_CASES = {
    "missing_topics": (_submission(["1"]), _requests(["1", "2"])),
    "new_topics": (_submission(["1", "9"]), _requests(["1"])),
    "duplicate_topics": (_submission(["1", "1"]), _requests(["1"])),
    "narrative": ([_rag26(narrative_id="1", narrative="WRONG")], _requests(["1"])),
}


@pytest.mark.parametrize("category", sorted(COVERAGE_CASES))
def test_coverage_category_fires_and_demotes(category):
    reports, requests = COVERAGE_CASES[category]
    hard = replace(RAG26, smells=())
    v = ReportVerification(reports, hard, requests=requests).coverage()
    assert any(f.category == category and f.severity == "error" for f in v.findings), \
        f"{category}: expected a hard coverage error"
    soft = replace(RAG26, smells=(category,))
    v = ReportVerification(reports, soft, requests=requests).coverage()
    assert any(f.category == category and f.severity == "warning" for f in v.findings)
    assert not [f for f in v.findings if f.category == category and f.severity == "error"]
