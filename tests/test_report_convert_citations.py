"""Tests for the citation path through convert().

convert() normalizes every input format to the *ragtime* shape ({doc_id: confidence})
and carries (doc_id, confidence) pairs from there to the emitter. Two things follow,
and both are what these tests pin:

- a ragtime source keeps its own confidences instead of having rank-derived ones
  synthesized over them (they used to be replaced by 1/rank unconditionally)
- the citation *order* is the confidence ranking, not the order the run happened to
  serialize the mapping in -- which is what makes "truncation keeps the top-ranked
  citations" true rather than accidental
"""
from dataclasses import replace

import pytest

from autojudge_base.report import (
    NeuclirReportSentence,
    Rag24ReportSentence,
    RagtimeReportSentence,
    Report,
    ReportMetaData,
    convert,
    rank_citations,
    rank_confidences,
)
from autojudge_base.track_spec import get_spec

DOC_A, DOC_B, DOC_C = "doc-a", "doc-b", "doc-c"

# Deliberately NOT in confidence order: doc-c is submitted first but ranks last.
# Anything that reads the mapping in insertion order gets this wrong.
UNORDERED = {DOC_C: 0.1, DOC_A: 0.9, DOC_B: 0.5}
BY_CONFIDENCE = [DOC_A, DOC_B, DOC_C]


def _report(sentences, references=None) -> Report:
    md = ReportMetaData(team_id="T", topic_id="1", run_id="run")
    return Report(metadata=md, responses=sentences, references=references)


def _citations_of(report: Report):
    return [s.citations for s in report.responses]


# ============ the two ranking helpers ============


class TestRankConfidences:
    """Order -> numbers, for formats that state priority as list position."""

    def test_confidence_decreases_with_rank(self):
        conf = rank_confidences(BY_CONFIDENCE)

        assert list(conf) == BY_CONFIDENCE
        assert conf[DOC_A] > conf[DOC_B] > conf[DOC_C]

    def test_duplicates_keep_their_first_position(self):
        conf = rank_confidences([DOC_A, DOC_B, DOC_A])

        assert list(conf) == [DOC_A, DOC_B]
        assert conf[DOC_A] > conf[DOC_B]  # the first occurrence wins, not the last

    def test_empty_list_gives_empty_mapping(self):
        assert rank_confidences([]) == {}


class TestRankCitations:
    """Numbers -> order, keeping the numbers."""

    def test_pairs_come_back_highest_first(self):
        assert rank_citations(UNORDERED) == [(DOC_A, 0.9), (DOC_B, 0.5), (DOC_C, 0.1)]

    def test_none_stays_none(self):
        """Cited nothing is not the same as cited an empty set."""
        assert rank_citations(None) is None

    def test_empty_mapping_stays_empty(self):
        assert rank_citations({}) == []

    def test_ties_keep_their_submitted_order(self):
        tied = {DOC_B: 0.5, DOC_A: 0.5}

        assert [d for d, _ in rank_citations(tied)] == [DOC_B, DOC_A]


# ============ the resolver ============


class TestResolveToRagtime:

    def test_ragtime_confidences_pass_through_untouched(self):
        report = _report([RagtimeReportSentence(text="A.", citations=dict(UNORDERED))])

        resolved = report.get_sentences_with_citation_confidences()

        assert resolved[0].citations == UNORDERED

    def test_neuclir_order_becomes_confidence(self):
        report = _report([NeuclirReportSentence(text="A.", citations=list(BY_CONFIDENCE))])

        conf = report.get_sentences_with_citation_confidences()[0].citations

        assert conf == rank_confidences(BY_CONFIDENCE)

    def test_rag24_indices_resolve_against_references(self):
        report = _report(
            [Rag24ReportSentence(text="A.", citations=[2, 0])],
            references=[DOC_A, DOC_B, DOC_C],
        )

        conf = report.get_sentences_with_citation_confidences()[0].citations

        assert list(conf) == [DOC_C, DOC_A]  # cited order, not references order
        assert conf[DOC_C] > conf[DOC_A]

    def test_a_sentence_that_cited_nothing_stays_none(self):
        report = _report([
            RagtimeReportSentence(text="A.", citations={DOC_A: 1.0}),
            RagtimeReportSentence(text="B.", citations=None),
        ])

        assert [s.citations for s in report.get_sentences_with_citation_confidences()][1] is None

    def test_unknown_sentence_format_raises_instead_of_vanishing(self):
        """A dropped sentence would silently shorten the report."""
        class _AlienSentence:
            text = "Alien."
            citations = None

        report = _report([RagtimeReportSentence(text="A.", citations={DOC_A: 1.0})])
        report.responses = [_AlienSentence()]

        with pytest.raises(RuntimeError, match="Unknown sentence format"):
            report.get_sentences_with_citation_confidences()


# ============ convert: confidences ============


class TestRagtimeToRagtime:

    def test_confidences_survive(self):
        """Regression: these were overwritten with 1/rank on every conversion."""
        report = _report([RagtimeReportSentence(text="A.", citations=dict(UNORDERED))])

        out = convert(report, "ragtime26")

        assert _citations_of(out) == [UNORDERED]

    def test_citations_are_ordered_by_confidence(self):
        report = _report([RagtimeReportSentence(text="A.", citations=dict(UNORDERED))])

        out = convert(report, "ragtime26")

        assert list(out.responses[0].citations) == BY_CONFIDENCE

    def test_a_sentence_that_cited_nothing_stays_none(self):
        report = _report([
            RagtimeReportSentence(text="A.", citations=dict(UNORDERED)),
            RagtimeReportSentence(text="B.", citations=None),
        ])

        out = convert(report, "ragtime26")

        assert _citations_of(out)[1] is None  # not {}

    def test_neuclir_source_gets_rank_derived_confidences(self):
        """Nothing to preserve here, so synthesized values are the honest answer."""
        report = _report([NeuclirReportSentence(text="A.", citations=list(BY_CONFIDENCE))])

        out = convert(report, "ragtime26")

        assert _citations_of(out) == [rank_confidences(BY_CONFIDENCE)]


class TestToOrderOnlyFormats:
    """neuclir and rag24 hold priority as list position, so confidences are dropped."""

    def test_neuclir_lists_doc_ids_highest_confidence_first(self):
        report = _report([RagtimeReportSentence(text="A.", citations=dict(UNORDERED))])

        out = convert(report, "kiddie")

        assert _citations_of(out) == [BY_CONFIDENCE]

    def test_rag24_cites_positions_highest_confidence_first(self):
        report = _report([RagtimeReportSentence(text="A.", citations=dict(UNORDERED))])

        out = convert(report, "rag26")

        assert [out.references[i] for i in out.responses[0].citations] == BY_CONFIDENCE

    def test_round_trip_through_ragtime_keeps_the_cited_doc_ids(self):
        report = _report(
            [Rag24ReportSentence(text="A.", citations=[2, 0])],
            references=[DOC_A, DOC_B, DOC_C],
        )

        out = convert(convert(report, "ragtime26"), "rag26")

        assert [out.references[i] for i in out.responses[0].citations] == [DOC_C, DOC_A]


# ============ convert: truncation ============


class TestTruncation:
    """Every bundled spec with a citation cap also declares citation_count a smell,
    so these drive a spec with the smell removed -- the cap only bites there."""

    ENFORCED = replace(get_spec("ragtime26"), max_citations_per_sentence=2, smells=())

    def test_the_lowest_confidence_citation_is_the_one_dropped(self):
        """Not the last-submitted one: doc-c is submitted first but ranks last."""
        report = _report([RagtimeReportSentence(text="A.", citations=dict(UNORDERED))])

        out = convert(report, self.ENFORCED)

        assert list(out.responses[0].citations) == [DOC_A, DOC_B]

    def test_the_survivors_keep_their_real_confidences(self):
        report = _report([RagtimeReportSentence(text="A.", citations=dict(UNORDERED))])

        out = convert(report, self.ENFORCED)

        assert _citations_of(out) == [{DOC_A: 0.9, DOC_B: 0.5}]

    def test_a_smell_means_the_cap_is_not_enforced(self):
        """A smell says the violation is known and must stay visible, not be repaired."""
        report = _report([RagtimeReportSentence(text="A.", citations=dict(UNORDERED))])

        out = convert(report, replace(self.ENFORCED, smells=("citation_count",)))

        assert len(out.responses[0].citations) == 3

    def test_truncated_citations_do_not_reach_references(self):
        report = _report([RagtimeReportSentence(text="A.", citations=dict(UNORDERED))])

        out = convert(report, replace(self.ENFORCED, references_kind="cited_only"))

        assert out.references == [DOC_A, DOC_B]


# ============ convert: references ============


class TestReferences:

    def test_cited_only_is_rebuilt_from_the_citations(self):
        report = _report(
            [RagtimeReportSentence(text="A.", citations=dict(UNORDERED))],
            references=["doc-never-cited"],
        )

        out = convert(report, "ragtime25")

        assert out.references == BY_CONFIDENCE  # citation-priority order, uncited dropped

    def test_other_kinds_keep_the_source_list(self):
        report = _report(
            [RagtimeReportSentence(text="A.", citations={DOC_A: 0.9})],
            references=["doc-never-cited"],
        )

        out = convert(report, "rag26")

        assert out.references == ["doc-never-cited", DOC_A]  # uncited entry survives

    def test_a_cited_doc_missing_from_references_is_appended(self):
        report = _report(
            [RagtimeReportSentence(text="A.", citations=dict(UNORDERED))],
            references=[DOC_B],
        )

        out = convert(report, "rag26")

        assert out.references == [DOC_B, DOC_A, DOC_C]
        assert [out.references[i] for i in out.responses[0].citations] == BY_CONFIDENCE

    def test_source_duplicates_are_collapsed(self):
        report = _report(
            [RagtimeReportSentence(text="A.", citations={DOC_A: 0.9})],
            references=[DOC_A, DOC_A],
        )

        out = convert(report, "rag26")

        assert out.references == [DOC_A]
