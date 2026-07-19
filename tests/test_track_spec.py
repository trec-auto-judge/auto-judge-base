"""Tests for the track-spec verify/convert framework (autojudge_base.track_spec).

Two groups:
  * BACKWARDS COMPATIBILITY - existing Report behavior is unchanged when no spec is
    passed (verify_ragtime with no spec; verify_rag defaulting to the latest RAG spec;
    the structural sniff path).
  * NEW SPEC FRAMEWORK - spec-driven verify per track and the converters.
"""
from types import SimpleNamespace

import pytest

from autojudge_base.report import (
    NeuclirReportSentence,
    Rag24ReportSentence,
    RagtimeReportSentence,
    Report,
    ReportMetaData,
)
from autojudge_base.track_spec import SPECS, TrackSpec, get_spec, to_rag, to_ragtime, verify

# --- sample doc-ids per collection ----------------------------------------------
SHARD, SHARD2, SHARD3 = "shard_00459_61697", "shard_01234_5678", "shard_09999_9"
MSM = "msmarco_v2.1_doc_02_165872989#4_290441710"
MSM2 = "msmarco_v2.1_doc_14_1198634226#9_2470404444"
UUID = "b6a21af8-9cc4-462d-9c70-00bb9f009401_56341480"
UUID2 = "042ce256-aaa2-4944-8725-7deb68b8b43f_125182681"


# --- fixtures (builders) --------------------------------------------------------

def rag26_report(sents=None, refs=None):
    if sents is None:
        sents = [Rag24ReportSentence(text="First sentence.", citations=[0]),
                 Rag24ReportSentence(text="Second sentence.", citations=[1])]
    if refs is None:
        refs = [SHARD, SHARD2]
    md = ReportMetaData(team_id="T", narrative_id="1", narrative="q text",
                        run_id="run", run_desc="desc")
    return Report(metadata=md, responses=sents, references=refs)


def ragtime_report():
    md = ReportMetaData(team_id="T", topic_id="300", run_id="run", run_desc="d",
                        task="multilingual")
    sents = [RagtimeReportSentence(text="Sentence one.", citations={UUID: 100.0}),
             RagtimeReportSentence(text="Sentence two.", citations={UUID2: 50.0})]
    return Report(metadata=md, responses=sents, references=[UUID, UUID2])


def dragun_report(sents=None, refs=None):
    if sents is None:
        sents = [NeuclirReportSentence(text="Sentence one.", citations=[MSM]),
                 NeuclirReportSentence(text="Uncited sentence.", citations=[])]
    md = ReportMetaData(team_id="T", run_id="run", topic_id="msmarco_v2.1_doc_04_420132660",
                        type="automatic", use_starter_kit=0)
    return Report(metadata=md, responses=sents, references=refs)


def rag25_report(sents, refs):
    md = ReportMetaData(team_id="T", run_id="run", type="automatic",
                        narrative_id="1", narrative="q")
    return Report(metadata=md, responses=sents, references=refs)


# ================================================================================
# BACKWARDS COMPATIBILITY
# ================================================================================

def test_verify_ragtime_no_spec_unchanged():
    # existing method, called the old way (no spec) -> still validates in place
    assert ragtime_report().verify_ragtime() is True


def test_verify_ragtime_no_spec_check_doc_ids_flag():
    # the earlier-added flag still silences the docid warning without raising
    assert ragtime_report().verify_ragtime(check_doc_ids=False) is True


def test_verify_ragtime_no_spec_rejects_mismatched_refs():
    r = ragtime_report()
    r.references = [UUID]  # citations reference UUID2 too -> reference set mismatch
    with pytest.raises(RuntimeError):
        r.verify_ragtime()


def test_verify_rag_defaults_to_rag26():
    # verify_rag() with no spec must behave as a RAG (rag26) check
    assert rag26_report().verify_rag() is True


def test_structural_sniff_accepts_valid():
    assert rag26_report().verify() is True  # spec=None -> structural only


def test_structural_sniff_rejects_bad_index():
    bad = [Rag24ReportSentence(text="x", citations=[9])]  # index out of range
    with pytest.raises(RuntimeError):
        rag26_report(sents=bad).verify()


# ================================================================================
# NEW SPEC FRAMEWORK
# ================================================================================

def test_registry_loads_five_specs():
    assert set(SPECS) == {"rag26", "rag25", "ragtime25", "ragtime26", "dragun25"}
    assert all(isinstance(s.sentence_type, tuple) for s in SPECS.values())
    assert SPECS["rag26"].sentence_type == ("rag24", "neuclir")


def test_scalar_sentence_type_coerced_to_list():
    spec = TrackSpec.from_dict({
        "track": "x", "task": "t", "sentence_type": "neuclir",
        "sentences_key": "responses", "topic_id_field": "topic_id",
        "mandatory_metadata": ["team_id", "run_id"],
    })
    assert spec.sentence_type == ("neuclir",)
    assert spec.emit_sentence_type == "neuclir"


def test_rag26_valid():
    assert rag26_report().verify(SPECS["rag26"]) is True


def test_rag26_too_many_citations():
    bad = [Rag24ReportSentence(text="x", citations=[0, 1, 0, 1])]
    with pytest.raises(RuntimeError, match="citations"):
        rag26_report(sents=bad).verify(SPECS["rag26"])


def test_rag26_bad_docid_pattern():
    r = rag26_report(sents=[Rag24ReportSentence(text="x", citations=[0])],
                     refs=["not_a_shard_id"])
    with pytest.raises(RuntimeError, match="docid"):
        r.verify(SPECS["rag26"])


def test_rag26_missing_metadata():
    r = rag26_report()
    r.metadata.run_desc = ""  # required, now empty
    with pytest.raises(RuntimeError, match="run_desc"):
        r.verify(SPECS["rag26"])


def test_rag26_wrong_sentence_type_rejected():
    # ragtime sentences are not accepted by rag26 (only rag24 | neuclir)
    r = Report(metadata=ReportMetaData(team_id="T", narrative_id="1", narrative="q",
                                       run_id="run", run_desc="d"),
               responses=[RagtimeReportSentence(text="x", citations={SHARD: 1.0})],
               references=[SHARD])
    with pytest.raises(RuntimeError, match="sentence_type"):
        r.verify(SPECS["rag26"])


def test_rag26_neuclir_now_accepted():
    # organizers added neuclir doc-id-string citations to rag26
    r = Report(metadata=ReportMetaData(team_id="T", narrative_id="1", narrative="q",
                                       run_id="run", run_desc="d"),
               responses=[NeuclirReportSentence(text="x", citations=[SHARD])],
               references=[SHARD])
    assert r.verify(SPECS["rag26"]) is True


def test_rag26_references_must_equal_cited():
    r = rag26_report(refs=[SHARD, SHARD2, SHARD3])  # SHARD3 never cited
    with pytest.raises(RuntimeError, match="cited"):
        r.verify(SPECS["rag26"])


def test_ragtime26_valid_and_length_from_request():
    r = ragtime_report()
    assert r.verify(SPECS["ragtime26"]) is True                    # no request -> length skipped
    assert r.verify(SPECS["ragtime26"], request=SimpleNamespace(limit=10_000)) is True
    with pytest.raises(RuntimeError, match="chars"):
        r.verify(SPECS["ragtime26"], request=SimpleNamespace(limit=5))


def test_ragtime26_rejects_rag_sentences():
    # RAG (rag24 int-index) sentences verified under the RAGTIME spec, which accepts
    # only 'ragtime' -> sentence_type rejection (the inverse of the rag26+ragtime case)
    with pytest.raises(RuntimeError, match="sentence_type"):
        rag26_report().verify(SPECS["ragtime26"])


def test_ragtime26_rejects_wrong_collection_docids():
    # The docid issue we hit converting a ClimbMix report to RAGTIME representation:
    # the sentence FORMAT is correct (ragtime score-dict), but the doc-ids are ClimbMix
    # 'shard_*' ids, which the RAGTIME NeuCLIR-uuid docid pattern rejects. Collection-
    # aware strictness catches a wrong-collection report even when the format matches.
    md = ReportMetaData(team_id="T", topic_id="1", run_id="r", run_desc="d")
    r = Report(metadata=md,
               responses=[RagtimeReportSentence(text="x", citations={SHARD: 1.0})],
               references=[SHARD])
    with pytest.raises(RuntimeError, match="docid"):
        r.verify(SPECS["ragtime26"])


def test_converted_report_fails_target_spec_on_collection():
    # end-to-end: rag26 (ClimbMix) -> to_ragtime gives valid RAGTIME *shape* but the
    # ClimbMix doc-ids make it fail the ragtime26 spec's docid check.
    rt = rag26_report().to_ragtime()
    with pytest.raises(RuntimeError, match="docid"):
        rt.verify(SPECS["ragtime26"])


def test_dragun_valid_no_references():
    assert dragun_report().verify(SPECS["dragun25"]) is True


def test_dragun_rejects_present_references():
    with pytest.raises(RuntimeError, match="references"):
        dragun_report(refs=[MSM]).verify(SPECS["dragun25"])


def test_dragun_too_many_citations():
    bad = [NeuclirReportSentence(text="x", citations=[MSM, MSM2, MSM, MSM2])]
    with pytest.raises(RuntimeError, match="citations"):
        dragun_report(sents=bad).verify(SPECS["dragun25"])


def test_rag25_accepts_both_formats():
    # Format 1: integer indices
    fmt1 = rag25_report([Rag24ReportSentence(text="x", citations=[0])], refs=[MSM])
    assert fmt1.verify(SPECS["rag25"]) is True
    # Format 2: doc-id strings
    fmt2 = rag25_report([NeuclirReportSentence(text="x", citations=[MSM])], refs=[MSM, MSM2])
    assert fmt2.verify(SPECS["rag25"]) is True


def test_rag25_retrieval_list_max():
    # references may exceed the cited set (retrieval list), but not the 100 cap
    over = rag25_report([Rag24ReportSentence(text="x", citations=[0])],
                        refs=[f"msmarco_v2.1_doc_00_{i}#0_{i}" for i in range(101)])
    with pytest.raises(RuntimeError, match="references"):
        over.verify(SPECS["rag25"])


def test_get_spec_task_guard():
    assert get_spec("rag26").track == "rag26"
    with pytest.raises(KeyError):
        get_spec("rag26", task="repgen")   # rag26 is 'generation'
    with pytest.raises(KeyError):
        get_spec("nonesuch")


# --- converters -----------------------------------------------------------------

def test_to_ragtime_converts_indices_to_score_dict():
    rt = rag26_report().to_ragtime()
    assert all(isinstance(s, RagtimeReportSentence) for s in rt.responses)
    assert rt.responses[0].citations == {SHARD: 1.0}
    assert rt.references == [SHARD, SHARD2]      # cited union, deduped, ordered
    # representation is valid (structural); a full RAGTIME check is intentionally NOT
    # applied here - these are ClimbMix doc-ids, which the RAGTIME docid pattern rejects.
    assert rt.verify() is True


def test_to_rag_converts_docids_to_indices():
    src = dragun_report()  # neuclir doc-id-list sentences
    rag = to_rag(src, SPECS["rag26"])
    assert all(isinstance(s, Rag24ReportSentence) for s in rag.responses)
    assert rag.references == [MSM]                 # cited union, deduped, ordered
    assert rag.responses[0].citations == [0]
    assert rag.responses[1].citations == []        # the uncited sentence stays empty


def test_convert_round_trips_verify():
    src = rag26_report()
    # same-collection round trip through the RAG spec passes the full strict check
    assert src.to_rag(SPECS["rag26"]).verify(SPECS["rag26"]) is True
    # representation change to ragtime is structurally valid (a full RAGTIME check
    # needs a matching-collection spec, which ClimbMix -> NeuCLIR deliberately is not)
    assert src.to_ragtime().verify() is True


# --- converters across all three INPUT sentence formats -------------------------
# Each builder uses a different source representation and doc-id collection:
#   rag26_report  -> Rag24    (int indices),   ClimbMix shard ids
#   ragtime_report-> Ragtime  (doc_id:score),  NeuCLIR uuid ids
#   dragun_report -> Neuclir  (doc_id list),   MS MARCO ids  (2nd sentence uncited)
_SOURCES = [
    (rag26_report, [[SHARD], [SHARD2]]),
    (ragtime_report, [[UUID], [UUID2]]),
    (dragun_report, [[MSM], []]),
]


@pytest.mark.parametrize("builder,expected_cited", _SOURCES)
def test_to_rag_from_any_source_format(builder, expected_cited):
    rag = builder().to_rag()  # default spec (rag25) -> emit rag24 int indices
    assert all(isinstance(s, Rag24ReportSentence) for s in rag.responses)
    refs = rag.references
    # resolve each sentence's indices back to doc-ids; must match the source's cited set
    resolved = [[refs[i] for i in s.citations] for s in rag.responses]
    assert resolved == expected_cited
    # references are exactly the cited union, deduped
    assert set(refs) == {d for sent in expected_cited for d in sent}


@pytest.mark.parametrize("builder,expected_cited", _SOURCES)
def test_to_ragtime_from_any_source_format(builder, expected_cited):
    rt = builder().to_ragtime()  # default spec (ragtime25) -> emit doc_id:score dict
    assert all(isinstance(s, RagtimeReportSentence) for s in rt.responses)
    cited = [list(s.citations.keys()) for s in rt.responses]
    assert cited == expected_cited
    # every emitted score is the canonical 1.0 (source scores are not carried through)
    assert all(v == 1.0 for s in rt.responses for v in s.citations.values())


def test_to_rag_from_ragtime_orders_multicite_by_confidence():
    # a single sentence citing two docs at different confidences
    md = ReportMetaData(team_id="T", topic_id="1", run_id="r", run_desc="d", task="english")
    src = Report(metadata=md,
                 responses=[RagtimeReportSentence(text="x", citations={UUID: 10.0, UUID2: 90.0})],
                 references=[UUID, UUID2])
    rag = src.to_rag()
    refs = rag.references
    # higher confidence (UUID2) comes first after conversion
    assert [refs[i] for i in rag.responses[0].citations] == [UUID2, UUID]


def test_to_ragtime_from_ragtime_normalizes_scores():
    # ragtime -> ragtime is lossy on scores (all become 1.0) but preserves the doc-id set
    rt = ragtime_report().to_ragtime()
    assert rt.responses[0].citations == {UUID: 1.0}
    assert rt.responses[1].citations == {UUID2: 1.0}


def test_converters_respect_explicit_spec():
    src = rag26_report()
    assert all(isinstance(s, Rag24ReportSentence) for s in to_rag(src, SPECS["rag26"]).responses)
    assert all(isinstance(s, RagtimeReportSentence) for s in to_ragtime(src, SPECS["ragtime26"]).responses)
