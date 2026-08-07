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
from autojudge_base.track_spec import (
    SPECS, TrackSpec, check, convert, findings, get_spec, length_count, to_rag, to_ragtime, verify,
)

# --- sample doc-ids per collection ----------------------------------------------
SHARD, SHARD2, SHARD3 = "shard_00459_61697", "shard_01234_5678", "shard_09999_9"
MSM = "msmarco_v2.1_doc_02_165872989#4_290441710"
MSM2 = "msmarco_v2.1_doc_14_1198634226#9_2470404444"
UUID = "b6a21af8-9cc4-462d-9c70-00bb9f009401_56341480"
UUID2 = "042ce256-aaa2-4944-8725-7deb68b8b43f_125182681"
UUID3 = "c8d39ea5-a01c-4b67-9ce7-d821a918ed21_385620303"


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


def kiddie_report():
    # synthetic smoke-test format: minimal metadata, neuclir citations with arbitrary
    # doc-ids (no docid pattern), no references
    md = ReportMetaData(team_id="T", run_id="run", topic_id="leaf")
    return Report(metadata=md,
                  responses=[NeuclirReportSentence(text="A sentence.", citations=["Toronto_Maple_Leafs"])],
                  references=None)


# ================================================================================
# BACKWARDS COMPATIBILITY
# ================================================================================

def test_verify_ragtime_no_spec_unchanged():
    # existing method, called the old way (no spec) -> still validates in place,
    # now DEPRECATED in favor of verify(spec=...)
    with pytest.warns(DeprecationWarning):
        assert ragtime_report().verify_ragtime() is True


def test_verify_ragtime_no_spec_check_doc_ids_flag():
    # the earlier-added flag still silences the docid warning without raising
    with pytest.warns(DeprecationWarning):
        assert ragtime_report().verify_ragtime(check_doc_ids=False) is True


def test_verify_ragtime_no_spec_rejects_mismatched_refs():
    r = ragtime_report()
    r.references = [UUID]  # citations reference UUID2 too -> reference set mismatch
    with pytest.warns(DeprecationWarning), pytest.raises(RuntimeError):
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

def test_check_collects_all_categories():
    # collect-mode returns every category finding at once (vs verify's first-raise).
    # Two sentences (one valid, one blank) so this is not a wholly-empty report, which
    # would short-circuit to a single "empty answer" message instead of collecting all.
    r = rag26_report(sents=[Rag24ReportSentence(text="ok.", citations=[0, 1, 0, 1]),
                            Rag24ReportSentence(text="", citations=[0])],
                     refs=[SHARD, SHARD2, SHARD3])
    r.metadata.run_desc = ""
    errs, warns = findings(r, SPECS["rag26"])
    joined = " ".join(errs + warns)  # blank-sentence text is a smell (warning); the rest errors
    assert "run_desc" in joined      # metadata (error)
    assert "text must be" in joined  # blank sentence text (smell)
    assert "citations (max 3)" in joined
    assert "never cited" in joined   # references: SHARD3 present but never cited


def test_check_valid_returns_empty():
    assert check(rag26_report(), SPECS["rag26"]) == []


def test_rag25_autojudge_accepts_ragtime_and_ignores_references():
    # the auto-judge round-trip re-emits rag runs with ragtime citations, drops
    # narrative/type, and may carry an unrelated references array -> all tolerated.
    md = ReportMetaData(team_id="T", run_id="asp", narrative_id="464", topic_id="464", task="rag")
    r = Report(metadata=md,
               responses=[RagtimeReportSentence(text="A sentence.", citations={MSM: 1.0})],
               references=["not-a-valid-docid", MSM2])   # ignored entirely (references_kind=ignore)
    assert check(r, SPECS["rag25-autojudge"]) == []
    assert r.verify(SPECS["rag25-autojudge"]) is True
    # but a bad CITED doc-id (which is not ignored) is still a hard error
    bad = Report(metadata=md, references=[],
                 responses=[RagtimeReportSentence(text="s.", citations={"bogus-id": 1.0})])
    assert any("docid pattern" in e for e in check(bad, SPECS["rag25-autojudge"]))


def test_check_empty_answer_messages_use_sentences_key():
    # an empty answer is a SMELL by default -> a WARNING (not a hard error), keyed by the
    # track's sentences_key ("answer" for RAG, "responses" for RAGTIME).
    err, warn = findings(rag26_report(sents=[], refs=[]), SPECS["rag26"])
    assert err == [] and warn == ["rag26: empty answer given (no sentences)"]
    blank_rag = rag26_report(sents=[Rag24ReportSentence(text="  ", citations=[0])], refs=[SHARD])
    err, warn = findings(blank_rag, SPECS["rag26"])
    assert err == [] and warn == ["rag26: empty answer given (all sentences blank)"]
    md = ReportMetaData(team_id="T", topic_id="300", run_id="run", task="multilingual")
    blank_rt = Report(metadata=md, references=[],
                      responses=[RagtimeReportSentence(text="", citations={})])
    err, warn = findings(blank_rt, SPECS["ragtime25"])
    assert err == [] and warn == ["ragtime25: empty responses given (all sentences blank)"]


def test_ragtime26_references_required_complete_exact():
    # ragtime26 (our submission year): references key is REQUIRED (the official
    # validator rejects its absence) and must be exactly the cited set -- absent and
    # incomplete are both hard errors.
    md = ReportMetaData(team_id="T", topic_id="300", run_id="run", run_desc="d")
    cited = [RagtimeReportSentence(text="Sentence.", citations={UUID: 1.0})]
    absent = check(Report(metadata=md, responses=cited, references=None), SPECS["ragtime26"])
    assert any("requires a references array" in e for e in absent)
    empty = check(Report(metadata=md, responses=cited, references=[]), SPECS["ragtime26"])
    assert any("not listed in references" in e for e in empty)   # incomplete -> hard
    exact = Report(metadata=md, responses=cited, references=[UUID])
    assert check(exact, SPECS["ragtime26"]) == []
    # ...and a wrong reference is wrong in BOTH directions, each a hard error: the
    # entry is uncited AND the cited doc is unlisted (complete-and-exact contract).
    mismatch = Report(metadata=md, responses=cited, references=[UUID2])
    err, _warn = findings(mismatch, SPECS["ragtime26"])
    assert any("never cited" in e for e in err)                # references_uncited
    assert any("not listed in references" in e for e in err)   # references_undeclared


def test_ragtime25_relaxed_references_and_task():
    # ragtime25 checks runs NIST already accepted: absent/incomplete references and a
    # missing task only WARN; an uncited reference entry stays a hard error.
    md = ReportMetaData(team_id="T", topic_id="300", run_id="run")   # no task
    cited = [RagtimeReportSentence(text="Sentence.", citations={UUID: 1.0})]
    err, warn = findings(Report(metadata=md, responses=cited, references=None),
                         SPECS["ragtime25"])
    assert err == []
    assert any("requires a references array" in w for w in warn)
    assert any("task" in w and "recommended" in w for w in warn)
    err, warn = findings(Report(metadata=md, responses=cited, references=[]),
                         SPECS["ragtime25"])
    assert err == [] and any("not listed in references" in w for w in warn)
    err, _warn = findings(Report(metadata=md, responses=cited, references=[UUID, UUID2]),
                          SPECS["ragtime25"])
    assert any("never cited" in e for e in err)   # uncited entry stays hard


def test_check_ragtime_length_from_request():
    # RAGTIME's char limit lives on the Request (length_limit_request_field=limit); it is
    # only enforced when a matching Request is supplied.
    from autojudge_base.request import Request
    md = ReportMetaData(team_id="T", topic_id="300", run_id="run", task="multilingual")
    r = Report(metadata=md, references=[UUID],   # exactly the cited set (required)
               responses=[RagtimeReportSentence(text="x" * 50, citations={UUID: 1.0})])
    assert findings(r, SPECS["ragtime25"]) == ([], [])            # no request -> length skipped
    err, warn = findings(r, SPECS["ragtime25"], request=Request(request_id="300", title="t", limit=10))
    assert err == [] and any("chars" in w and "limit 10" in w for w in warn)   # 50 chars > 10 (smell)


def test_report_check_method():
    # Report.check(spec) mirrors the module function (collect-mode)
    assert rag26_report().check(SPECS["rag26"]) == []
    r = rag26_report()
    r.metadata.run_desc = ""
    assert any("run_desc" in e for e in r.check(SPECS["rag26"]))


def test_check_structural_none_spec():
    assert check(rag26_report()) == []
    assert len(check(rag26_report(sents=[Rag24ReportSentence(text="x", citations=[9])]))) == 1


def test_registry_loads_specs():
    assert set(SPECS) == {"rag26", "rag25", "rag25-autojudge", "ragtime25", "ragtime26", "dragun25", "kiddie"}
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


def test_rag26_too_many_citations_is_a_smell():
    # citation_count is a SMELL for rag26 (#4): over-limit sentences are accepted by
    # the organizers but left unjudged, so verify passes and the finding is a warning.
    bad = [Rag24ReportSentence(text="x", citations=[0, 1, 0, 1])]
    r = rag26_report(sents=bad)
    err, warn = findings(r, SPECS["rag26"])
    assert err == [] and any("citations" in w for w in warn)
    assert r.verify(SPECS["rag26"]) is True


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
    with pytest.raises(RuntimeError, match="citation format"):
        r.verify(SPECS["rag26"])


def test_rag26_neuclir_now_accepted():
    # organizers added neuclir doc-id-string citations to rag26
    r = Report(metadata=ReportMetaData(team_id="T", narrative_id="1", narrative="q",
                                       run_id="run", run_desc="d"),
               responses=[NeuclirReportSentence(text="x", citations=[SHARD])],
               references=[SHARD])
    assert r.verify(SPECS["rag26"]) is True


def test_rag26_uncited_reference_is_a_smell():
    # rag-task.md: "Uncited entries do not hurt the score" -- the retrieval list may
    # carry them; the organizers suggest not to, so it warns without failing.
    r = rag26_report(refs=[SHARD, SHARD2, SHARD3])  # SHARD3 never cited
    err, warn = findings(r, SPECS["rag26"])
    assert err == [] and any("never cited" in w for w in warn)
    assert r.verify(SPECS["rag26"]) is True


def test_ragtime26_valid_and_length_from_request():
    r = ragtime_report()
    assert r.verify(SPECS["ragtime26"]) is True                    # no request -> length skipped
    assert r.verify(SPECS["ragtime26"], request=SimpleNamespace(limit=10_000)) is True
    # over the per-request limit is a SMELL for RAGTIME (validator truncates, not rejects)
    err, warn = findings(r, SPECS["ragtime26"], request=SimpleNamespace(limit=5))
    assert err == [] and any("chars" in w for w in warn)
    assert r.verify(SPECS["ragtime26"], request=SimpleNamespace(limit=5)) is True


def test_ragtime26_rejects_rag_sentences():
    # RAG (rag24 int-index) sentences verified under the RAGTIME spec, which accepts
    # only 'ragtime' -> sentence_type rejection (the inverse of the rag26+ragtime case)
    with pytest.raises(RuntimeError, match="citation format"):
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
    # end-to-end: rag26 (ClimbMix) -> to_ragtime(ragtime26) gives valid RAGTIME *shape*
    # (metadata remapped, run_desc carried) but the ClimbMix doc-ids fail ragtime26's
    # docid check. Convert and verify with the SAME spec to isolate the docid failure.
    rt = rag26_report().convert("ragtime26")
    with pytest.raises(RuntimeError, match="docid"):
        rt.verify(SPECS["ragtime26"])


def test_dragun_valid_no_references():
    assert dragun_report().verify(SPECS["dragun25"]) is True


def test_dragun_present_references_is_a_smell():
    # dragun uses no references array; a stray one is a SMELL (warning), not a failure
    r = dragun_report(refs=[MSM])
    err, warn = findings(r, SPECS["dragun25"])
    assert err == [] and any("references" in w for w in warn)
    assert r.verify(SPECS["dragun25"]) is True


def test_dragun_too_many_citations_is_a_smell():
    # citation_count is a SMELL for dragun25 (#4), like rag26.
    bad = [NeuclirReportSentence(text="x", citations=[MSM, MSM2, MSM, MSM2])]
    r = dragun_report(sents=bad)
    err, warn = findings(r, SPECS["dragun25"])
    assert err == [] and any("citations" in w for w in warn)
    assert r.verify(SPECS["dragun25"]) is True


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


# ================================================================================
# PER-FORMAT COVERAGE (each track's valid case + its distinctive rules)
# ================================================================================

def _valid_rag25():
    return rag25_report([Rag24ReportSentence(text="A sentence.", citations=[0])], [MSM])


#: a canonical VALID report per track (ragtime_report carries both task and run_desc,
#: so it satisfies ragtime25 and ragtime26)
VALID_BUILDERS = {
    "rag26": rag26_report,
    "rag25": _valid_rag25,
    "ragtime25": ragtime_report,
    "ragtime26": ragtime_report,
    "dragun25": dragun_report,
    "kiddie": kiddie_report,
}


@pytest.mark.parametrize("track", sorted(VALID_BUILDERS))
def test_valid_report_passes_its_spec(track):
    report = VALID_BUILDERS[track]()
    spec = SPECS[track]
    assert report.verify(spec) is True
    assert report.check(spec) == []


@pytest.mark.parametrize("track", sorted(VALID_BUILDERS))
def test_invalid_report_fails_its_spec(track):
    # symmetric negative baseline: break a universally-mandatory field so the report
    # is no longer valid. verify() raises (it never returns False); check() is non-empty.
    spec = SPECS[track]
    report = VALID_BUILDERS[track]()
    report.metadata.team_id = ""  # required by every spec
    assert report.check(spec) != []
    with pytest.raises(RuntimeError, match="team_id"):
        report.verify(spec)


@pytest.mark.parametrize("track", sorted(VALID_BUILDERS))
def test_wrong_collection_docids_fail_each_spec(track):
    # start from a VALID report (metadata etc. fine) and swap in a doc-id from another
    # collection: the ONLY violation must be the spec's docid check.
    spec = SPECS[track]
    if spec.docid_pattern is None:
        pytest.skip(f"{track} has no docid pattern (accepts any doc-id)")
    bogus = "wrong_collection_id_999"
    report = VALID_BUILDERS[track]()
    tag = spec.emit_sentence_type
    if tag == "rag24":
        report.responses = [Rag24ReportSentence(text="s.", citations=[0])]
        report.references = [bogus]
    elif tag == "ragtime":
        report.responses = [RagtimeReportSentence(text="s.", citations={bogus: 1.0})]
        report.references = [bogus]
    else:  # neuclir
        report.responses = [NeuclirReportSentence(text="s.", citations=[bogus])]
        report.references = None if spec.references_kind == "none" else [bogus]
    assert any("docid" in e for e in report.check(spec))
    with pytest.raises(RuntimeError, match="docid"):
        report.verify(spec)


# --- rag25: retrieval_list references, 400 words, mandatory 'type' ---------------

def test_rag25_retrieval_list_uncited_references_are_a_smell():
    # the retrieval list MAY contain uncited docs, but the organizers suggest not to:
    # verify still passes, and the uncited entry is surfaced as a warning
    r = rag25_report([Rag24ReportSentence(text="s.", citations=[0])], [MSM, MSM2])
    err, warn = findings(r, SPECS["rag25"])
    assert err == [] and any("never cited" in w for w in warn)
    assert r.verify(SPECS["rag25"]) is True


def test_ragtime26_uncited_reference_is_a_hard_error():
    # ragtime26: references, when given, must be exactly the union of cited docs
    md = ReportMetaData(team_id="T", topic_id="300", run_id="run", run_desc="d")
    r = Report(metadata=md,
               responses=[RagtimeReportSentence(text="s.", citations={UUID: 1.0})],
               references=[UUID, UUID2])
    err, _warn = findings(r, SPECS["ragtime26"])
    assert any("never cited" in e for e in err)


def test_ragtime26_missing_run_desc_is_a_smell():
    md = ReportMetaData(team_id="T", topic_id="300", run_id="run")
    r = Report(metadata=md,
               responses=[RagtimeReportSentence(text="s.", citations={UUID: 1.0})],
               references=[UUID])
    err, warn = findings(r, SPECS["ragtime26"])
    assert err == [] and any("run_desc" in w and "recommended" in w for w in warn)


def test_ragtime26_run_id_over_25_chars_rejected():
    md = ReportMetaData(team_id="T", topic_id="300", run_id="x" * 26, run_desc="d")
    r = Report(metadata=md,
               responses=[RagtimeReportSentence(text="s.", citations={UUID: 1.0})],
               references=[UUID])
    err, _warn = findings(r, SPECS["ragtime26"])
    assert any("25-character limit" in e for e in err)


def test_rag26_extra_metadata_fields_tolerated():
    # rag-task.md: metadata "may also contain any additional participant-defined fields"
    assert SPECS["rag26"].forbid_extra_metadata is False


def test_rag25_word_limit_is_a_smell():
    # "less than 400 words" carries no rejection language, and this spec mostly checks
    # participant runs the organizers already accepted -> warn, don't fail.
    over = SPECS["rag25"].length_limit + 10  # derive from the spec, don't hardcode 400
    long = rag25_report([Rag24ReportSentence(text="word " * over, citations=[0])], [MSM])
    err, warn = findings(long, SPECS["rag25"])
    assert err == [] and any("words" in w for w in warn)
    assert long.verify(SPECS["rag25"]) is True


def test_rag25_requires_type():
    r = _valid_rag25()
    r.metadata.type = None
    assert any("metadata.type" in e for e in r.check(SPECS["rag25"]))


# --- ragtime25: mandatory 'task', cited_only union ------------------------------

def test_ragtime25_task_recommended_not_required():
    # 835 accepted 2025 runs lack metadata.task -> recommended, its absence only warns
    r = ragtime_report()
    r.metadata.task = None
    err, warn = findings(r, SPECS["ragtime25"])
    assert not any("metadata.task" in e for e in err)
    assert any("metadata.task" in w for w in warn)


def test_ragtime25_uncited_reference_is_a_hard_error():
    # ragtime does not allow uncited references: an entry cited by no sentence fails
    r = ragtime_report()
    r.references = [UUID, UUID2, UUID3]  # UUID3 present but never cited
    err, _warn = findings(r, SPECS["ragtime25"])
    assert any("never cited" in e for e in err)
    with pytest.raises(RuntimeError, match="never cited"):
        r.verify(SPECS["ragtime25"])


def test_ragtime_length_is_chars_from_request():
    # RAGTIME length is per-request NFKC Unicode chars (not words); derive the boundary
    # from the framework's own char counter rather than a magic number. Over-length is a
    # SMELL for RAGTIME (matching the official validator's default), not a failure.
    r = ragtime_report()
    spec = SPECS["ragtime26"]
    assert spec.length_unit == "chars"
    n_chars = length_count([s.text for s in r.responses], spec.length_unit)
    assert findings(r, spec, request=SimpleNamespace(limit=n_chars)) == ([], [])   # exactly at limit
    err, warn = findings(r, spec, request=SimpleNamespace(limit=n_chars - 1))       # one char over
    assert err == [] and any("chars" in w for w in warn)
    assert r.verify(spec, request=SimpleNamespace(limit=n_chars - 1)) is True       # smell -> no raise


# --- ragtime26: run_id <= 25, mandatory run_desc --------------------------------

def test_run_id_length_constraint_per_spec():
    # RAGTIME caps run_id at 25 chars ("Maximum of 25 characters"); the RAG family
    # states no constraint.
    assert SPECS["ragtime25"].run_id_max_len == 25
    assert SPECS["ragtime26"].run_id_max_len == 25
    assert SPECS["rag26"].run_id_max_len is None
    r = ragtime_report()
    r.metadata.run_id = "x" * 100
    with pytest.raises(RuntimeError, match="25-character limit"):
        r.verify(SPECS["ragtime26"])


def test_ragtime26_run_desc_recommended_not_required():
    # the guideline names run_desc but gives no guidance on its content -> its absence
    # is a metadata_recommended smell, never a hard error
    r = ragtime_report()
    r.metadata.run_desc = None
    err, warn = findings(r, SPECS["ragtime26"])
    assert not any("run_desc" in e for e in err)
    assert any("run_desc" in w for w in warn)


# --- dragun25: neuclir docids, no references, mandatory type/use_starter_kit -----

def test_dragun_bad_docid():
    bad = dragun_report(sents=[NeuclirReportSentence(text="s.", citations=["not_msmarco"])])
    with pytest.raises(RuntimeError, match="docid"):
        bad.verify(SPECS["dragun25"])


def test_dragun_requires_type_and_starter_kit_flag():
    r = dragun_report()
    r.metadata.type = None
    assert any("metadata.type" in e for e in r.check(SPECS["dragun25"]))
    # use_starter_kit == 0 is a valid (present) value, not "missing"
    ok = dragun_report()
    assert not any("use_starter_kit" in e for e in ok.check(SPECS["dragun25"]))


def test_dragun_word_limit():
    over = SPECS["dragun25"].length_limit + 10  # derive from the spec, don't hardcode 250
    long = dragun_report(sents=[NeuclirReportSentence(text="word " * over, citations=[MSM])])
    with pytest.raises(RuntimeError, match="words"):
        long.verify(SPECS["dragun25"])


def test_get_spec_task_guard():
    assert get_spec("rag26").track == "rag26"
    with pytest.raises(KeyError):
        get_spec("rag26", task="repgen")   # rag26 is 'generation'
    with pytest.raises(KeyError):
        get_spec("nonesuch")


# --- converters -----------------------------------------------------------------

def test_to_ragtime_converts_indices_to_score_dict():
    rt = convert(rag26_report(), "ragtime26")
    assert all(isinstance(s, RagtimeReportSentence) for s in rt.responses)
    assert rt.responses[0].citations == {SHARD: 1.0}
    assert rt.references == [SHARD, SHARD2]      # cited union, deduped, ordered
    # representation is valid (structural); a full RAGTIME check is intentionally NOT
    # applied here - these are ClimbMix doc-ids, which the RAGTIME docid pattern rejects.
    assert rt.verify() is True


def test_to_rag_converts_docids_to_indices():
    src = dragun_report()  # neuclir doc-id-list sentences
    rag = convert(src, SPECS["rag26"])
    assert all(isinstance(s, Rag24ReportSentence) for s in rag.responses)
    assert rag.references == [MSM]                 # cited union, deduped, ordered
    assert rag.responses[0].citations == [0]
    assert rag.responses[1].citations == []        # the uncited sentence stays empty


def test_convert_round_trips_verify():
    src = rag26_report()
    # same-collection round trip through the RAG spec passes the full strict check
    assert src.convert("rag26").verify("rag26") is True
    # representation change to ragtime is structurally valid (a full RAGTIME check
    # needs a matching-collection spec, which ClimbMix -> NeuCLIR deliberately is not)
    assert convert(src, "ragtime26").verify() is True


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
    rag = convert(builder(), "rag26")  # explicit spec -> emit rag24 int indices
    assert all(isinstance(s, Rag24ReportSentence) for s in rag.responses)
    refs = rag.references
    # resolve each sentence's indices back to doc-ids; must match the source's cited set
    resolved = [[refs[i] for i in s.citations] for s in rag.responses]
    assert resolved == expected_cited
    # references are exactly the cited union, deduped
    assert set(refs) == {d for sent in expected_cited for d in sent}


@pytest.mark.parametrize("builder,expected_cited", _SOURCES)
def test_to_ragtime_from_any_source_format(builder, expected_cited):
    rt = convert(builder(), "ragtime26")  # explicit spec -> emit doc_id:score dict
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
    rag = convert(src, "rag26")
    refs = rag.references
    # higher confidence (UUID2) comes first after conversion
    assert [refs[i] for i in rag.responses[0].citations] == [UUID2, UUID]


def test_to_ragtime_from_ragtime_normalizes_scores():
    # ragtime -> ragtime is lossy on scores (all become 1.0) but preserves the doc-id set
    rt = convert(ragtime_report(), "ragtime26")
    assert rt.responses[0].citations == {UUID: 1.0}
    assert rt.responses[1].citations == {UUID2: 1.0}


def test_convert_truncates_to_max_citations():
    # a sentence citing 5 docs at different confidences; rag26 caps citations at 3
    docs = {f"shard_0000{i}_{i}": float(100 - i * 10) for i in range(5)}
    src = Report(metadata=ReportMetaData(team_id="T", topic_id="1", run_id="r", run_desc="d"),
                 responses=[RagtimeReportSentence(text="x", citations=docs)],
                 references=list(docs))
    rag = convert(src, SPECS["rag26"])
    assert len(rag.responses[0].citations) == 3          # truncated to the cap
    assert len(rag.references) == 3                        # references only the kept 3
    # kept the top-3 by confidence (100, 90, 80), references built AFTER truncation
    assert rag.references == ["shard_00000_0", "shard_00001_1", "shard_00002_2"]


def test_converters_respect_explicit_spec():
    src = rag26_report()
    assert all(isinstance(s, Rag24ReportSentence) for s in convert(src, SPECS["rag26"]).responses)
    assert all(isinstance(s, RagtimeReportSentence) for s in convert(src, SPECS["ragtime26"]).responses)


# --- API surface: string specs, Report.convert, deprecations --------------------

def test_verify_check_convert_accept_track_id_string():
    r = rag26_report()
    assert r.verify("rag26") is True                 # string resolves via SPECS
    assert r.check("rag26") == []
    assert convert(r, "ragtime26").responses          # convert accepts a string too
    bad = rag26_report()
    bad.metadata.run_desc = ""
    with pytest.raises(RuntimeError, match="run_desc"):
        bad.verify("rag26")


def test_report_convert_method():
    rt = rag26_report().convert("ragtime26")
    assert all(isinstance(s, RagtimeReportSentence) for s in rt.responses)


def test_to_rag_to_ragtime_deprecated():
    with pytest.warns(DeprecationWarning):
        to_rag(rag26_report())
    with pytest.warns(DeprecationWarning):
        to_ragtime(rag26_report())


def test_verification_error_is_runtimeerror_subclass():
    from autojudge_base.report_spec_verification import TrackSpecVerificationError
    assert issubclass(TrackSpecVerificationError, RuntimeError)
    bad = rag26_report()
    bad.metadata.run_desc = ""
    with pytest.raises(TrackSpecVerificationError):
        bad.verify("rag26")


# --- metadata imposition by the converters --------------------------------------

def test_converter_maps_topic_id_to_target_field():
    # ragtime source uses topic_id; converting to a RAG spec must expose it as narrative_id
    rag = convert(ragtime_report(), "rag26")
    assert rag.metadata.narrative_id == "300"
    assert rag.metadata.topic_id == "300"       # kept in sync


def test_converter_carries_target_mandatory_and_drops_others():
    # rag26 source has run_desc; ragtime26 lists run_desc so it is carried, and the
    # RAG-only 'narrative' is dropped (ragtime26 does not list it)
    rt = rag26_report().convert("ragtime26")
    assert rt.metadata.topic_id == "1"
    assert rt.metadata.run_desc == "desc"       # in ragtime26.mandatory_metadata -> carried
    assert rt.metadata.narrative is None        # not a ragtime26 field -> dropped


def test_converter_leaves_unavailable_required_field_none():
    # ragtime source has no 'type'/'narrative'; converting to rag25 (which requires them)
    # leaves them None so a later verify(rag25) will flag the gap
    rag = convert(ragtime_report(), "rag25")
    assert rag.metadata.narrative_id == "300"
    assert rag.metadata.type is None
    assert rag.metadata.narrative is None
    # rag25 requires 'type' (and 'narrative'); the first missing one is flagged
    with pytest.raises(RuntimeError, match="type"):
        rag.verify(SPECS["rag25"])
