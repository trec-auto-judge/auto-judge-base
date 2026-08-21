"""Tests for the ragtime26 nugget bank format and its conversions."""

import json
import tempfile
from pathlib import Path

import pytest

from autojudge_base.nugget_data import (
    AggregatorType, Answer, NuggetBank, NuggetClaim, NuggetQuestion, Reference,
    RAGTIME26_FORMAT_VERSION,
    Ragtime26Answer, Ragtime26Nugget, Ragtime26NuggetBank, Ragtime26NuggetBanks,
    Ragtime26NuggetMetadata,
    load_ragtime26_nugget_banks_from_file, load_ragtime26_nugget_banks_from_directory,
    write_ragtime26_nugget_banks,
)


SUBMISSION_LINE = {
    "metadata": {
        "team_id": "example-team",
        "topic_id": "topic-1",
        "run_id": "example-run",
        "run_desc": "example run description",
    },
    "nugget_bank": [
        {
            "question": "First example question?",
            "aggregator_type": "AND",
            "answers": [
                {"answer": "example answer", "references": ["doc-1", "doc-2"]},
                {"answer": "other answer", "references": []},
            ],
        },
        {
            "question": "Second example question?",
            "aggregator_type": "OR",
            "answers": [{"answer": "second answer", "references": ["doc-3"]}],
        },
    ],
}


def build_ragtime26_nugget_bank() -> Ragtime26NuggetBank:
    """Parse the sample submission line into a Ragtime26NuggetBank."""
    return Ragtime26NuggetBank.model_validate(SUBMISSION_LINE)


# ============ parsing the submission shape ============


def test_parse_submission_line():
    """A submission line parses into nuggets, answers and references."""
    nugget_bank = build_ragtime26_nugget_bank()

    assert nugget_bank.format_version == RAGTIME26_FORMAT_VERSION
    assert nugget_bank.metadata.team_id == "example-team"
    assert nugget_bank.metadata.run_id == "example-run"
    assert len(nugget_bank.nuggets_as_list()) == 2

    first = nugget_bank.nuggets_as_list()[0]
    assert first.question == "First example question?"
    assert first.aggregator_type == "AND"
    assert len(first.answers) == 2
    assert first.answers[0].references == ["doc-1", "doc-2"]


def test_query_id_is_topic_id():
    """query_id is a property over metadata.topic_id."""
    nugget_bank = build_ragtime26_nugget_bank()
    assert nugget_bank.query_id == "topic-1"
    assert nugget_bank.query_id == nugget_bank.metadata.topic_id


def test_unknown_metadata_fields_are_preserved():
    """The spec allows other metadata fields; they must survive a round-trip."""
    line = json.loads(json.dumps(SUBMISSION_LINE))
    line["metadata"]["submitted_at"] = "2026-01-01"

    nugget_bank = Ragtime26NuggetBank.model_validate(line)
    dumped = json.loads(nugget_bank.model_dump_json(exclude_none=True))

    assert dumped["metadata"]["submitted_at"] == "2026-01-01"


def test_team_id_and_run_id_are_optional():
    """Only topic_id is needed to key a nugget bank."""
    metadata = Ragtime26NuggetMetadata.model_validate({"topic_id": "topic-1"})
    assert metadata.team_id is None
    assert metadata.run_id is None


def test_missing_topic_id_is_rejected():
    """topic_id is what the container keys on, so it is required."""
    with pytest.raises(Exception):
        Ragtime26NuggetMetadata.model_validate({"team_id": "example-team"})


def test_aggregator_type_restricted_to_and_or():
    """The spec allows AND and OR only."""
    assert Ragtime26Nugget(question="Q?", aggregator_type="OR").aggregator_type == "OR"

    with pytest.raises(Exception):
        Ragtime26Nugget(question="Q?", aggregator_type="SUM")


def test_references_accept_str_and_reference():
    """references are doc ids today, Reference objects once offsets are wanted."""
    answer = Ragtime26Answer.model_validate({
        "answer": "example answer",
        "references": ["doc-1", {"doc_id": "doc-2", "collection": "example-corpus"}],
    })

    assert isinstance(answer.references[0], str)
    assert isinstance(answer.references[1], Reference)
    assert answer.reference_doc_ids() == ["doc-1", "doc-2"]


def test_references_are_not_normalized():
    """A bare doc id stays a bare doc id, so a nugget bank round-trips unchanged."""
    nugget_bank = build_ragtime26_nugget_bank()
    dumped = json.loads(nugget_bank.model_dump_json(exclude_none=True))

    assert dumped["nugget_bank"][0]["answers"][0]["references"] == ["doc-1", "doc-2"]


def test_nugget_order_is_preserved():
    """Nuggets are unordered per the spec, but a round-trip keeps what was submitted."""
    nugget_bank = build_ragtime26_nugget_bank()
    questions = [n.question for n in nugget_bank.nuggets_as_list()]
    assert questions == ["First example question?", "Second example question?"]


def test_importance_and_metadata_are_carried():
    """Questions carry an optional importance and an open metadata dict."""
    nugget = Ragtime26Nugget(
        question="Q?", importance="VITAL", metadata={"origin": "example"}
    )
    assert nugget.importance == "VITAL"
    assert nugget.metadata == {"origin": "example"}


# ============ conversion to ragtime25 (v3) ============


def test_to_nugget_bank():
    """Converting to ragtime25 keys nuggets by question text and answers by hash."""
    ragtime25 = build_ragtime26_nugget_bank().to_nugget_bank()

    assert isinstance(ragtime25, NuggetBank)
    assert ragtime25.query_id == "topic-1"
    assert set(ragtime25.nugget_bank) == {
        "First example question?", "Second example question?"
    }

    question = ragtime25.nugget_bank["First example question?"]
    assert question.aggregator_type == AggregatorType.AND
    assert len(question.answers) == 2


def test_to_nugget_bank_normalizes_references():
    """Bare doc ids become Reference objects on the ragtime25 side."""
    ragtime25 = build_ragtime26_nugget_bank().to_nugget_bank()

    question = ragtime25.nugget_bank["First example question?"]
    answer = next(a for a in question.answers.values() if a.answer == "example answer")
    assert [r.doc_id for r in answer.references] == ["doc-1", "doc-2"]


def test_to_nugget_bank_keeps_run_metadata():
    """Run metadata survives into the ragtime25 bank's metadata dict."""
    ragtime25 = build_ragtime26_nugget_bank().to_nugget_bank()

    assert ragtime25.metadata["team_id"] == "example-team"
    assert ragtime25.metadata["run_id"] == "example-run"


def test_to_nugget_bank_carries_importance_and_metadata():
    nugget_bank = Ragtime26NuggetBank(
        metadata=Ragtime26NuggetMetadata(topic_id="topic-1"),
        nugget_bank=[Ragtime26Nugget(
            question="Q?", importance=1, metadata={"origin": "example"}
        )],
    )

    question = nugget_bank.to_nugget_bank().nugget_bank["Q?"]
    assert question.importance == 1
    assert question.metadata == {"origin": "example"}


# ============ conversion back to ragtime26 ============


def test_to_ragtime26_nugget_bank_round_trip():
    """ragtime26 -> ragtime25 -> ragtime26 preserves questions, answers and references."""
    original = build_ragtime26_nugget_bank()
    round_tripped = original.to_nugget_bank().to_ragtime26_nugget_bank()

    assert round_tripped.query_id == original.query_id
    assert round_tripped.metadata.team_id == "example-team"

    by_question = {n.question: n for n in round_tripped.nuggets_as_list()}
    assert set(by_question) == {"First example question?", "Second example question?"}

    first = by_question["First example question?"]
    assert first.aggregator_type == "AND"
    assert sorted(a.answer for a in first.answers) == ["example answer", "other answer"]

    example = next(a for a in first.answers if a.answer == "example answer")
    assert example.reference_doc_ids() == ["doc-1", "doc-2"]


def test_to_ragtime26_nugget_bank_topic_id_falls_back_to_query_id():
    """A ragtime25 bank without run metadata still converts, keyed by query_id."""
    ragtime25 = NuggetBank(query_id="topic-1", title_query="Example topic")
    ragtime25.add_nuggets(NuggetQuestion.from_lazy("topic-1", "Q?", ["A"]))

    converted = ragtime25.to_ragtime26_nugget_bank()

    assert converted.metadata.topic_id == "topic-1"
    assert converted.metadata.team_id is None


def test_to_ragtime26_nugget_bank_rejects_claims():
    """ragtime26 has no claim_bank, so claims must not be silently dropped."""
    ragtime25 = NuggetBank(query_id="topic-1", title_query="Example topic")
    ragtime25.add_nuggets(NuggetClaim.from_lazy("topic-1", "An example claim"))

    with pytest.raises(ValueError, match="no claim_bank"):
        ragtime25.to_ragtime26_nugget_bank()


def test_to_ragtime26_nugget_bank_rejects_unsupported_aggregator():
    """Only AND/OR can be expressed in ragtime26."""
    ragtime25 = NuggetBank(query_id="topic-1", title_query="Example topic")
    ragtime25.add_nuggets(NuggetQuestion(
        query_id="topic-1", question="Q?", aggregator_type=AggregatorType.SUM
    ))

    with pytest.raises(ValueError, match="not one of AND/OR"):
        ragtime25.to_ragtime26_nugget_bank()


def test_to_ragtime26_nugget_bank_treats_default_aggregator_as_unspecified():
    ragtime25 = NuggetBank(query_id="topic-1", title_query="Example topic")
    ragtime25.add_nuggets(NuggetQuestion(
        query_id="topic-1", question="Q?", aggregator_type=AggregatorType.Default
    ))

    converted = ragtime25.to_ragtime26_nugget_bank()
    assert converted.nuggets_as_list()[0].aggregator_type is None


# ============ container ============


def test_from_banks_list_keys_by_topic_id():
    bank1 = Ragtime26NuggetBank(metadata=Ragtime26NuggetMetadata(topic_id="topic-1"))
    bank2 = Ragtime26NuggetBank(metadata=Ragtime26NuggetMetadata(topic_id="topic-2"))

    nugget_banks = Ragtime26NuggetBanks.from_banks_list([bank1, bank2])

    assert set(nugget_banks.banks) == {"topic-1", "topic-2"}
    assert nugget_banks.format_version == RAGTIME26_FORMAT_VERSION


def test_from_banks_list_duplicate_error():
    bank1 = Ragtime26NuggetBank(metadata=Ragtime26NuggetMetadata(topic_id="topic-1"))
    bank2 = Ragtime26NuggetBank(metadata=Ragtime26NuggetMetadata(topic_id="topic-1"))

    with pytest.raises(ValueError, match="Duplicate topic_id"):
        Ragtime26NuggetBanks.from_banks_list([bank1, bank2])

    nugget_banks = Ragtime26NuggetBanks.from_banks_list([bank1, bank2], overwrite=True)
    assert len(nugget_banks.banks) == 1


def test_get_bank_model():
    assert Ragtime26NuggetBanks.get_bank_model() is Ragtime26NuggetBank


# ============ I/O ============


def test_write_read_jsonl():
    """Write and read back through the curried io functions."""
    nugget_banks = Ragtime26NuggetBanks.from_banks_list([build_ragtime26_nugget_bank()])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nugget_banks.jsonl"
        write_ragtime26_nugget_banks(nugget_banks, path, format="jsonl")

        loaded = load_ragtime26_nugget_banks_from_file(path)

        assert set(loaded.banks) == {"topic-1"}
        reloaded = loaded.banks["topic-1"]
        assert len(reloaded.nuggets_as_list()) == 2
        assert reloaded.nuggets_as_list()[0].answers[0].references == ["doc-1", "doc-2"]


def test_write_read_directory():
    nugget_banks = Ragtime26NuggetBanks.from_banks_list([build_ragtime26_nugget_bank()])

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "nugget_banks"
        write_ragtime26_nugget_banks(nugget_banks, out_dir, format="directory")

        assert (out_dir / "topic-1.json.gz").exists()

        loaded = load_ragtime26_nugget_banks_from_directory(out_dir)
        assert set(loaded.banks) == {"topic-1"}


# ============ protocol compliance and verification ============


def test_protocol_compliance():
    """Ragtime26 models satisfy the nugget bank protocols, like the other formats."""
    from autojudge_base.nugget_data.protocols import NuggetBankProtocol, NuggetBanksProtocol

    assert isinstance(Ragtime26NuggetBanks(banks={}), NuggetBanksProtocol)

    nugget_bank = Ragtime26NuggetBank(metadata=Ragtime26NuggetMetadata(topic_id="topic-1"))
    assert isinstance(nugget_bank, NuggetBankProtocol)
    assert nugget_bank.query_id == "topic-1"


def test_verification_passes():
    from autojudge_base.nugget_data import NuggetBanksVerification

    nugget_banks = Ragtime26NuggetBanks.from_banks_list([build_ragtime26_nugget_bank()])
    NuggetBanksVerification(nugget_banks, ["topic-1"]).all()


def test_verification_reports_empty_nugget_bank():
    from autojudge_base.nugget_data import (
        NuggetBanksVerification, NuggetBanksVerificationError
    )

    empty = Ragtime26NuggetBank(metadata=Ragtime26NuggetMetadata(topic_id="topic-1"))
    nugget_banks = Ragtime26NuggetBanks.from_banks_list([empty])

    with pytest.raises(NuggetBanksVerificationError, match="Empty nugget banks.*topic-1"):
        NuggetBanksVerification(nugget_banks, ["topic-1"]).non_empty_banks()
