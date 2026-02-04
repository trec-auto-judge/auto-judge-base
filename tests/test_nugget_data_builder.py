"""Tests for NuggetBank and NuggetBanks builders."""

import tempfile
from pathlib import Path

from trec_auto_judge.nugget_data import (
    Creator, NuggetQuestion, AggregatorType, Answer, NuggetBank, NuggetClaim,
    Offsets, Reference, load_nugget_bank_json, print_nugget_json,
    nugget_to_dict, question_nugget_to_dict, write_nugget_bank_json,
    merge_nugget_claims, merge_nugget_questions
)
from trec_auto_judge.nugget_data import (
    NuggetBanks, load_nugget_banks_from_file, load_nugget_banks_from_directory,
    write_nugget_banks
)


def build_nugget_bank() -> NuggetBank:
    """Build a sample NuggetBank with questions and claims."""
    nuggets = [
        NuggetQuestion(
            question="What modern conservation efforts are in place to preserve Machu Picchu?",
            question_id="1",
            query_id="1",
            sub_nuggets=[
                NuggetQuestion(
                    query_id="1",
                    question="How are visitor impacts limited?"
                ).add_answers([
                    Answer(answer="restricted numbers"),
                    Answer(answer="daily quotas")
                ]),
                NuggetQuestion(
                    query_id="1",
                    question="Which organization sets preservation guidelines?"
                ).add_answers([Answer(answer="UNESCO")])
            ]
        ).add_answers([
            Answer(answer="visitor numbers are restricted to reduce wear"),
            Answer(answer="UNESCO provides guidelines for preservation"),
            Answer(answer="restoration projects use traditional materials and methods")
        ]).add_creator(Creator(is_human=True, contact=["Laura"])),

        NuggetQuestion(
            question="How did the Incas likely transport the massive stones used at Machu Picchu?",
            question_id="2",
            query_id="1",
            sub_nuggets=[
                NuggetQuestion(
                    query_id="1",
                    question="Which cylindrical timbers helped move stones?"
                ).add_answers([Answer(answer="wooden rollers")]),
                NuggetQuestion(
                    query_id="1",
                    question="Which simple machine was used to pry stones?"
                ).add_answers([Answer(answer="levers")])
            ],
            aggregator_type=AggregatorType.OR,
            importance=1,
            creator=[Creator(is_human=True, contact=["Laura"])],
            metadata={"creator": "Laura", "version": "0.0.1"}
        ),

        # example with answer predicates
        NuggetQuestion(
            question="Where is Machu Picchu located?",
            query_id="1",
            aggregator_type=AggregatorType.AND
        ).add_answers([
            Answer(answer="in Peru"),
            Answer(answer="in South America")
        ]).add_creator(Creator(is_human=True, contact=["Laura"])),

        # example with references
        NuggetQuestion(
            query_id="1",
            question="When was Machu Picchu built?",
            aggregator_type=AggregatorType.OR
        ).add_answers([
            Answer.from_lazy(
                answer="15th Century",
                references=["03b53809-fbee-46aa-b036-8934a2b1f556",
                           "0e589240-39b6-49ca-b82f-84f7c55d21f5",
                           "17770e9d-4865-48e5-9140-ffe85401c9d6"]
            )
        ]).add_creator(Creator(is_human=True, contact=["Laura"])),

        NuggetQuestion(
            query_id="1",
            question="Is machu pichu the right name?"
        ).add_answers([
            Answer(answer="yes"),
            Answer(answer="no")
        ]).add_creator(Creator(is_human=True, contact=["Laura"])),

        # Claims (using simplified interface - no related_question_text/related_answer)
        NuggetClaim(
            query_id="1",
            claim="Machu Picchu was built in the 15th century",
            references=[Reference(
                doc_id="123",
                collection="historical_docs",
                offsets=Offsets(start_offset=0, end_offset=10, encoding="UTF-8")
            )]
        ),

        NuggetClaim.from_lazy(
            query_id="1",
            claim="The site was never discovered by Spanish conquistadors",
            references=["doc-456", "doc-789"]
        ),

        # minimal example
        NuggetQuestion.from_lazy(
            "1",
            "Is machu pichu the right name?",
            ["yes", "no"],
            creator=Creator(is_human=True, contact=["Laura"])
        )
    ]

    # backwards compatible to v1 style dictionary
    print("nuggets as v1 style dictionaries")
    for nugget in nuggets:
        print(nugget_to_dict(nugget))

    print("add nuggets")
    nuggetBank = NuggetBank(query_id="123", title_query="Laura Pichu")
    nuggetBank.add_nuggets(nuggets)

    print(nuggetBank)

    nuggetBank.index_nuggets()

    print("question roots (nugget_bank)")
    if nuggetBank.nugget_bank:
        print("\n".join(nuggetBank.nugget_bank.keys()))

    print("claim roots (claim_bank)")
    if nuggetBank.claim_bank:
        print("\n".join(nuggetBank.claim_bank.keys()))

    print("all (flattened view)")
    print("\n".join(nuggetBank.all_nuggets_view.keys()))

    return nuggetBank


def test_build_nuggets():
    """Test building and serializing NuggetBank."""
    nuggetBank = build_nugget_bank()
    print_nugget_json(nuggetBank)
    print(nuggetBank.model_dump_json(indent=2, exclude_none=True))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nugget_bank_example.json.gz"
        write_nugget_bank_json(nuggetBank, path)
        loaded = load_nugget_bank_json(path)
        assert loaded.query_id == nuggetBank.query_id
        assert len(loaded.nuggets_as_list()) == len(nuggetBank.nuggets_as_list())


def test_nugget_bank_separate_storage():
    """Test that questions go to nugget_bank and claims go to claim_bank."""
    bank = NuggetBank(query_id="test", title_query="Test Query")

    question = NuggetQuestion.from_lazy("test", "What is X?", ["answer"])
    claim = NuggetClaim.from_lazy("test", "X is true")

    bank.add_nuggets([question, claim])

    # Check separate storage
    assert bank.nugget_bank is not None
    assert bank.claim_bank is not None
    assert "What is X?" in bank.nugget_bank
    assert "X is true" in bank.claim_bank

    # Check combined list
    all_nuggets = bank.nuggets_as_list()
    assert len(all_nuggets) == 2

    # Check flattened view includes both
    assert "What is X?" in bank.all_nuggets_view
    assert "X is true" in bank.all_nuggets_view


def test_nugget_bank_read_only_view():
    """Test that all_nuggets_view is read-only."""
    bank = NuggetBank(query_id="test", title_query="Test Query")
    bank.add_nuggets(NuggetQuestion.from_lazy("test", "Q?", ["A"]))

    # Should be able to read
    assert "Q?" in bank.all_nuggets_view

    # Should not be able to modify
    try:
        bank.all_nuggets_view["new_key"] = NuggetQuestion.from_lazy("test", "New?")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass  # Expected


def test_iterate_nugget_bank():
    """Test iterating over questions and claims with answers."""
    nuggetBank = build_nugget_bank()

    for nugget in nuggetBank.nuggets_as_list():
        if isinstance(nugget, NuggetQuestion):
            if nugget.answers is not None:
                for answer_text, answer in nugget.answers.items():
                    print(f"id={nugget.question_id}: question={nugget.question} "
                          f"answer={answer.answer} nugget.references={nugget.references} "
                          f"answer.references={answer.references}")
            else:
                print(f"id={nugget.question_id}: question={nugget.question} "
                      f"nugget.references={nugget.references}")
        elif isinstance(nugget, NuggetClaim):
            print(f"id={nugget.claim_id}: claim={nugget.claim} "
                  f"references={nugget.references}")


def test_build_creator():
    """Test that creators are preserved."""
    nuggetBank = build_nugget_bank()

    for nugget in nuggetBank.nugget_bank.values():
        if any("Laura" in c.contact for c in (nugget.creator or []) if c.contact):
            return  # Found Laura
    assert False, "Should have found creator Laura"


def test_claim_from_lazy():
    """Test NuggetClaim.from_lazy factory method."""
    claim = NuggetClaim.from_lazy(
        query_id="q1",
        claim="The sky is blue",
        references=["doc1", "doc2"],
        creator=Creator(is_human=True, contact=["Test"])
    )

    assert claim.query_id == "q1"
    assert claim.claim == "The sky is blue"
    assert claim.claim_id is not None  # Auto-generated
    assert len(claim.references) == 2
    assert claim.references[0].doc_id == "doc1"
    assert claim.creator is not None


def test_merge_nugget_claims():
    """Test merging multiple claims."""
    claim1 = NuggetClaim.from_lazy("q1", "Fact A", references=["doc1"])
    claim2 = NuggetClaim.from_lazy("q1", "Fact A", references=["doc2"])

    merged = merge_nugget_claims([claim1, claim2])

    assert merged.claim == "Fact A"
    assert len(merged.references) == 2
    assert {r.doc_id for r in merged.references} == {"doc1", "doc2"}


def test_nugget_to_dict_question():
    """Test nugget_to_dict for questions."""
    question = NuggetQuestion.from_lazy(
        "q1", "What is X?",
        gold_answers=["A1", "A2"],
        references=["source-doc"]
    )

    d = nugget_to_dict(question)

    assert d["query_id"] == "q1"
    assert d["question_text"] == "What is X?"
    assert "question_id" in d
    assert len(d["gold_answers"]) == 2
    assert "references" in d  # Question-level references
    assert d["references"] == ["source-doc"]


def test_nugget_to_dict_claim():
    """Test nugget_to_dict for claims."""
    claim = NuggetClaim.from_lazy("q1", "X is true", references=["doc1"])

    d = nugget_to_dict(claim)

    assert d["query_id"] == "q1"
    assert d["claim_text"] == "X is true"
    assert "claim_id" in d
    assert "references" in d
    assert d["references"] == ["doc1"]


def test_nugget_to_dict_backwards_compat():
    """Test that question_nugget_to_dict is an alias for nugget_to_dict."""
    question = NuggetQuestion.from_lazy("q1", "What?", ["A"])

    assert nugget_to_dict(question) == question_nugget_to_dict(question)


# ============ NuggetBanks (multi-topic container) tests ============


def test_nugget_banks_empty():
    """Test creating empty NuggetBanks container."""
    banks = NuggetBanks(banks={})
    assert len(banks.banks) == 0
    assert list(banks.banks) == []


def test_nugget_banks_from_banks_list():
    """Test creating NuggetBanks from a list."""
    bank1 = NuggetBank(query_id="topic-1", title_query="Topic 1")
    bank1.add_nuggets(NuggetQuestion.from_lazy("topic-1", "Q1?", ["A1"]))

    bank2 = NuggetBank(query_id="topic-2", title_query="Topic 2")
    bank2.add_nuggets(NuggetQuestion.from_lazy("topic-2", "Q2?", ["A2"]))

    banks = NuggetBanks.from_banks_list([bank1, bank2])

    assert len(banks.banks) == 2
    assert "topic-1" in banks.banks
    assert "topic-2" in banks.banks
    assert banks.banks["topic-1"].title_query == "Topic 1"
    assert list(banks.banks.keys()) == ["topic-1", "topic-2"]


def test_nugget_banks_direct_access():
    """Test direct access to banks field."""
    bank = NuggetBank(query_id="q1", title_query="Query 1")
    banks = NuggetBanks.from_banks_list([bank])

    # Direct dict access
    assert banks.banks["q1"] is not None
    assert banks.banks.get("nonexistent") is None

    # Membership
    assert "q1" in banks.banks
    assert "q2" not in banks.banks

    # Iteration
    assert list(banks.banks) == ["q1"]

    # Length
    assert len(banks.banks) == 1

    # Get with default
    assert banks.banks.get("q1") is not None
    assert banks.banks.get("missing", "default") == "default"


def test_nugget_banks_duplicate_error():
    """Test that from_banks_list raises on duplicate query_id by default."""
    bank1 = NuggetBank(query_id="q1", title_query="Query 1")
    bank1.add_nuggets(NuggetQuestion.from_lazy("q1", "Question A?", ["A"]))

    bank2 = NuggetBank(query_id="q1", title_query="Query 1")
    bank2.add_nuggets(NuggetQuestion.from_lazy("q1", "Question B?", ["B"]))

    # Should raise on duplicate
    import pytest
    with pytest.raises(ValueError, match="Duplicate query_id"):
        NuggetBanks.from_banks_list([bank1, bank2])

    # With overwrite=True, should succeed (last wins)
    banks = NuggetBanks.from_banks_list([bank1, bank2], overwrite=True)
    assert len(banks.banks) == 1
    assert banks.banks["q1"] is not None


def test_nugget_banks_write_read_jsonl():
    """Test writing and reading NuggetBanks in JSONL format."""
    from trec_auto_judge.nugget_data import NuggetBank, NuggetBanks

    bank1 = NuggetBank(query_id="t1", title_query="Topic 1")
    bank1.add_nuggets(NuggetQuestion.from_lazy("t1", "Q1?", ["A1"]))

    bank2 = NuggetBank(query_id="t2", title_query="Topic 2")
    bank2.add_nuggets(NuggetClaim.from_lazy("t2", "Claim 2"))

    banks = NuggetBanks.from_banks_list([bank1, bank2])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nuggets.jsonl"
        write_nugget_banks(banks, path, format="jsonl")

        loaded = load_nugget_banks_from_file(path)

        assert len(loaded.banks) == 2
        assert "t1" in loaded.banks
        assert "t2" in loaded.banks


def test_nugget_banks_write_read_jsonl_gz():
    """Test writing and reading compressed JSONL."""
    from trec_auto_judge.nugget_data import NuggetBank, NuggetBanks, load_nugget_banks_from_file

    bank = NuggetBank(query_id="t1", title_query="Topic 1")
    bank.add_nuggets(NuggetQuestion.from_lazy("t1", "Q?", ["A"]))
    banks = NuggetBanks.from_banks_list([bank])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nuggets.jsonl.gz"
        write_nugget_banks(banks, path)

        loaded = load_nugget_banks_from_file(path)
        assert len(loaded.banks) == 1


def test_nugget_banks_write_read_directory():
    """Test writing and reading from directory format."""
    from trec_auto_judge.nugget_data import NuggetBank, NuggetBanks, load_nugget_banks_from_directory

    bank1 = NuggetBank(query_id="topic-1", title_query="Topic 1")
    bank1.add_nuggets(NuggetQuestion.from_lazy("topic-1", "Q1?", ["A1"]))

    bank2 = NuggetBank(query_id="topic-2", title_query="Topic 2")
    bank2.add_nuggets(NuggetQuestion.from_lazy("topic-2", "Q2?", ["A2"]))

    banks = NuggetBanks.from_banks_list([bank1, bank2])

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "nuggets"
        write_nugget_banks(banks, out_dir, format="directory")

        # Check files were created
        assert (out_dir / "topic-1.json.gz").exists()
        assert (out_dir / "topic-2.json.gz").exists()

        # Load back
        loaded = load_nugget_banks_from_directory(out_dir)
        assert len(loaded.banks) == 2
        assert "topic-1" in loaded.banks
        assert "topic-2" in loaded.banks


def test_backwards_compat_old_claim_fields():
    """Test that old NuggetClaim fields are ignored (backwards compatibility)."""
    # Simulate loading old JSON with deprecated fields
    old_data = {
        "query_id": "q1",
        "claim": "Old claim",
        "related_question_text": "Should be ignored",
        "related_answer": [{"answer": "Should be ignored"}]
    }

    # Should not raise - extra fields are ignored
    claim = NuggetClaim.model_validate(old_data)
    assert claim.claim == "Old claim"
    assert claim.query_id == "q1"
    # Old fields should not be present
    assert not hasattr(claim, "related_question_text") or claim.__dict__.get("related_question_text") is None


# ============ Protocol-based I/O tests ============

def test_import_nugget_banks_type():
    """Test dynamic import of NuggetBanks container types."""
    from trec_auto_judge.nugget_data.io import import_nugget_banks_type

    # Import NuggetBanks
    nb_type = import_nugget_banks_type("trec_auto_judge.nugget_data.NuggetBanks")
    assert nb_type is NuggetBanks

    # Import NuggetizerNuggetBanks
    from trec_auto_judge.nugget_data import NuggetizerNuggetBanks
    nnb_type = import_nugget_banks_type("trec_auto_judge.nugget_data.NuggetizerNuggetBanks")
    assert nnb_type is NuggetizerNuggetBanks


def test_import_nugget_banks_type_invalid():
    """Test that invalid import paths raise appropriate errors."""
    from trec_auto_judge.nugget_data.io import import_nugget_banks_type
    import pytest

    # Non-existent module
    with pytest.raises(ModuleNotFoundError):
        import_nugget_banks_type("nonexistent.module.Class")

    # Non-existent class
    with pytest.raises(AttributeError):
        import_nugget_banks_type("trec_auto_judge.nugget_data.NonExistentClass")


def test_load_nugget_banks_generic():
    """Test protocol-based generic loading."""
    from trec_auto_judge.nugget_data import (
        NuggetBanks, NuggetBank, NuggetQuestion,
        load_nugget_banks_generic, write_nugget_banks_generic
    )

    # Create test data
    bank = NuggetBank(query_id="test-1", title_query="Test Topic")
    bank.add_nuggets(NuggetQuestion.from_lazy("test-1", "What is X?", ["Answer"]))
    banks = NuggetBanks.from_banks_list([bank])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "generic_test.jsonl"

        # Write using generic function
        write_nugget_banks_generic(banks, path)

        # Load using generic function
        loaded = load_nugget_banks_generic(path, NuggetBanks)

        assert len(loaded.banks) == 1
        assert "test-1" in loaded.banks
        assert loaded.banks["test-1"].title_query == "Test Topic"


def test_write_nugget_banks_generic_directory():
    """Test protocol-based generic writing to directory."""
    from trec_auto_judge.nugget_data import (
        NuggetBanks, NuggetBank, NuggetQuestion,
        write_nugget_banks_generic, load_nugget_banks_from_directory_generic
    )

    # Create test data with multiple topics
    bank1 = NuggetBank(query_id="t1", title_query="Topic 1")
    bank1.add_nuggets(NuggetQuestion.from_lazy("t1", "Q1?", ["A1"]))

    bank2 = NuggetBank(query_id="t2", title_query="Topic 2")
    bank2.add_nuggets(NuggetQuestion.from_lazy("t2", "Q2?", ["A2"]))

    banks = NuggetBanks.from_banks_list([bank1, bank2])

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "nuggets_dir"

        # Write using generic function with directory format
        write_nugget_banks_generic(banks, out_dir, format="directory")

        # Check files created
        assert (out_dir / "t1.json.gz").exists()
        assert (out_dir / "t2.json.gz").exists()

        # Load using generic function
        loaded = load_nugget_banks_from_directory_generic(out_dir, NuggetBanks)
        assert len(loaded.banks) == 2


# ============ Protocol compliance tests ============

def test_nugget_banks_protocol_compliance():
    """Test that NuggetBanks satisfies NuggetBanksProtocol."""
    from trec_auto_judge.nugget_data import NuggetBanks, NuggetBank
    from trec_auto_judge.nugget_data.protocols import NuggetBanksProtocol, NuggetBankProtocol

    # Check protocol compliance
    assert isinstance(NuggetBanks(banks={}), NuggetBanksProtocol)

    # Check get_bank_model() method
    assert hasattr(NuggetBanks, "get_bank_model")
    assert NuggetBanks.get_bank_model() is NuggetBank

    # Check NuggetBank protocol compliance
    bank = NuggetBank(query_id="test", title_query="Test")
    assert isinstance(bank, NuggetBankProtocol)
    assert bank.query_id == "test"


def test_nuggetizer_nugget_banks_protocol_compliance():
    """Test that NuggetizerNuggetBanks satisfies NuggetBanksProtocol."""
    from trec_auto_judge.nugget_data import NuggetizerNuggetBanks, NuggetizerNuggetBank
    from trec_auto_judge.nugget_data.protocols import NuggetBanksProtocol, NuggetBankProtocol

    # Check protocol compliance
    assert isinstance(NuggetizerNuggetBanks(banks={}), NuggetBanksProtocol)

    # Check get_bank_model() method
    assert hasattr(NuggetizerNuggetBanks, "get_bank_model")
    assert NuggetizerNuggetBanks.get_bank_model() is NuggetizerNuggetBank

    # Check NuggetizerNuggetBank protocol compliance
    bank = NuggetizerNuggetBank(qid="test", query="Test Query")
    assert isinstance(bank, NuggetBankProtocol)
    assert bank.query_id == "test"  # Uses property


def test_nuggetizer_nugget_banks_io():
    """Test I/O for NuggetizerNuggetBanks using generic functions."""
    from trec_auto_judge.nugget_data import (
        NuggetizerNuggetBanks, NuggetizerNuggetBank,
        load_nugget_banks_generic, write_nugget_banks_generic
    )

    # Create test data
    bank = NuggetizerNuggetBank(qid="nuggetizer-1", query="Nuggetizer Query")
    banks = NuggetizerNuggetBanks.from_banks_list([bank])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nuggetizer.jsonl"

        # Write
        write_nugget_banks_generic(banks, path)

        # Read back
        loaded = load_nugget_banks_generic(path, NuggetizerNuggetBanks)

        assert len(loaded.banks) == 1
        assert "nuggetizer-1" in loaded.banks
        assert loaded.banks["nuggetizer-1"].query == "Nuggetizer Query"


# ============ NuggetBanks Verification tests ============

def test_nugget_banks_verification_all_pass():
    """Test that verification passes for complete, valid nugget banks."""
    import pytest
    from trec_auto_judge.nugget_data import (
        NuggetBanks, NuggetBank, NuggetQuestion,
        NuggetBanksVerification
    )
    from trec_auto_judge.request import Request

    # Create topics
    topics = [
        Request(request_id="t1", title="Topic 1"),
        Request(request_id="t2", title="Topic 2"),
    ]
    topic_ids = [t.request_id for t in topics]

    # Create matching nugget banks
    bank1 = NuggetBank(query_id="t1", title_query="Topic 1")
    bank1.add_nuggets(NuggetQuestion.from_lazy("t1", "Q1?", ["A1"]))

    bank2 = NuggetBank(query_id="t2", title_query="Topic 2")
    bank2.add_nuggets(NuggetQuestion.from_lazy("t2", "Q2?", ["A2"]))

    banks = NuggetBanks.from_banks_list([bank1, bank2])

    # Should pass all checks
    NuggetBanksVerification(banks, topic_ids).all()


def test_nugget_banks_verification_complete_topics():
    """Test that verification fails when topics are missing nugget banks."""
    import pytest
    from trec_auto_judge.nugget_data import (
        NuggetBanks, NuggetBank, NuggetQuestion,
        NuggetBanksVerification, NuggetBanksVerificationError
    )
    from trec_auto_judge.request import Request

    # Create 3 topics
    topics = [
        Request(request_id="t1", title="Topic 1"),
        Request(request_id="t2", title="Topic 2"),
        Request(request_id="t3", title="Topic 3"),
    ]
    topic_ids = [t.request_id for t in topics]

    # Only create nugget banks for 2 topics
    bank1 = NuggetBank(query_id="t1", title_query="Topic 1")
    bank1.add_nuggets(NuggetQuestion.from_lazy("t1", "Q1?", ["A1"]))

    bank2 = NuggetBank(query_id="t2", title_query="Topic 2")
    bank2.add_nuggets(NuggetQuestion.from_lazy("t2", "Q2?", ["A2"]))

    banks = NuggetBanks.from_banks_list([bank1, bank2])

    # Should fail on complete_topics check
    with pytest.raises(NuggetBanksVerificationError, match="Missing nugget banks.*t3"):
        NuggetBanksVerification(banks, topic_ids).complete_topics()


def test_nugget_banks_verification_no_extra_topics():
    """Test that verification fails when nugget banks exist for unknown topics."""
    import pytest
    from trec_auto_judge.nugget_data import (
        NuggetBanks, NuggetBank, NuggetQuestion,
        NuggetBanksVerification, NuggetBanksVerificationError
    )
    from trec_auto_judge.request import Request

    # Create 1 topic
    topics = [
        Request(request_id="t1", title="Topic 1"),
    ]
    topic_ids = [t.request_id for t in topics]

    # Create nugget banks for 2 topics (one extra)
    bank1 = NuggetBank(query_id="t1", title_query="Topic 1")
    bank1.add_nuggets(NuggetQuestion.from_lazy("t1", "Q1?", ["A1"]))

    bank_extra = NuggetBank(query_id="extra", title_query="Extra Topic")
    bank_extra.add_nuggets(NuggetQuestion.from_lazy("extra", "Q?", ["A"]))

    banks = NuggetBanks.from_banks_list([bank1, bank_extra])

    # Should fail on no_extra_topics check
    with pytest.raises(NuggetBanksVerificationError, match="unknown topic.*extra"):
        NuggetBanksVerification(banks, topic_ids).no_extra_topics()


def test_nugget_banks_verification_non_empty_banks():
    """Test that verification fails when nugget banks are empty."""
    import pytest
    from trec_auto_judge.nugget_data import (
        NuggetBanks, NuggetBank,
        NuggetBanksVerification, NuggetBanksVerificationError
    )
    from trec_auto_judge.request import Request

    # Create topics
    topics = [
        Request(request_id="t1", title="Topic 1"),
    ]
    topic_ids = [t.request_id for t in topics]

    # Create empty nugget bank (no nuggets added)
    empty_bank = NuggetBank(query_id="t1", title_query="Topic 1")
    banks = NuggetBanks.from_banks_list([empty_bank])

    # Should fail on non_empty_banks check
    with pytest.raises(NuggetBanksVerificationError, match="Empty nugget banks.*t1"):
        NuggetBanksVerification(banks, topic_ids).non_empty_banks()


def test_nugget_banks_verification_chaining():
    """Test that verification methods can be chained."""
    from trec_auto_judge.nugget_data import (
        NuggetBanks, NuggetBank, NuggetQuestion,
        NuggetBanksVerification
    )
    from trec_auto_judge.request import Request

    topics = [Request(request_id="t1", title="Topic 1")]
    topic_ids = [t.request_id for t in topics]
    
    bank = NuggetBank(query_id="t1", title_query="Topic 1")
    bank.add_nuggets(NuggetQuestion.from_lazy("t1", "Q?", ["A"]))
    banks = NuggetBanks.from_banks_list([bank])

    # All methods return self for chaining
    result = NuggetBanksVerification(banks, topic_ids).complete_topics().no_extra_topics().non_empty_banks()
    assert isinstance(result, NuggetBanksVerification)


def test_nugget_banks_verification_fail_fast():
    """Test that verification fails on first error (fail-fast)."""
    import pytest
    from trec_auto_judge.nugget_data import (
        NuggetBanks, NuggetBank,
        NuggetBanksVerification, NuggetBanksVerificationError
    )
    from trec_auto_judge.request import Request

    # Create topics
    topics = [
        Request(request_id="t1", title="Topic 1"),
        Request(request_id="t2", title="Topic 2"),
    ]
    topic_ids = [t.request_id for t in topics]

    # Create empty bank for t1 (missing t2)
    empty_bank = NuggetBank(query_id="t1", title_query="Topic 1")
    banks = NuggetBanks.from_banks_list([empty_bank])

    # all() should fail on complete_topics first (before non_empty_banks)
    with pytest.raises(NuggetBanksVerificationError, match="Missing nugget banks"):
        NuggetBanksVerification(banks, topic_ids).all()


def test_nugget_banks_verification_with_nuggetizer_format():
    """Test verification works with NuggetizerNuggetBanks."""
    from trec_auto_judge.nugget_data import (
        NuggetizerNuggetBanks, NuggetizerNuggetBank,
        NuggetBanksVerification
    )
    from trec_auto_judge.nugget_data.nuggetizer.nuggetizer_data import NuggetizerNugget
    from trec_auto_judge.request import Request

    topics = [Request(request_id="t1", title="Topic 1")]
    topic_ids = [t.request_id for t in topics]
    
    # Create NuggetizerNuggetBank with nuggets
    bank = NuggetizerNuggetBank(qid="t1", query="Topic 1")
    bank.nuggets = [NuggetizerNugget(text="Key fact")]
    banks = NuggetizerNuggetBanks.from_banks_list([bank])

    # Should pass all checks
    NuggetBanksVerification(banks, topic_ids).all()


def test_nugget_banks_verification_with_claims():
    """Test verification recognizes claims as valid nuggets."""
    from trec_auto_judge.nugget_data import (
        NuggetBanks, NuggetBank, NuggetClaim,
        NuggetBanksVerification
    )
    from trec_auto_judge.request import Request

    topics = [Request(request_id="t1", title="Topic 1")]
    topic_ids = [t.request_id for t in topics]
    
    # Create bank with only claims (no questions)
    bank = NuggetBank(query_id="t1", title_query="Topic 1")
    bank.add_nuggets(NuggetClaim.from_lazy("t1", "This is a claim"))
    banks = NuggetBanks.from_banks_list([bank])

    # Should pass - claims count as valid nuggets
    NuggetBanksVerification(banks, topic_ids).all()