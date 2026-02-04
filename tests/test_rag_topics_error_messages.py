import unittest
from click.testing import CliRunner
import click
from pathlib import Path
from trec_auto_judge.click_plus import option_rag_responses, option_rag_topics
from pathlib import Path
from unittest import mock

@click.command()
@option_rag_topics()
def topics_stats(rag_topics):
    print("Topics:",  len(rag_topics))
    return 0

@click.command()
@option_rag_topics()
@option_rag_responses()
def rag_stats2(rag_responses, rag_topics):
    print("Topics:",  len(rag_topics))
    return 0

@click.command()
@option_rag_responses()
@option_rag_topics()
def rag_stats3(rag_topics, rag_responses):
    print("Topics:",  len(rag_topics))
    return 0

RESORUCES_DIR = Path(__file__).parent / "resources"

class TestRagTopicsErrorMessagesIntegration(unittest.TestCase):
    def test_local_file_that_is_empty(self):
        runner = CliRunner()
        result = runner.invoke(topics_stats, ["--rag-topics", str(RESORUCES_DIR / "empty-rag-topics.jsonl")])

        self.assertIn("Usage: topics-stats [OPTIONS]", result.output)
        self.assertIn("empty-rag-topics.jsonl' contains 0 RAG topics.", result.output)

        self.assertEqual(2, result.exit_code)

    def test_local_file_that_is_not_valid(self):
        runner = CliRunner()
        result = runner.invoke(topics_stats, ["--rag-topics", str(RESORUCES_DIR / "example-rag-topics-invalid.jsonl")])

        self.assertIn("Usage: topics-stats [OPTIONS]", result.output)
        self.assertIn("1 validation error for Request", result.output)
        self.assertIn("title\n  Field required", result.output)

        self.assertEqual(2, result.exit_code)

    def test_ir_datasets_id_that_does_not_exists(self):
        runner = CliRunner()
        result = runner.invoke(topics_stats, ["--rag-topics", "this-is-not-an-ir-datasets-id-and-not-a-file"])

        self.assertEqual(2, result.exit_code)
        self.assertIn("Usage: topics-stats [OPTIONS]", result.output)
        self.assertIn("The argument passed to --rag-topics is not a file.", result.output)
        self.assertIn("The argument is also not a valid ir_datasets identifier that could be loaded.", result.output)

    def test_ir_datasets_is_not_installed(self):
        with mock.patch.dict("sys.modules", {"ir_datasets": None}):
            runner = CliRunner()
            result = runner.invoke(topics_stats, ["--rag-topics", "cranfield"])

            self.assertEqual(2, result.exit_code)
            self.assertIn("Usage: topics-stats [OPTIONS]", result.output)
            self.assertIn("The argument passed to --rag-topics is not a file", result.output)
            self.assertIn("ir_datasets is not installed", result.output)
            self.assertIn("Please install ir_datasets to load data from there.", result.output)

    def test_tira_is_not_installed(self):
        with mock.patch.dict("sys.modules", {"tira": None}):
            runner = CliRunner()
            result = runner.invoke(topics_stats, ["--rag-topics", "cranfield"])

            self.assertEqual(2, result.exit_code)
            self.assertIn("Usage: topics-stats [OPTIONS]", result.output)
            self.assertIn("The argument passed to --rag-topics is not a file", result.output)
            self.assertIn("tira is not installed", result.output)
            self.assertIn("Please install tira to load data from there.", result.output)
