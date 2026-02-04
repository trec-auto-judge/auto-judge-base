import unittest
from click.testing import CliRunner
import click
import json
from pathlib import Path
from trec_auto_judge import option_ir_dataset, option_rag_responses, option_rag_topics, load_requests_from_file
from . import TREC_25_DATA
from pathlib import Path

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

class TestRagTopicsIntegration(unittest.TestCase):
    def test_cranfield_explicitly_specified(self):
        runner = CliRunner()
        result = runner.invoke(topics_stats, ["--rag-topics", "cranfield"])
        self.assertIn("Topics: 225", result.output)
        self.assertEqual(0, result.exit_code)

    def test_load_topics(self):
        requests = load_requests_from_file(RESORUCES_DIR / "example-rag-topics.jsonl")
        r0 = requests[0]
        r1 = requests[1]
        self.assertEqual(r0.request_id,"28")
        self.assertEqual(r0.title,"my-topic number 28")
        self.assertEqual(r1.request_id,"29")
        self.assertEqual(r1.title,"my-topic number 29")

    def test_local_directory_explicit(self):
        runner = CliRunner()
        result = runner.invoke(topics_stats, ["--rag-topics", str(RESORUCES_DIR / "example-irds-corpus")])
        print(result.exception)
        self.assertIn("Topics: 5", result.output)
        self.assertEqual(0, result.exit_code)

    def test_local_file_explicit(self):
        runner = CliRunner()
        result = runner.invoke(topics_stats, ["--rag-topics", str(RESORUCES_DIR / "example-rag-topics.jsonl")])
        print(result.exception)
        self.assertIn("Topics: 2", result.output)
        self.assertEqual(0, result.exit_code)

    def test_local_directory_implicit_via_rag_responses(self):
        runner = CliRunner()
        result = runner.invoke(rag_stats2, ["--rag-responses", TREC_25_DATA / "spot-check-dataset" / "runs"])
        print(result.exception)
        print(result.output)
        self.assertIn("Topics: 1", result.output)
        self.assertEqual(0, result.exit_code)

    def test_local_directory_implicit_via_rag_responses3(self):
        runner = CliRunner()
        result = runner.invoke(rag_stats3, ["--rag-topics", str(RESORUCES_DIR / "example-irds-corpus"), "--rag-responses", TREC_25_DATA / "spot-check-dataset" / "runs"])
        print(result.exception)
        print(result.output)
        self.assertIn("Topics: 5", result.output)
        self.assertEqual(0, result.exit_code)

    def test_local_directory_implicit_via_rag_responses4(self):
        runner = CliRunner()
        result = runner.invoke(rag_stats3, ["--rag-responses", TREC_25_DATA / "spot-check-dataset" / "runs", "--rag-topics", str(RESORUCES_DIR / "example-irds-corpus")])
        print(result.exception)
        print(result.output)
        self.assertIn("Topics: 5", result.output)
        self.assertEqual(0, result.exit_code)

    def test_local_directory_implicit_via_rag_responses5(self):
        runner = CliRunner()
        result = runner.invoke(rag_stats2, ["--rag-responses", TREC_25_DATA / "spot-check-dataset" / "runs", "--rag-topics", str(RESORUCES_DIR / "example-irds-corpus")])
        print(result.exception)
        print(result.output)
        self.assertIn("Topics: 5", result.output)
        self.assertEqual(0, result.exit_code)

    def test_local_directory_implicit_on_local_directory1(self):
        runner = CliRunner()
        result = runner.invoke(rag_stats2, ["--rag-responses", str(RESORUCES_DIR / "spot-check-fully-local" / "runs" )])
        print(result.exception)
        print(result.output)
        self.assertIn("Topics: 5", result.output)
        self.assertEqual(0, result.exit_code)

    def test_local_directory_implicit_on_local_directory2(self):
        runner = CliRunner()
        result = runner.invoke(rag_stats3, ["--rag-responses", str(RESORUCES_DIR / "spot-check-fully-local" / "runs")])
        print(result.exception)
        print(result.output)
        self.assertIn("Topics: 5", result.output)
        self.assertEqual(0, result.exit_code)
