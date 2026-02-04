import unittest
from click.testing import CliRunner
import click
import json
from pathlib import Path
from trec_auto_judge import option_ir_dataset, option_rag_responses
from . import TREC_25_DATA
from pathlib import Path

@click.command()
@option_ir_dataset()
def ir_dataset_stats(ir_dataset):
    print("Docs:",  len(list(ir_dataset.docs_iter())))
    print("Topics:",  len(list(ir_dataset.queries_iter())))
    return 0

@click.command()
@option_ir_dataset()
@option_rag_responses()
def ir_dataset_stats2(rag_responses, ir_dataset):
    print("Docs:",  len(list(ir_dataset.docs_iter())))
    print("Topics:",  len(list(ir_dataset.queries_iter())))
    return 0

@click.command()
@option_rag_responses()
@option_ir_dataset()
def ir_dataset_stats3(ir_dataset, rag_responses):
    print("Docs:",  len(list(ir_dataset.docs_iter())))
    print("Topics:",  len(list(ir_dataset.queries_iter())))
    return 0

RESORUCES_DIR = Path(__file__).parent / "resources"

class TestIrDatasetsIntegration(unittest.TestCase):
    def test_cranfield_explicitly_specified(self):
        runner = CliRunner()
        result = runner.invoke(ir_dataset_stats, ["--ir-dataset", "cranfield"])
        print(result.exception)
        self.assertIn("Docs: 1400", result.output)
        self.assertIn("Topics: 225", result.output)
        self.assertEqual(0, result.exit_code)

    def test_local_directory_explicit(self):
        runner = CliRunner()
        result = runner.invoke(ir_dataset_stats, ["--ir-dataset", str(RESORUCES_DIR / "example-irds-corpus")])
        print(result.exception)
        self.assertIn("Docs: 3", result.output)
        self.assertIn("Topics: 5", result.output)
        self.assertEqual(0, result.exit_code)

    def test_local_directory_implicit_via_rag_responses(self):
        runner = CliRunner()
        result = runner.invoke(ir_dataset_stats2, ["--rag-responses", TREC_25_DATA / "spot-check-dataset" / "runs"])
        print(result.exception)
        print(result.output)
        self.assertIn("Docs: 1400", result.output)
        self.assertIn("Topics: 1", result.output)
        self.assertEqual(0, result.exit_code)

    def test_local_directory_implicit_via_rag_responses2(self):
        runner = CliRunner()
        result = runner.invoke(ir_dataset_stats3, ["--rag-responses", TREC_25_DATA / "spot-check-dataset" / "runs"])
        print(result.exception)
        print(result.output)
        self.assertIn("Docs: 1400", result.output)
        self.assertIn("Topics: 1", result.output)
        self.assertEqual(0, result.exit_code)

    def test_local_directory_implicit_via_rag_responses3(self):
        runner = CliRunner()
        result = runner.invoke(ir_dataset_stats3, ["--ir-dataset", str(RESORUCES_DIR / "example-irds-corpus"), "--rag-responses", TREC_25_DATA / "spot-check-dataset" / "runs"])
        print(result.exception)
        print(result.output)
        self.assertIn("Docs: 3", result.output)
        self.assertIn("Topics: 5", result.output)
        self.assertEqual(0, result.exit_code)

    def test_local_directory_implicit_via_rag_responses4(self):
        runner = CliRunner()
        result = runner.invoke(ir_dataset_stats3, ["--rag-responses", TREC_25_DATA / "spot-check-dataset" / "runs", "--ir-dataset", str(RESORUCES_DIR / "example-irds-corpus")])
        print(result.exception)
        print(result.output)
        self.assertIn("Docs: 3", result.output)
        self.assertIn("Topics: 5", result.output)
        self.assertEqual(0, result.exit_code)

    def test_local_directory_implicit_via_rag_responses5(self):
        runner = CliRunner()
        result = runner.invoke(ir_dataset_stats2, ["--rag-responses", TREC_25_DATA / "spot-check-dataset" / "runs", "--ir-dataset", str(RESORUCES_DIR / "example-irds-corpus")])
        print(result.exception)
        print(result.output)
        self.assertIn("Docs: 3", result.output)
        self.assertIn("Topics: 5", result.output)
        self.assertEqual(0, result.exit_code)

    def test_local_directory_implicit_on_local_directory1(self):
        runner = CliRunner()
        result = runner.invoke(ir_dataset_stats2, ["--rag-responses", str(RESORUCES_DIR / "spot-check-fully-local" / "runs")])
        print(result.exception)
        print(result.output)
        self.assertIn("Docs: 3", result.output)
        self.assertIn("Topics: 5", result.output)
        self.assertEqual(0, result.exit_code)

    def test_local_directory_implicit_on_local_directory2(self):
        runner = CliRunner()
        result = runner.invoke(ir_dataset_stats3, ["--rag-responses", str(RESORUCES_DIR / "spot-check-fully-local" / "runs")])
        print(result.exception)
        print(result.output)
        self.assertIn("Docs: 3", result.output)
        self.assertIn("Topics: 5", result.output)
        self.assertEqual(0, result.exit_code)
