import unittest
from trec_auto_judge import main
from tempfile import TemporaryDirectory
from click.testing import CliRunner
from . import TREC_25_DATA
from shutil import copytree
from pathlib import Path

class TestExportCorpora(unittest.TestCase):
    def test_fails_on_wrong_directory(self):
        with TemporaryDirectory() as tmp:
            runner = CliRunner()
            result = runner.invoke(main, ["export-corpus", str(tmp)])

            self.assertIsNotNone(result.exception)
            self.assertEqual(result.exit_code, 1)

    def test_correct_directory(self):
        with TemporaryDirectory() as tmp:
            target_dir = str(tmp) + "/spot-check"
            copytree(TREC_25_DATA / "spot-check-dataset", target_dir)
            target_path = Path(target_dir) / "queries.jsonl"
            ff = (Path(target_dir) / "README.md").read_text()
            ff = ff.replace('#  ir_datasets_id: "cranfield"', '  ir_datasets_id: "cranfield"')
            (Path(target_dir) / "README.md").write_text(ff)
            target_path.unlink()
            self.assertFalse(target_path.is_file())

            runner = CliRunner()
            result = runner.invoke(main, ["export-corpus", str(target_dir)])

            self.assertTrue(target_path.is_file())            
            self.assertIsNone(result.exception)
            self.assertEqual(result.exit_code, 0)
            self.assertIn('{"query_id": "220", "text": "find a calculation procedure applicable', target_path.read_text())
