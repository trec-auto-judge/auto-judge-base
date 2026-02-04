import unittest
from click.testing import CliRunner
from pathlib import Path
from . import TREC_25_DATA
from trec_auto_judge import main
from tempfile import TemporaryDirectory

EXAMPLE_LEADERBOARD = str((TREC_25_DATA / "spot-check-dataset" / "trec-leaderboard.txt").absolute())


def run_cmd_on_main(cmd):
    runner = CliRunner()
    ret = runner.invoke(main, cmd)
    return ret, ' '.join(ret.stdout.split())


class TestLeaderboardInterface(unittest.TestCase):
    def test_leaderboard_statistics(self):
        """Test that leaderboard command computes statistics for each run."""
        cmd = ["leaderboard", "--input", EXAMPLE_LEADERBOARD, "--eval-format", "tot"]
        result, stdout = run_cmd_on_main(cmd)

        self.assertIsNone(result.exception)
        self.assertEqual(result.exit_code, 0)

        # Check that output contains expected columns
        self.assertIn("Judge", stdout)
        self.assertIn("RunID", stdout)
        self.assertIn("Measure", stdout)
        self.assertIn("Mean", stdout)
        self.assertIn("Stdev", stdout)

        # Check specific values for Measure-01 (runs have values 1, 2, 3)
        # Each run has only 1 topic, so stdev is 0.0
        self.assertIn("my_best_run_01 Measure-01 1 1.0", stdout)
        self.assertIn("my_best_run_02 Measure-01 1 2.0", stdout)
        self.assertIn("my_best_run_02_citations Measure-01 1 3.0", stdout)

        # Check Measure-02 has inverse values (3, 2, 1)
        self.assertIn("my_best_run_01 Measure-02 1 3.0", stdout)

    def test_leaderboard_measure_filter(self):
        """Test that --eval-measure filters to specific measure."""
        cmd = ["leaderboard", "--input", EXAMPLE_LEADERBOARD, "--eval-format", "tot",
               "--eval-measure", "Measure-01"]
        result, stdout = run_cmd_on_main(cmd)

        self.assertIsNone(result.exception)
        self.assertEqual(result.exit_code, 0)

        # Should have Measure-01 but not Measure-02
        self.assertIn("Measure-01", stdout)
        self.assertNotIn("Measure-02", stdout)

    def test_leaderboard_with_jsonl_output(self):
        """Test that leaderboard command can write JSONL output."""
        with TemporaryDirectory() as tmp_dir:
            target_file = (Path(tmp_dir) / "results.jsonl").absolute()

            cmd = ["leaderboard", "--input", EXAMPLE_LEADERBOARD, "--eval-format", "tot",
                   "--output", target_file]
            result, stdout = run_cmd_on_main(cmd)

            self.assertIsNone(result.exception)
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(target_file.is_file())

            content = target_file.read_text()
            # Check specific run appears with correct values
            self.assertIn('"RunID":"my_best_run_02"', content)
            self.assertIn('"Measure":"Measure-01"', content)

    def test_leaderboard_with_csv_output(self):
        """Test that leaderboard command can write CSV output."""
        with TemporaryDirectory() as tmp_dir:
            target_file = (Path(tmp_dir) / "results.csv").absolute()

            cmd = ["leaderboard", "--input", EXAMPLE_LEADERBOARD, "--eval-format", "tot",
                   "--output", target_file]
            result, stdout = run_cmd_on_main(cmd)

            self.assertIsNone(result.exception)
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(target_file.is_file())

            content = target_file.read_text()
            # CSV should have header row
            self.assertIn("Judge,RunID,Measure,Topics,Mean", content)
            self.assertIn("my_best_run_01", content)