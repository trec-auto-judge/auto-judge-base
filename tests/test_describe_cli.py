import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from autojudge_base._commands._describe import (
    JudgeDescription,
    JudgeMeasureDescription,
    describe_leaderboard,
    describe_leaderboard_click,
    filter_judge_measures_for_leaderboard,
    find_judge_description_by_remote,
    load_config_remote,
    resolve_describe_paths,
    format_judge_description,
)
from autojudge_base.cli import main


RESOURCES_DIR = Path(__file__).parent / "resources"
EXAMPLE_LEADERBOARD_DIR = RESOURCES_DIR / "example-leaderboard-ir-axioms"


class TestDescribeHelpers(unittest.TestCase):
    def test_resolve_describe_paths_for_eval_file(self):
        leaderboard_path, config_path = resolve_describe_paths(
            EXAMPLE_LEADERBOARD_DIR / "ir_axioms.eval.txt"
        )

        self.assertEqual(
            leaderboard_path,
            EXAMPLE_LEADERBOARD_DIR / "ir_axioms.eval.txt",
        )
        self.assertEqual(
            config_path,
            EXAMPLE_LEADERBOARD_DIR / "ir_axioms.config.yml",
        )

    def test_load_config_remote_reads_git_remote(self):
        remote = load_config_remote(EXAMPLE_LEADERBOARD_DIR / "ir_axioms.config.yml")

        self.assertEqual(
            remote,
            "git@github.com:trec-auto-judge/auto-judge-starter-kit.git",
        )

    def test_load_config_remote_requires_git_remote(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "missing-remote.config.yml"
            config_path.write_text("git:\n  commit: abc123\n", encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                load_config_remote(config_path)

        self.assertIn("does not contain git.remote", str(ctx.exception))

    def test_find_judge_description_by_remote_matches_catalog(self):
        catalog = (
            JudgeDescription(
                name="Judge A",
                description="Description A",
                git_remotes=("git@github.com:example/a.git",),
                measures=(JudgeMeasureDescription(name="M1", description="First"),),
            ),
            JudgeDescription(
                name="Judge B",
                description="Description B",
                git_remotes=("git@github.com:example/b.git",),
                measures=(JudgeMeasureDescription(name="M2", description="Second"),),
            ),
        )

        judge = find_judge_description_by_remote(
            "git@github.com:example/b.git",
            catalog,
        )

        self.assertEqual(judge.name, "Judge B")
        self.assertEqual(judge.measures[0].name, "M2")

    def test_filter_judge_measures_for_leaderboard_keeps_only_reported_measures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            leaderboard_path = Path(tmpdir) / "partial.eval.txt"
            original_lines = (
                EXAMPLE_LEADERBOARD_DIR / "ir_axioms.eval.txt"
            ).read_text(encoding="utf-8").splitlines()
            retained_prefixes = ("GEN-TFC1", "GEN-LNC1")
            filtered_lines = [
                line
                for line in original_lines
                if any(f"\t{prefix}\t" in line for prefix in retained_prefixes)
            ]
            leaderboard_path.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")

            judge = JudgeDescription(
                name="IR-Axioms",
                description="Axiomatic leaderboard for RAG systems.",
                git_remotes=("git@github.com:trec-auto-judge/auto-judge-starter-kit.git",),
                measures=(
                    JudgeMeasureDescription(name="GEN-TFC1", description="TFC1 measure"),
                    JudgeMeasureDescription(name="GEN-LNC1", description="LNC1 measure"),
                    JudgeMeasureDescription(name="GEN-REG", description="REG measure"),
                ),
            )

            filtered_judge = filter_judge_measures_for_leaderboard(leaderboard_path, judge)

        self.assertEqual(
            tuple(measure.name for measure in filtered_judge.measures),
            ("GEN-TFC1", "GEN-LNC1"),
        )


class TestDescribeFunction(unittest.TestCase):
    def test_describe_leaderboard_returns_description(self):
        catalog = (
            JudgeDescription(
                name="IR-Axioms",
                description="Axiomatic leaderboard for RAG systems.",
                git_remotes=("git@github.com:trec-auto-judge/auto-judge-starter-kit.git",),
                measures=(
                    JudgeMeasureDescription(name="GEN-TFC1", description="TFC1 measure"),
                    JudgeMeasureDescription(name="GEN-LNC1", description="LNC1 measure"),
                ),
            ),
        )

        description = describe_leaderboard(
            EXAMPLE_LEADERBOARD_DIR / "ir_axioms.eval.txt",
            catalog=catalog,
        )

        description = format_judge_description(description)
        self.assertIn("Judge: IR-Axioms", description)
        self.assertIn("- GEN-TFC1: TFC1 measure", description)

    def test_describe_leaderboard_includes_placeholder_for_undescribed_measure(self):
        catalog = (
            JudgeDescription(
                name="IR-Axioms",
                description="Axiomatic leaderboard for RAG systems.",
                git_remotes=("git@github.com:trec-auto-judge/auto-judge-starter-kit.git",),
                measures=(
                    JudgeMeasureDescription(name="GEN-LNC1", description="LNC1 measure"),
                ),
            ),
        )

        description = describe_leaderboard(
            EXAMPLE_LEADERBOARD_DIR / "ir_axioms.eval.txt",
            catalog=catalog,
        )
        description = format_judge_description(description)

        self.assertIn("- GEN-LNC1: LNC1 measure", description)
        self.assertIn(
            "- GEN-TFC1: No description available. Please add it via a pull request to the file https://github.com/trec-auto-judge/auto-judge-starter-kit/blob/main/judges.yml",
            description,
        )


class TestDescribeCliIntegration(unittest.TestCase):
    def test_describe_click_example_leaderboard(self):
        runner = CliRunner()
        catalog = (
            JudgeDescription(
                name="IR-Axioms",
                description="Axiomatic leaderboard for RAG systems.",
                git_remotes=("git@github.com:trec-auto-judge/auto-judge-starter-kit.git",),
                measures=(
                    JudgeMeasureDescription(name="GEN-TFC1", description="TFC1 measure"),
                    JudgeMeasureDescription(name="GEN-LNC1", description="LNC1 measure"),
                ),
            ),
        )

        with patch(
            "autojudge_base._commands._describe.fetch_judge_catalog",
            return_value=catalog,
        ):
            result = runner.invoke(
                describe_leaderboard_click,
                [str(EXAMPLE_LEADERBOARD_DIR / "ir_axioms.eval.txt")],
            )

        self.assertEqual(0, result.exit_code)
        self.assertIn("Judge: IR-Axioms", result.output)
        self.assertIn("- GEN-TFC1: TFC1 measure", result.output)

    def test_describe_click_output_includes_placeholder_for_undescribed_measure(self):
        runner = CliRunner()
        catalog = (
            JudgeDescription(
                name="IR-Axioms",
                description="Axiomatic leaderboard for RAG systems.",
                git_remotes=("git@github.com:trec-auto-judge/auto-judge-starter-kit.git",),
                measures=(
                    JudgeMeasureDescription(name="GEN-LNC1", description="LNC1 measure"),
                ),
            ),
        )

        with patch(
            "autojudge_base._commands._describe.fetch_judge_catalog",
            return_value=catalog,
        ):
            result = runner.invoke(
                describe_leaderboard_click,
                [str(EXAMPLE_LEADERBOARD_DIR / "ir_axioms.eval.txt")],
            )

        self.assertEqual(0, result.exit_code)
        self.assertIn("- GEN-LNC1: LNC1 measure", result.output)
        self.assertIn("- GEN-TFC1: No description available. Please add it via a pull request to the file https://github.com/trec-auto-judge/auto-judge-starter-kit/blob/main/judges.yml", result.output)
