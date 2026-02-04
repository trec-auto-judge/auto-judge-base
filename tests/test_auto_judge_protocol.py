import unittest
from typing import Type
from trec_auto_judge import AutoJudge, Sequence, Report, Request, Leaderboard, Optional, Qrels, MeasureSpec, LeaderboardSpec, LeaderboardBuilder, NuggetBanks
from trec_auto_judge import auto_judge_to_click_command
from trec_auto_judge.nugget_data import NuggetBanksProtocol
from trec_auto_judge.llm import MinimaLlmConfig
from click.testing import CliRunner
from . import TREC_25_DATA
from pathlib import Path
from tempfile import TemporaryDirectory

class NaiveJudge(AutoJudge):
    """A simple judge that uses NuggetBanks format."""
    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def create_nuggets(
        self,
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional["NuggetBanksProtocol"] = None,
        rag_topics: Optional[Sequence["Request"]] = None,
        **kwargs
    ) -> Optional["NuggetBanksProtocol"]:
        return None

    def judge(self,
              rag_responses: Sequence["Report"],
              rag_topics: Sequence["Request"],
              llm_config: MinimaLlmConfig,
              nugget_banks: Optional[NuggetBanksProtocol] = None,
              **kwargs) -> "Leaderboard":
        ret = LeaderboardBuilder(self.leaderboard_spec())

        for r in rag_responses:
            ret.add(
                topic_id=r.metadata.topic_id,
                run_id=r.metadata.run_id,
                values={"measure-01": 1}
            )

        return ret.build()

    def leaderboard_spec(self) -> LeaderboardSpec:
        return LeaderboardSpec(measures=(
            MeasureSpec("measure-01"),  # dtype=float is default
        ))

class TestAutoJudgeProtocoll(unittest.TestCase):
    def test_minimal_auto_judge(self):
        cmd = auto_judge_to_click_command(NaiveJudge(), "my-command")

        runner = CliRunner()

        with TemporaryDirectory() as tmp_dir:
            filebase = Path(tmp_dir) / "leaderboard"
            result = runner.invoke(cmd, ["judge", "--rag-responses", str(TREC_25_DATA / "spot-check-dataset" / "runs"), "--output", str(filebase)])

            print(result.output)
            print(result.exception)
            self.assertIsNone(result.exception)
            self.assertEqual(result.exit_code, 0)
            # Output file gets .eval.txt suffix added by resolve_leaderboard_file_path()
            leaderboard_file = filebase.parent / f"{filebase.name}.eval.txt"
            self.assertTrue(leaderboard_file.is_file())
            actual_leaderboard = leaderboard_file.read_text()
            self.assertIn("measure-01\t28\t1", actual_leaderboard)
            self.assertIn("measure-01\tall\t1.0", actual_leaderboard)

