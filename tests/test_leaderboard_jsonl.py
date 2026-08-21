"""Tests for Leaderboard.write(format="jsonl").

The jsonl output is a wire format shared with autojudge_evaluate.eval_results.io,
which produces the official *.eval.jsonl files that meta-evaluate reads back. The
shape is therefore a compatibility contract, not an internal detail: one JSON
object per (entry, measure), keys in the order run_id, topic_id, measure, value,
serialized with json.dumps defaults.
"""

import json

import pytest

from autojudge_base import LeaderboardBuilder, LeaderboardSpec, MeasureSpec
from autojudge_base.leaderboard.leaderboard import Leaderboard


SCORES = {
    "runA": {"t1": 6.3943, "t2": 0.2501, "t3": 2.7503},
    "runB": {"t1": 1.5, "t2": 2.5, "t3": 3.5},
}


@pytest.fixture
def leaderboard():
    builder = LeaderboardBuilder(
        LeaderboardSpec(measures=(MeasureSpec("random", dtype=float),))
    )
    for run_id, topics in SCORES.items():
        for topic_id, value in topics.items():
            builder.add(run_id=run_id, topic_id=topic_id, values={"random": value})
    return builder.build()


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestJsonlFormat:
    """The wire format shared with autojudge_evaluate."""

    def test_key_order_is_fixed(self, leaderboard, tmp_path):
        path = tmp_path / "eval.jsonl"
        leaderboard.write(path, format="jsonl")

        for record in read(path):
            assert list(record) == ["run_id", "topic_id", "measure", "value"]

    def test_serialized_with_json_dumps_defaults(self, leaderboard, tmp_path):
        """Spacing matters: autojudge_evaluate uses json.dumps(obj) with no separators."""
        path = tmp_path / "eval.jsonl"
        leaderboard.write(path, format="jsonl")

        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first == json.dumps(json.loads(first))
        assert '", "' in first  # spaces after commas, i.e. not the compact form

    def test_values_stay_numeric(self, leaderboard, tmp_path):
        path = tmp_path / "eval.jsonl"
        leaderboard.write(path, format="jsonl")

        for record in read(path):
            assert isinstance(record["value"], float)

    def test_one_record_per_entry_and_measure(self, leaderboard, tmp_path):
        path = tmp_path / "eval.jsonl"
        leaderboard.write(path, format="jsonl")

        records = read(path)
        # 2 runs x 3 topics, plus one aggregate row per run
        assert len(records) == 8
        pairs = {(r["run_id"], r["topic_id"]) for r in records}
        assert ("runA", "all") in pairs
        assert ("runB", "all") in pairs

    def test_aggregate_keeps_the_measure_name(self, leaderboard, tmp_path):
        path = tmp_path / "eval.jsonl"
        leaderboard.write(path, format="jsonl")

        assert {r["measure"] for r in read(path)} == {"random"}

    def test_aggregate_is_the_unrounded_mean(self, leaderboard, tmp_path):
        path = tmp_path / "eval.jsonl"
        leaderboard.write(path, format="jsonl")

        by_key = {(r["run_id"], r["topic_id"]): r["value"] for r in read(path)}
        expected = sum(SCORES["runA"].values()) / len(SCORES["runA"])
        assert by_key[("runA", "all")] == pytest.approx(expected, abs=1e-12)


class TestRoundTrip:
    """write -> load must be lossless."""

    def test_load_reads_back_what_write_produced(self, leaderboard, tmp_path):
        path = tmp_path / "eval.jsonl"
        leaderboard.write(path, format="jsonl")

        loaded = Leaderboard.load(path, format="jsonl")

        assert loaded.measures == ("random",)
        assert len(loaded.entries) == len(leaderboard.entries)

        original = {(e.run_id, e.topic_id): e.values["random"] for e in leaderboard.entries}
        restored = {(e.run_id, e.topic_id): float(e.values["random"]) for e in loaded.entries}
        assert restored == pytest.approx(original)


class TestOtherFormats:
    """The new branch must not disturb the existing ones."""

    def test_ir_measures_still_tab_separated(self, leaderboard, tmp_path):
        path = tmp_path / "eval.tsv"
        leaderboard.write(path, format="ir_measures")

        line = path.read_text(encoding="utf-8").splitlines()[0]
        run_id, topic_id, measure, value = line.split("\t")
        assert measure == "random"
        assert float(value) >= 0.0

    def test_unknown_format_still_raises(self, leaderboard, tmp_path):
        with pytest.raises(ValueError, match="Unknown format"):
            leaderboard.write(tmp_path / "eval.out", format="nonsense")
