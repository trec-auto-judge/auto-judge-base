"""Test MeasureSpec description and measures.yml export."""

import tempfile
from pathlib import Path

import yaml

from autojudge_base.leaderboard import (
    LeaderboardBuilder,
    LeaderboardSpec,
    MeasureSpec,
)


def test_measure_spec_description():
    """MeasureSpec stores description field."""
    spec = MeasureSpec("SCORE", description="Test description")
    assert spec.name == "SCORE"
    assert spec.dtype == float
    assert spec.description == "Test description"


def test_measure_spec_dtype_int():
    """MeasureSpec with int dtype."""
    spec = MeasureSpec("COUNT", int, description="Number of items")
    assert spec.name == "COUNT"
    assert spec.dtype == int
    assert spec.description == "Number of items"
    assert spec.cast(3.7) == 3  # int truncates
    assert spec.default == 0


def test_measure_spec_dtype_str():
    """MeasureSpec with str dtype."""
    spec = MeasureSpec("CATEGORY", str)
    assert spec.dtype == str
    assert spec.cast(123) == "123"
    assert spec.default == ""


def test_leaderboard_spec_to_measures_dict():
    """LeaderboardSpec.to_measures_dict() exports for YAML."""
    spec = LeaderboardSpec(measures=(
        MeasureSpec("RELEVANCE", description="How relevant"),
        MeasureSpec("COUNT", int, description="Number of citations"),
        MeasureSpec("LABEL", str),  # No description
    ))

    result = spec.to_measures_dict()

    assert len(result) == 3
    assert result[0] == {"name": "RELEVANCE", "dtype": "float", "description": "How relevant"}
    assert result[1] == {"name": "COUNT", "dtype": "int", "description": "Number of citations"}
    assert result[2] == {"name": "LABEL", "dtype": "str"}  # No description key when empty


def test_write_measures_yaml():
    """LeaderboardSpec.write_measures_yaml() produces valid YAML."""
    spec = LeaderboardSpec(measures=(
        MeasureSpec("RELEVANCE", description="How relevant the response is"),
        MeasureSpec("COUNT", int, description="Citation count"),
    ))

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "test.measures.yml"
        spec.write_measures_yaml(output)

        assert output.exists()

        with open(output, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "measures" in data
        assert len(data["measures"]) == 2
        assert data["measures"][0]["name"] == "RELEVANCE"
        assert data["measures"][0]["dtype"] == "float"
        assert data["measures"][0]["description"] == "How relevant the response is"
        assert data["measures"][1]["name"] == "COUNT"
        assert data["measures"][1]["dtype"] == "int"


def test_leaderboard_has_spec():
    """Leaderboard built via builder has spec attached."""
    spec = LeaderboardSpec(measures=(
        MeasureSpec("SCORE", description="Overall score"),
    ))

    builder = LeaderboardBuilder(spec)
    builder.add(run_id="run1", topic_id="t1", values={"SCORE": 0.8})
    leaderboard = builder.build()

    assert leaderboard.spec is spec
    assert leaderboard.spec.measures[0].description == "Overall score"