"""Tests for submission_dict and the report_tool CLI (check / convert)."""
import json
from pathlib import Path

from click.testing import CliRunner

from autojudge_base.report import (
    Rag24ReportSentence,
    Report,
    ReportMetaData,
    convert,
    submission_dict,
)
from autojudge_base.report_tool import main

SHARD, SHARD2 = "shard_00459_61697", "shard_01234_5678"


def _rag26_report():
    md = ReportMetaData(team_id="T", narrative_id="1", narrative="q text",
                        run_id="run", run_desc="desc")
    sents = [Rag24ReportSentence(text="First.", citations=[0]),
             Rag24ReportSentence(text="Second.", citations=[1])]
    return Report(metadata=md, responses=sents, references=[SHARD, SHARD2])


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "run.jsonl"
    p.write_text(json.dumps(submission_dict(_rag26_report(), "rag26")) + "\n", encoding="utf-8")
    return p


def test_submission_dict_rag26_shape():
    obj = submission_dict(_rag26_report(), "rag26")
    assert set(obj.keys()) == {"metadata", "references", "answer"}
    assert set(obj["metadata"].keys()) == {"team_id", "narrative_id", "narrative", "run_id", "run_desc"}
    assert obj["answer"][0]["citations"] == [0]


def test_cli_verify_valid(tmp_path: Path):
    result = CliRunner().invoke(main, ["verify", str(_write(tmp_path)), "--spec", "rag26"])
    assert result.exit_code == 0
    assert "valid" in result.output


def test_cli_verify_failfast(tmp_path: Path):
    # ragtime26 rejects the rag24 sentences -> verify stops at the first violation
    result = CliRunner().invoke(main, ["verify", str(_write(tmp_path)), "--spec", "ragtime26"])
    assert result.exit_code == 255


def test_cli_verify_accepts_glob_and_multiple_files(tmp_path: Path):
    for name in ("a.jsonl", "b.jsonl"):
        (tmp_path / name).write_text(
            json.dumps(submission_dict(_rag26_report(), "rag26")) + "\n", encoding="utf-8")
    # a quoted glob pattern expands to both files
    r1 = CliRunner().invoke(main, ["verify", str(tmp_path / "*.jsonl"), "--spec", "rag26"])
    assert r1.exit_code == 0
    assert "all 2 reports valid" in r1.output
    # explicit multiple files also work
    r2 = CliRunner().invoke(
        main, ["verify", str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl"), "--spec", "rag26"])
    assert r2.exit_code == 0
    # a glob that matches nothing is a usage error
    r3 = CliRunner().invoke(main, ["verify", str(tmp_path / "none-*.jsonl"), "--spec", "rag26"])
    assert r3.exit_code != 0


def test_cli_check_valid(tmp_path: Path):
    result = CliRunner().invoke(main, ["check", str(_write(tmp_path)), "--spec", "rag26"])
    assert result.exit_code == 0
    assert "1/1 reports valid" in result.output


def test_cli_check_reports_violations(tmp_path: Path):
    # ragtime26 rejects the rag24 sentences + ClimbMix docids -> non-zero exit
    result = CliRunner().invoke(main, ["check", str(_write(tmp_path)), "--spec", "ragtime26"])
    assert result.exit_code == 255
    assert "PROBLEM" in result.output


def test_cli_check_collates_across_topics(tmp_path: Path):
    # three ragtime reports all missing `task` -> one grouped line, not three
    uuid = "b6a21af8-9cc4-462d-9c70-00bb9f009401_56341480"
    lines = [json.dumps({
        "metadata": {"team_id": "t", "topic_id": str(i), "run_id": "r"},  # no 'task'
        "references": [uuid],
        "responses": [{"text": "s.", "citations": {uuid: 1.0}}],
    }) for i in (1, 2, 3)]
    p = tmp_path / "runs.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = CliRunner().invoke(main, ["check", str(p), "--spec", "ragtime25"])
    assert result.exit_code == 255
    assert result.output.count("PROBLEM (") == 1     # collated to a single issue
    assert "metadata.task" in result.output          # the example message
    assert "team=t run=r" in result.output           # offending run identified
    assert "1-3" in result.output                    # its topics, compressed


def test_cli_check_directory_names_offender_path_team_run(tmp_path: Path):
    # a DIRECTORY of extensionless run files; each violating report is located by
    # path + team + run_id in the collated output.
    def obj(topic, team, run):
        return {"metadata": {"team_id": team, "narrative_id": topic, "narrative": "n",
                             "run_id": run, "run_desc": "d"},
                "references": [SHARD],
                "answer": [{"text": "x", "citations": [5]}]}   # citation index out of range
    (tmp_path / "runA").write_text(json.dumps(obj("1", "teamA", "runA")) + "\n", encoding="utf-8")
    (tmp_path / "runB").write_text(json.dumps(obj("2", "teamB", "runB")) + "\n", encoding="utf-8")

    result = CliRunner().invoke(main, ["check", str(tmp_path), "--spec", "rag26"])
    assert result.exit_code == 255
    assert "team=teamA run=runA" in result.output
    assert "team=teamB run=runB" in result.output
    assert "runA" in result.output and "runB" in result.output   # source paths shown


def test_cli_check_smell_warns_not_fails(tmp_path: Path):
    # an empty answer is a smell -> printed under SMELL, but exit stays 0 (not a failure)
    obj = {"metadata": {"team_id": "t", "narrative_id": "1", "narrative": "n",
                        "run_id": "r", "run_desc": "d"},
           "references": [], "answer": []}
    p = tmp_path / "run.jsonl"
    p.write_text(json.dumps(obj) + "\n", encoding="utf-8")
    result = CliRunner().invoke(main, ["check", str(p), "--spec", "rag26"])
    assert result.exit_code == 0                 # a smell does not fail the check
    assert "SMELL (affects" in result.output
    assert "empty answer" in result.output
    assert "PROBLEM (affects" not in result.output   # no hard-error block


def test_cli_check_strict_and_suppress_warnings(tmp_path: Path):
    obj = {"metadata": {"team_id": "t", "narrative_id": "1", "narrative": "n",
                        "run_id": "r", "run_desc": "d"},
           "references": [], "answer": []}   # empty answer -> a smell by default
    p = tmp_path / "run.jsonl"
    p.write_text(json.dumps(obj) + "\n", encoding="utf-8")

    # --strict: the smell becomes a hard error -> exit 255, shown under PROBLEM
    rs = CliRunner().invoke(main, ["check", str(p), "--spec", "rag26", "--strict"])
    assert rs.exit_code == 255
    assert "PROBLEM (affects" in rs.output and "SMELL (affects" not in rs.output

    # --suppress-warnings: the smell is hidden, exit stays 0
    rp = CliRunner().invoke(main, ["check", str(p), "--spec", "rag26", "--suppress-warnings"])
    assert rp.exit_code == 0
    assert "SMELL (affects" not in rp.output and "empty answer" not in rp.output


def test_cli_check_topics_coverage(tmp_path: Path):
    # reports for topics 1, 2, 9; the topics file declares 1, 2, 3
    def rag26_obj(tid, narrative):
        return {
            "metadata": {"team_id": "T", "narrative_id": tid, "narrative": narrative,
                         "run_id": "r", "run_desc": "d"},
            "references": [SHARD, SHARD2],
            "answer": [{"text": "First.", "citations": [0]}, {"text": "Second.", "citations": [1]}],
        }
    reports = tmp_path / "reports.jsonl"
    reports.write_text(
        "\n".join(json.dumps(rag26_obj(t, f"topic {t}")) for t in ("1", "2", "9")) + "\n",
        encoding="utf-8")
    topics = tmp_path / "topics.jsonl"
    topics.write_text(
        "\n".join(json.dumps({"request_id": t, "title": f"topic {t}"}) for t in ("1", "2", "3")) + "\n",
        encoding="utf-8")

    result = CliRunner().invoke(
        main, ["check", str(reports), "--spec", "rag26", "--topics", str(topics)])
    assert result.exit_code == 255                 # coverage issues -> non-zero
    assert "[MISSING]" in result.output and "3" in result.output   # topic 3 has no report
    assert "[NEW]" in result.output and "9" in result.output       # topic 9 not in topics file


def test_cli_check_topics_enforces_ragtime_length(tmp_path: Path):
    # RAGTIME char limit comes from the Request; --topics supplies it
    uuid = "b6a21af8-9cc4-462d-9c70-00bb9f009401_56341480"
    report = {"metadata": {"team_id": "t", "topic_id": "1", "run_id": "r", "task": "multilingual"},
              "references": [],
              "responses": [{"text": "x" * 50, "citations": {uuid: 1.0}}]}
    reports = tmp_path / "r.jsonl"
    reports.write_text(json.dumps(report) + "\n", encoding="utf-8")
    topics = tmp_path / "t.jsonl"
    topics.write_text(json.dumps({"request_id": "1", "title": "t", "limit": 10}) + "\n", encoding="utf-8")

    # over-length is a SMELL for ragtime -> printed, exit 0
    result = CliRunner().invoke(
        main, ["check", str(reports), "--spec", "ragtime25", "--topics", str(topics)])
    assert result.exit_code == 0
    assert "SMELL (affects" in result.output and "chars" in result.output and "limit 10" in result.output
    # --strict escalates it to a hard error (like the validator's --strict_on_length)
    strict = CliRunner().invoke(
        main, ["check", str(reports), "--spec", "ragtime25", "--topics", str(topics), "--strict"])
    assert strict.exit_code == 255


def test_fmt_topics():
    from autojudge_base.report_tool import _fmt_topics, _signature
    assert _fmt_topics(["1", "2", "3"], 3) == "all 3 topics"
    assert _fmt_topics(["1001", "1002", "1003", "1005"], 10) == "1001-1003, 1005"
    assert _fmt_topics(["a", "b"], 5) == "a, b"       # non-integer ids -> plain list
    # similar messages share a signature (masked doc-id)
    assert _signature("cited doc-id 'x' bad") == _signature("cited doc-id 'y' bad")


def test_cli_convert(tmp_path: Path):
    src = _write(tmp_path)
    out = tmp_path / "as_ragtime.jsonl"
    result = CliRunner().invoke(main, ["convert", str(src), "--to", "ragtime26", "-o", str(out)])
    assert result.exit_code == 0
    obj = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert "responses" in obj and "answer" not in obj      # ragtime sentences key
    assert isinstance(obj["responses"][0]["citations"], dict)  # doc_id -> score
