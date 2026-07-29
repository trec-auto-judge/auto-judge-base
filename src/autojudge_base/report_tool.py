"""Command-line utility for TREC RAG-family report submissions.

Read a submission JSONL and verify/check it against a track spec, or convert it to
another track's format:

    # fail fast: stop at the FIRST violation anywhere (submission gate). REPORTS may be
    # individual files and/or glob patterns -- quote globs so the tool expands them:
    python -m autojudge_base.report_tool verify  'runs/*.jsonl' --spec rag26

    # list EVERY violation (collated); --topics also checks topic-id coverage
    python -m autojudge_base.report_tool check   run.jsonl --spec rag26 --topics topics.jsonl

    # convert to another format and write a new submission file
    python -m autojudge_base.report_tool convert run.jsonl --to rag26 -o run.rag26.jsonl

Reports load with the permissive `Report` binding, so any track's format reads in; the
`--spec`/`--to` value is the caller-supplied target (a track id from the built-in
registry, e.g. rag26 / rag25 / ragtime25 / ragtime26 / dragun25).

This is a thin CLI over the reusable library layer: `autojudge_base.report_verification`
(ReportVerification: per-report spec compliance + cross-report coverage) and
`autojudge_base.report_export` (emit a submission JSONL).
"""
import glob
import re
import sys
from pathlib import Path

import click

from autojudge_base.report_export import write_submission_output
from autojudge_base.report import load_report
from autojudge_base.report_verification import ReportVerification, fmt_ids, local_path
from autojudge_base.request import load_requests_from_file
from autojudge_base.track_spec import SPECS

_MAX_OFFENDERS = 25   # cap the run files listed per issue (avoids a wall of paths)


def _signature(msg: str) -> str:
    """Normalize a finding message into a GROUPING KEY so similar ones collate: mask
    quoted values (doc-ids etc.), sentence indices, and bare counts. The key only groups;
    the display shows a real example message (with real doc-ids/counts), not this form."""
    s = re.sub(r"\(got [^)]*\)", "(got …)", msg)  # (got None)/(got '') collate together
    s = re.sub(r"'[^']*'", "'…'", s)       # 'shard_00122_5199' -> '…'
    s = re.sub(r"\[\d+\]", "[i]", s)        # answer[0] -> answer[i]
    # keep the length LIMIT as a grouping key (it varies per topic in RAGTIME) while
    # still masking the variable count -- so "limit 2000" and "limit 10000" don't merge
    limit = re.search(r"\(limit (\d+)\)", s)
    s = re.sub(r"\(limit \d+\)", "(limit \x00)", s)
    s = re.sub(r"\d+", "N", s)              # counts/lengths (e.g. "4 cited") -> N
    if limit:
        s = s.replace("\x00", limit.group(1))
    return s


def _group(findings) -> dict:
    """Collate spec findings by normalized signature, tracking offending run files +
    topics. Returns signature -> {count, example message, offenders: origin -> topics}."""
    groups: dict = {}
    for f in findings:
        g = groups.setdefault(_signature(f.message), {"count": 0, "example": f.message, "offenders": {}})
        g["count"] += 1
        g["offenders"].setdefault(f.origin, {})[f.topic] = None   # dedupes topics per run file
    return groups


def _spec_help() -> str:
    return "track id: " + ", ".join(sorted(SPECS))


def _resolve_inputs(patterns) -> list:
    """Expand REPORTS args into concrete files. Each arg may be a single file, a glob
    pattern (e.g. 'runs/*.jsonl'), or a directory (all files directly inside it are used
    -- run dirs hold extensionless run files, so a glob would not match them). Preserves
    arg order, sorts within a glob/directory, dedupes."""
    out, seen = [], set()
    for pat in patterns:
        matches = sorted(glob.glob(pat, recursive=True)) if glob.has_magic(pat) else [pat]
        if not matches:
            raise click.BadParameter(f"no files match pattern {pat!r}")
        for m in matches:
            p = Path(m)
            if p.is_dir():
                files = sorted(f for f in p.iterdir() if f.is_file())
                if not files:
                    raise click.BadParameter(f"{m}: directory contains no files")
            elif p.is_file():
                files = [p]
            else:
                raise click.BadParameter(f"{m}: no such file or directory")
            for f in files:
                if f not in seen:
                    seen.add(f)
                    out.append(f)
    return out


def _load_reports(patterns):
    """Load and concatenate the reports from every file matched by the REPORTS args."""
    loaded = []
    for p in _resolve_inputs(patterns):
        loaded.extend(load_report(p))
    return loaded


def _print_groups(label: str, groups: dict) -> None:
    """Print collated finding groups (from validate.collate) with their offending runs."""
    for g in groups.values():
        click.echo(f"{label} (affects {g['count']} report(s)):  {g['example']}")
        offenders = list(g["offenders"].items())
        for (path, team, run), topics in offenders[:_MAX_OFFENDERS]:
            click.echo(f"    {path}  team={team} run={run}  "
                       f"({len(topics)} topics: {fmt_ids(list(topics))})")
        if len(offenders) > _MAX_OFFENDERS:
            click.echo(f"    ... and {len(offenders) - _MAX_OFFENDERS} more run file(s)")
        click.echo("")


@click.group()
def main():
    """Check or convert TREC RAG-family report submissions."""


@main.command()
@click.argument("reports", nargs=-1, required=True, metavar="REPORTS...")
@click.option("--spec", "spec_id", default=None, help=f"{_spec_help()} (omit for structural-only)")
def verify(reports, spec_id):
    """Verify REPORTS against a spec, FAILING FAST at the first violation (submission gate)."""
    loaded = _load_reports(reports)
    for r in loaded:
        try:
            r.verify(spec_id)  # raises TrackSpecVerificationError on the first violation
        except RuntimeError as e:
            md = r.metadata
            click.echo(f"[FAIL] {local_path(r.path)}  team={md.team_id} run={md.run_id} "
                       f"topic={md.topic_id}: {e}", err=True)
            sys.exit(255)
    target = f"against {spec_id}" if spec_id else "(structural checks only)"
    click.echo(f"all {len(loaded)} reports valid {target}")


@main.command()
@click.argument("reports", nargs=-1, required=True, metavar="REPORTS...")
@click.option("--spec", "spec_id", default=None, help=f"{_spec_help()} (omit for structural-only)")
@click.option("--topics", "topics_path", type=click.Path(exists=True, path_type=Path), default=None,
              help="topics/requests JSONL (Request objects): topic-id coverage + per-request length limits")
@click.option("--strict", is_flag=True, help="treat smells (warnings) as hard errors -- fail on them too")
@click.option("--suppress-warnings", "suppress_warnings", is_flag=True, help="hide the SMELL/warning output")
def check(reports, spec_id, topics_path, strict, suppress_warnings):
    """Check every report in REPORTS and list all violations, COLLATED by issue.

    Similar findings collate by issue: one example message, then the run files where it
    occurs (path + team + run_id + topics). PROBLEM = hard failure; SMELL = warning only
    (categories a spec lists in `smells`; see track_specs.yml). With --topics, also
    cross-check topic-id coverage and enforce per-request length limits (RAGTIME).

    Exit code is 255 if there is any PROBLEM (or coverage error), else 0. --strict turns
    every smell into a hard error (so it fails too); --suppress-warnings hides smells.
    """
    loaded = _load_reports(reports)
    total = len(loaded)
    # topic-id -> Request; supplies per-request length limits and the coverage cross-check
    requests = {str(q.request_id): q for q in load_requests_from_file(topics_path)} if topics_path else {}

    v = ReportVerification(loaded, spec_id, requests=requests, strict=strict).spec_compliant()
    if topics_path:
        v.coverage()

    spec_findings = [f for f in v.findings if f.category == "spec"]
    cov_findings = [f for f in v.findings if f.category != "spec"]
    problems = _group([f for f in spec_findings if f.severity == "error"])
    smells = _group([f for f in spec_findings if f.severity == "warning"])
    ok = v.ok

    show_smells = bool(smells) and not suppress_warnings
    if problems or show_smells:
        click.echo(f"Found {len(problems)} problem kind(s) and {len(smells)} smell kind(s) across {total} reports.\n")
        click.echo("How to read this: each block is one kind of finding -- one real example, then")
        click.echo("the run files where it occurs. Each run-file line is:")
        click.echo("    <run file>  team=<team_id> run=<run_id>  (<n> topics: <topic ids>)")
        click.echo("In an example, 'answer[0]'/'responses[0]' = the first sentence (index 0) of that")
        click.echo("report's answer list; a 'docid pattern' is the required document-id format;")
        click.echo("'cited' = referenced by the answer text. PROBLEM = failure; SMELL = warning.\n")
    _print_groups("PROBLEM", problems)
    if show_smells:
        click.echo("--- SMELLS (warnings only; do not cause failure) ---\n")
        _print_groups("SMELL", smells)

    cov_err = any(f.severity == "error" for f in cov_findings)
    cov_warn = any(f.severity == "warning" for f in cov_findings)
    if topics_path:
        if not cov_findings:
            click.echo(f"[coverage] all {len(requests)} topics present, no extras or duplicates")
        for f in cov_findings:
            if f.severity == "warning":
                if not suppress_warnings:
                    click.echo(f"[{f.label}] (smell) {f.message}")
            else:
                click.echo(f"[{f.label}] {f.message}")

    target = f"against {spec_id}" if spec_id else "(structural checks only)"
    if smells or cov_warn:
        warn_note = "  (warnings hidden by --suppress-warnings)" if suppress_warnings else "  (plus warnings above)"
    else:
        warn_note = ""
    click.echo(f"\n{ok}/{total} reports valid {target}{warn_note}")
    sys.exit(0 if (not problems and not cov_err) else 255)


@main.command()
@click.argument("reports", nargs=-1, required=True, metavar="REPORTS...")
@click.option("--to", "target", required=True, help=f"target {_spec_help()}")
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True,
              help="output JSONL path")
def convert(reports, target: str, output: Path):
    """Convert every report in REPORTS to the --to format and write --output JSONL."""
    loaded = _load_reports(reports)
    n = write_submission_output(loaded, output, target)
    click.echo(f"converted {n} reports to {target} -> {output}")


if __name__ == "__main__":
    main()
