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
"""
import glob
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import click

from autojudge_base.report import convert as convert_report
from autojudge_base.report import load_report, submission_dict
from autojudge_base.request import load_requests_from_file
from autojudge_base.track_spec import SPECS


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


def _local_path(p) -> str:
    """Render a report's source path relative to the current working directory."""
    if p is None:
        return "?"
    try:
        return os.path.relpath(str(p))
    except ValueError:                 # different drive on Windows -> keep absolute
        return str(p)


def _load_reports(patterns):
    """Load and concatenate the reports from every file matched by the REPORTS args."""
    loaded = []
    for p in _resolve_inputs(patterns):
        loaded.extend(load_report(p))
    return loaded


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
            click.echo(f"[FAIL] {_local_path(r.path)}  team={md.team_id} run={md.run_id} "
                       f"topic={md.topic_id}: {e}", err=True)
            sys.exit(255)
    target = f"against {spec_id}" if spec_id else "(structural checks only)"
    click.echo(f"all {len(loaded)} reports valid {target}")


def _compress_ids(tids) -> str:
    """Compress integer-like topic ids into sorted ranges, e.g. '1001-1005, 1008'.

    Raises ValueError if any id is not an integer (caller falls back to a plain list).
    """
    nums = sorted(int(x) for x in tids)
    parts, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            parts.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = n
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ", ".join(parts)


def _fmt_ids(ids) -> str:
    """Compact rendering of an id list: integer ranges, or a truncated plain list."""
    try:
        return _compress_ids(ids)
    except ValueError:
        head = ", ".join(ids[:12])
        return head + (f", ... (+{len(ids) - 12} more)" if len(ids) > 12 else "")


def _fmt_topics(tids, total: int) -> str:
    """Human-readable topic set: 'all N topics' when it covers everything, else _fmt_ids."""
    return f"all {total} topics" if len(tids) == total else _fmt_ids(tids)


def _sortkey(x):
    """Sort ids numerically when possible, else lexically (numbers before strings)."""
    try:
        return (0, int(x))
    except ValueError:
        return (1, str(x))


def _report_coverage(loaded, requests: dict, spec_id, *, strict=False, suppress_warnings=False):
    """Cross-check report topic ids against the topics/requests (mapping topic-id ->
    Request, already loaded). Returns (has_error, has_warning).

    Flags topics with no report (missing_topics), report topic ids not in the file
    (new_topics), duplicate topic ids (duplicate_topics), and -- for specs that require
    it -- reports whose narrative is not an exact copy of the topic text (narrative).
    Each category is a hard error unless the spec lists it in `smells` (then a warning);
    --strict treats every category as an error, --suppress-warnings hides warning lines.
    """
    spec = SPECS.get(spec_id) if spec_id else None
    smells = () if strict else (spec.smells if spec is not None else ())
    expected = {tid: getattr(req, "title", None) for tid, req in requests.items()}
    report_ids = [str(r.metadata.topic_id) for r in loaded]
    report_set = set(report_ids)

    missing = sorted(set(expected) - report_set, key=_sortkey)
    new = sorted(report_set - set(expected), key=_sortkey)
    dups = sorted((t for t, c in Counter(report_ids).items() if c > 1), key=_sortkey)

    seen = {"err": False, "warn": False}

    def emit(label, category, msg):
        if category in smells:
            if not suppress_warnings:
                click.echo(f"[{label}] (smell) {msg}")
            seen["warn"] = True
        else:
            click.echo(f"[{label}] {msg}")
            seen["err"] = True

    if missing:
        emit("MISSING", "missing_topics", f"{len(missing)} topic(s) in the topics file have no report: {_fmt_ids(missing)}")
    if new:
        emit("NEW", "new_topics", f"{len(new)} report topic id(s) not in the topics file: {_fmt_ids(new)}")
    if dups:
        emit("DUP", "duplicate_topics", f"{len(dups)} topic id(s) appear more than once: {_fmt_ids(dups)}")
    if spec is not None and spec.require_exact_narrative:
        mism = sorted({str(r.metadata.topic_id) for r in loaded
                       if str(r.metadata.topic_id) in expected
                       and r.metadata.narrative != expected[str(r.metadata.topic_id)]}, key=_sortkey)
        if mism:
            emit("NARRATIVE", "narrative", f"{len(mism)} report(s): narrative is not an exact copy of the topic text: {_fmt_ids(mism)}")

    if not (seen["err"] or seen["warn"]):
        click.echo(f"[coverage] all {len(expected)} topics present, no extras or duplicates")
    return seen["err"], seen["warn"]


def _accumulate(groups: dict, messages, origin, tid: str) -> None:
    """Group findings by normalized signature, tracking offending run files + topics."""
    for e in messages:
        g = groups.setdefault(_signature(e), {"count": 0, "example": e, "offenders": {}})
        g["count"] += 1
        g["offenders"].setdefault(origin, {})[tid] = None   # dedupes topics per run file


def _print_groups(label: str, groups: dict) -> None:
    for g in groups.values():
        click.echo(f"{label} (affects {g['count']} report(s)):  {g['example']}")
        offenders = list(g["offenders"].items())
        for (path, team, run), topics in offenders[:_MAX_OFFENDERS]:
            click.echo(f"    {path}  team={team} run={run}  "
                       f"({len(topics)} topics: {_fmt_ids(list(topics))})")
        if len(offenders) > _MAX_OFFENDERS:
            click.echo(f"    ... and {len(offenders) - _MAX_OFFENDERS} more run file(s)")
        click.echo("")


_MAX_OFFENDERS = 25   # cap the run files listed per issue (avoids a wall of paths)


def _signature(msg: str) -> str:
    """Normalize a violation message into a GROUPING KEY so similar ones collate: mask
    quoted values (doc-ids etc.), sentence indices, and bare counts, all of which vary
    per topic. The key is only used to group; the CLI prints a real example message from
    the group (with real doc-ids/counts), not this masked form."""
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
    # signature -> {count, example message, offenders: (path, team, run) -> ordered topics}
    problems: "dict[str, dict]" = {}
    smells: "dict[str, dict]" = {}
    ok = 0
    for r in loaded:
        md = r.metadata
        tid = str(md.topic_id)
        errs, warns = r.check_findings(spec_id, request=requests.get(tid))
        if strict:                    # smells become hard errors
            errs, warns = errs + warns, []
        if not errs:
            ok += 1
        origin = (_local_path(r.path), md.team_id, md.run_id)
        _accumulate(problems, errs, origin, tid)
        _accumulate(smells, warns, origin, tid)

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

    cov_err, cov_warn = (_report_coverage(loaded, requests, spec_id, strict=strict,
                                          suppress_warnings=suppress_warnings)
                         if topics_path else (False, False))

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
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for r in loaded:
            obj = submission_dict(convert_report(r, target), target)
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    click.echo(f"converted {len(loaded)} reports to {target} -> {output}")


if __name__ == "__main__":
    main()
