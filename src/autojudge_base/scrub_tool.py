"""CLI for :mod:`autojudge_base.scrub` -- the ``scrub`` console script.

    scrub RUNFILE                      # tier 1: shape only
    scrub --chars RUNFILE              # tier 2: lengths and punctuation too
    scrub --topic 37 --run plum RUNFILE
    cat record.json | scrub

Reads JSONL (or a single JSON document) and writes the scrubbed form to stdout.
``--topic`` / ``--run`` / ``--index`` select the failing record *without*
reading the file first, which is the point: locating a record by eye is exactly
what the policy forbids.

Every invocation appends one structural entry to the scrub log -- counts, paths
and selectors, never a value -- so what was scrubbed is auditable and the log
can ship with a submission. Set ``SCRUB_LOG`` to relocate it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, TextIO

from .scrub import ScrubStats, scrub_json_line

DEFAULT_LOG = "scrub-log.jsonl"


def _selected(line: str, topic: Optional[str], run: Optional[str]) -> bool:
    """Match a record by its anonymised identifiers without inspecting content.

    Only ``metadata`` identifiers are read, and only compared -- they are the
    anonymised layer, and are visible under the policy.
    """
    if topic is None and run is None:
        return True
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return False              # unparseable lines cannot be selected by id
    meta = rec.get("metadata") if isinstance(rec, dict) else None
    meta = meta if isinstance(meta, dict) else {}
    if topic is not None and str(meta.get("topic_id", "")) != str(topic):
        return False
    if run is not None and str(meta.get("run_id", "")) != str(run):
        return False
    return True


def scrub_stream(src: Iterable[str], out: TextIO, *, chars: bool,
                 topic: Optional[str] = None, run: Optional[str] = None,
                 index: Optional[int] = None) -> ScrubStats:
    stats = ScrubStats()
    kept = 0
    for i, line in enumerate(src):
        if not line.strip():
            continue
        if index is not None and i != index:
            continue
        if not _selected(line, topic, run):
            continue
        kept += 1
        out.write(scrub_json_line(line, chars, stats) + "\n")
    if kept == 0:
        # A structural fact, not content: the selector matched nothing.
        print("scrub: no record matched the selector", file=sys.stderr)
    return stats


def _log(path: Path, argv: list[str], chars: bool, source: str,
         stats: ScrubStats) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tier": 2 if chars else 1,
        "source": source,
        "argv": argv,
        **stats.as_dict(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError as exc:                      # logging must never lose the output
        print(f"scrub: could not write log {path}: {exc}", file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="scrub",
        description="Content-blind transform for restricted evaluation records. "
                    "Tier 1 keeps structure only; --chars also keeps string "
                    "lengths and punctuation. There is deliberately no option "
                    "to scrub only part of a record.")
    ap.add_argument("input", nargs="?", help="JSONL file (default: stdin)")
    ap.add_argument("--chars", action="store_true",
                    help="tier 2: preserve string lengths and non-alphanumerics. "
                         "Leaks a run's formatting template -- do not commit the "
                         "output or compare it across runs.")
    ap.add_argument("--topic", help="select records with this topic_id")
    ap.add_argument("--run", help="select records with this run_id")
    ap.add_argument("--index", type=int, help="select the Nth line (0-based)")
    ap.add_argument("--output", help="write here instead of stdout")
    ap.add_argument("--log", default=os.environ.get("SCRUB_LOG", DEFAULT_LOG),
                    help=f"scrub log to append to (default: {DEFAULT_LOG})")
    args = ap.parse_args(argv)

    # Tier 2 preserves a run's formatting template, which is a fingerprint. One
    # record at a time is a debugging aid; a whole file of them is a table you
    # can compare across runs, which is identification. Tier 1 is uniform by
    # construction, so bulk is harmless there.
    if args.chars and args.topic is None and args.run is None and args.index is None:
        print("scrub: --chars needs a selector (--topic / --run / --index): "
              "it preserves each run's formatting template, so scrubbing a whole "
              "file at once produces a fingerprint table, not a reproducer.",
              file=sys.stderr)
        return 2

    src: Iterable[str]
    if args.input:
        path = Path(args.input)
        if not path.is_file():
            print(f"scrub: no such file: {path}", file=sys.stderr)
            return 2
        fh_in = open(path, "r", encoding="utf-8")
        src, source = fh_in, str(path)
    else:
        fh_in = None
        src, source = sys.stdin, "<stdin>"

    fh_out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        stats = scrub_stream(src, fh_out, chars=args.chars, topic=args.topic,
                             run=args.run, index=args.index)
    finally:
        if fh_in is not None:
            fh_in.close()
        if args.output:
            fh_out.close()

    _log(Path(args.log), (argv if argv is not None else sys.argv[1:]),
         args.chars, source, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
