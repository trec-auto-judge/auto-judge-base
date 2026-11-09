"""Merge several nugget-bank containers into one, as a union per topic.

Nugget banks cannot be concatenated when they share topics: ``from_banks_list`` (and so
``load_nugget_banks_from_directory``) rejects a duplicate ``query_id``. ``merge_nugget_banks``
unions them instead, combining each topic's banks with ``NuggetBank.add_nuggets``, which
keys a nugget by its text and calls ``merge_nugget_questions`` / ``merge_nugget_claims`` on
a collision. Identical nuggets across sources collapse into one; distinct ones coexist.
Because nuggets are keyed by a content hash, the union needs no id namespacing.

``merge_nugget_bank_files`` loads files and merges them, skipping any that do not parse
(participant banks come in several schemas). The module is runnable as a CLI:

  python -m autojudge_base.nugget_data.merge --out OUT.nuggets.jsonl \
      [--topic ID ...] BANK.nuggets.jsonl ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel

from .nugget_banks import (NuggetBanks, load_nugget_banks_from_file,
                           write_nugget_banks)

C = TypeVar("C", bound=BaseModel)   # a NuggetBanks-like container (.banks, .from_banks_list)


def merge_nugget_banks(sources: Iterable[C],
                       topics: Optional[Collection[str]] = None) -> C:
    """Union several nugget-bank containers into one, per topic.

    Each topic gets a fresh bank into which every source's nuggets are added with
    ``add_nuggets`` (merge on a text collision), so identical nuggets across sources
    collapse while distinct ones coexist and the inputs are left untouched. ``topics``,
    when given, restricts to those request ids. All sources must be the same container
    type; raises ValueError if ``sources`` is empty.
    """
    want = set(topics) if topics is not None else None
    target: Dict[str, Any] = {}
    container_type: Optional[Type[C]] = None
    for banks in sources:
        container_type = type(banks)
        for tid, bank in banks.banks.items():
            if want is not None and tid not in want:
                continue
            tgt = target.get(tid)
            if tgt is None:
                tgt = type(bank)(query_id=tid,
                                 title_query=getattr(bank, "title_query", None))
                target[tid] = tgt
            tgt.add_nuggets(bank.nuggets_as_list())
    if container_type is None:
        raise ValueError("merge_nugget_banks: no sources given")
    return container_type.from_banks_list(list(target.values()))


def merge_nugget_bank_files(paths: Iterable[Path],
                            topics: Optional[Collection[str]] = None,
                            skip_unreadable: bool = True,
                            ) -> Tuple[NuggetBanks, List[Tuple[Path, Exception]]]:
    """Load bank files and merge them per topic; return (merged, skipped).

    A file that does not load as the standard ``NuggetBank`` schema is recorded in
    ``skipped`` rather than aborting the merge, unless ``skip_unreadable`` is False.
    """
    loaded: List[NuggetBanks] = []
    skipped: List[Tuple[Path, Exception]] = []
    for p in paths:
        path = Path(p)
        try:
            loaded.append(load_nugget_banks_from_file(path))
        except Exception as exc:
            if not skip_unreadable:
                raise
            skipped.append((path, exc))
    return merge_nugget_banks(loaded, topics), skipped


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("banks", nargs="+", type=Path, help="nugget bank files to merge")
    ap.add_argument("--out", required=True, type=Path, help="merged output .nuggets.jsonl")
    ap.add_argument("--topic", action="append", default=[],
                    help="restrict to this request_id; repeat (default: all topics)")
    args = ap.parse_args()

    topics = set(args.topic) if args.topic else None
    merged, skipped = merge_nugget_bank_files(args.banks, topics)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_nugget_banks(merged, args.out)

    print("merged %d bank file(s) into %s" % (len(args.banks) - len(skipped), args.out))
    for tid in sorted(merged.banks):
        print("  %s: %d nuggets" % (tid, len(merged.banks[tid].nuggets_as_list())))
    if topics:
        missing = sorted(topics - set(merged.banks))
        if missing:
            print("  no nuggets found for: %s" % ", ".join(missing))
    if skipped:
        print("  skipped %d unreadable bank(s): %s"
              % (len(skipped), ", ".join("%s (%s)" % (p.name, type(e).__name__)
                                         for p, e in skipped)))


if __name__ == "__main__":
    main()
