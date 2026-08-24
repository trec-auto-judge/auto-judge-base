"""``scrub`` -- the only sanctioned path from a restricted record to a developer's view.

TREC AutoJudge evaluation data is anonymised and mostly off-limits to the coding
agent building a judge: the judge may read every report, the agent may read only
the permitted window of topics. That rule is unworkable without a way to debug a
crash on a restricted record, so this module provides a fixed, content-blind
transform that turns such a record into something safe to look at.

Two tiers:

* **tier 1** (default) keeps the shape and throws away the text: schema keys,
  types, nesting, list lengths, booleans and nulls survive; every string becomes
  ``wiffle waffle``. Enough for missing fields, wrong types, empty lists and bad
  nesting. It carries nothing from the strings, so tier-1 output is a permitted
  fixture: inspect it, commit it, debug on it.
* **tier 2** (``chars=True``) keeps string *lengths* and every non-alphanumeric
  character. Encoding, parsing, markup, whitespace and length bugs reproduce on
  it. It leaks a run's formatting template, which is a fingerprint, so tier-2
  output must not be committed or compared across runs.

What tier 2 does to a character
-------------------------------
* ASCII letters become ``a``; ASCII digits become ``1``.
* **Diacritics survive.** Text is decomposed (NFD) before substitution and
  recomposed (NFC) after, so the base letter is replaced while its combining
  marks are kept: ``café`` becomes ``aaaá``. Accents, cedillas and umlauts are
  what normalisation and collation bugs are made of, and they carry almost no
  content once the letters are gone.
* Letters and digits outside ASCII are replaced by a representative **from the
  same script with the same UTF-8 byte length** -- CJK for CJK, Arabic for
  Arabic. A single CJK character is a morpheme, so leaving it would leave the
  text readable; replacing it with ``a`` would turn three bytes into one and
  hide the very width and encoding bugs tier 2 exists for.
* Everything else -- punctuation, symbols, whitespace, control characters,
  emoji, markup -- passes through untouched.

Design constraints that follow from the policy, and are not negotiable:

* **No partial application.** There is deliberately no option to keep selected
  fields, sample strings, or preserve "just the short ones". Any such knob makes
  the transform something the agent chose, which is inspection.
* **Deterministic and content-independent.** A character's replacement depends
  only on its own class and script, never on the surrounding text, so the output
  cannot encode content the tier was meant to remove.
* **Identifiers are scrubbed too** -- ``run_id``, ``team_id``, ``topic_id``.
  Select the record you want on the *input* side instead; see
  :mod:`autojudge_base.scrub_tool`.
* **Malformed input is preserved, not repaired.** A record with a wrong type or
  a missing field scrubs to a record with a wrong type or a missing field: that
  is the reproducer.

Dict keys are kept when they look like schema (``[A-Za-z_][A-Za-z0-9_.-]*``, at
most 64 characters) and scrubbed otherwise, so a mapping keyed by answer text
does not pass through untouched.
"""
from __future__ import annotations

import dataclasses
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

if TYPE_CHECKING:  # typing only -- keeps scrub free of import cycles
    from .document import Document
    from .leaderboard import Leaderboard
    from .nugget_data import (NuggetBanks, NuggetizerNuggetBanks,
                              Ragtime26NuggetBanks)
    from .qrels import Qrels
    from .report import Report
    from .request import Request

__all__ = [
    "TIER1_STRING", "ScrubStats", "is_schema_key",
    "scrub_string", "scrub_value", "scrub_json_line", "scrub_model",
]

#: Every string becomes this in tier 1 -- a fixed phrase, not a length-preserving
#: filler: tier 1 destroys length as well as content.
TIER1_STRING = "wiffle waffle"

_SCHEMA_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")
_SCHEMA_KEY_MAX = 64

# Representative letter and digit per script, each matching the UTF-8 byte length
# of the characters in its range. (start, end, letter, digit)
_BLOCKS: Tuple[Tuple[int, int, str, str], ...] = (
    (0x0370, 0x03FF, "α", "α"),   # Greek             2 bytes
    (0x0400, 0x04FF, "а", "а"),   # Cyrillic          2 bytes
    (0x0590, 0x05FF, "א", "א"),   # Hebrew            2 bytes
    (0x0600, 0x06FF, "ب", "١"),   # Arabic            2 bytes
    (0x0700, 0x074F, "ܐ", "ܐ"),   # Syriac            2 bytes
    (0x0900, 0x097F, "क", "१"),   # Devanagari        3 bytes
    (0x0980, 0x09FF, "ক", "১"),   # Bengali           3 bytes
    (0x0E00, 0x0E7F, "ก", "๑"),   # Thai              3 bytes
    (0x3040, 0x309F, "あ", "あ"),  # Hiragana          3 bytes
    (0x30A0, 0x30FF, "ア", "ア"),  # Katakana          3 bytes
    (0x3400, 0x4DBF, "㐀", "㐀"),  # CJK Extension A   3 bytes
    (0x4E00, 0x9FFF, "啊", "啊"),  # CJK Unified       3 bytes
    (0xAC00, 0xD7AF, "가", "가"),  # Hangul syllables  3 bytes
    (0xFF00, 0xFFEF, "ａ", "１"),  # Fullwidth forms   3 bytes
)

#: Last-resort substitute by UTF-8 byte length, for scripts not listed above.
_BY_BYTE_LEN: Dict[int, str] = {1: "a", 2: "ѫ", 3: "啊", 4: "\U0001d41a"}


@dataclass
class ScrubStats:
    """Structural counts only -- never a value. Safe to log and to ship."""

    records: int = 0
    strings: int = 0
    chars: int = 0
    keys_scrubbed: int = 0
    parse_errors: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {"records": self.records, "strings": self.strings,
                "chars": self.chars, "keys_scrubbed": self.keys_scrubbed,
                "parse_errors": self.parse_errors}


def is_schema_key(key: Any) -> bool:
    """True when a mapping key looks like schema rather than data."""
    return (isinstance(key, str) and len(key) <= _SCHEMA_KEY_MAX
            and bool(_SCHEMA_KEY.match(key)))


def _substitute(ch: str) -> str:
    """The stand-in for one alphanumeric character, preserving script and bytes."""
    if ch.isascii():
        return "1" if ch.isdigit() else "a"
    cp = ord(ch)
    for start, end, letter, digit in _BLOCKS:
        if start <= cp <= end:
            rep = digit if ch.isdigit() else letter
            if len(rep.encode("utf-8")) == len(ch.encode("utf-8")):
                return rep
            break
    return _BY_BYTE_LEN.get(len(ch.encode("utf-8")), "a")


def scrub_string(s: str, chars: bool = False,
                 stats: Optional[ScrubStats] = None) -> str:
    """Tier-1 or tier-2 replacement for one string."""
    if stats is not None:
        stats.strings += 1
    if not chars:
        return TIER1_STRING
    out = []
    for ch in s:
        if unicodedata.category(ch).startswith("M"):
            out.append(ch)          # a free-standing combining mark (Thai, Arabic
            continue                # vowel signs): keeps its bytes, carries little
        if not ch.isalnum():
            out.append(ch)
            continue
        if ch.isascii():
            out.append("1" if ch.isdigit() else "a")
        else:
            # Decompose ONLY to rescue a Latin accent. `cafe-acute` becomes
            # `a`+acute, recomposed to a single 2-byte character, so the accent
            # survives at the original byte length. Decomposing anything else
            # breaks it: Hangul syllables explode into conjoining Jamo (3 bytes
            # become 6-9), and kana voicing marks have no precomposed form with
            # the substitute, so they would be dropped or appended. Those scripts
            # are substituted whole instead.
            nfd = unicodedata.normalize("NFD", ch)
            if nfd[0].isascii():
                out.append(unicodedata.normalize("NFC", "a" + nfd[1:]))
            else:
                out.append(_substitute(ch))
        if stats is not None:
            stats.chars += 1
    return "".join(out)


def scrub_value(value: Any, chars: bool = False,
                stats: Optional[ScrubStats] = None) -> Any:
    """Recursively scrub a decoded JSON value, preserving its structure exactly.

    Numbers, booleans and nulls pass through: they are schema-shaped facts, and a
    wrong type or an out-of-range count is frequently the bug being chased.
    """
    if isinstance(value, str):
        return scrub_string(value, chars, stats)
    if isinstance(value, list):
        return [scrub_value(v, chars, stats) for v in value]
    if isinstance(value, tuple):
        return tuple(scrub_value(v, chars, stats) for v in value)
    if isinstance(value, dict):
        out: Dict[Any, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and not is_schema_key(k):
                key: Any = scrub_string(k, chars, stats)
                if stats is not None:
                    stats.keys_scrubbed += 1
            else:
                key = k
            out[key] = scrub_value(v, chars, stats)
        return out
    return value


def scrub_json_line(line: str, chars: bool = False,
                    stats: Optional[ScrubStats] = None) -> str:
    """Scrub one JSONL line, keeping malformed input malformed.

    A line that does not parse is usually the reproducer, not a tool failure. In
    tier 1 it becomes a structural error record -- the parser's message names a
    position, not content. In tier 2 the raw text is character-scrubbed, so the
    delimiters, quoting and whitespace that caused the failure survive intact.
    """
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        if stats is not None:
            stats.parse_errors += 1
        if chars:
            return scrub_string(line.rstrip("\n"), True, stats)
        return json.dumps({"__scrub_parse_error__": {
            "msg": exc.msg, "lineno": exc.lineno, "colno": exc.colno,
            "pos": exc.pos, "length": len(line)}})
    if stats is not None:
        stats.records += 1
    return json.dumps(scrub_value(value, chars, stats), ensure_ascii=False)


def scrub_model(
    obj: Union["Report", "Request", "Document", "Leaderboard", "Qrels",
               "NuggetBanks", "NuggetizerNuggetBanks", "Ragtime26NuggetBanks"],
    chars: bool = False,
    stats: Optional[ScrubStats] = None,
) -> Union["Report", "Request", "Document", "Leaderboard", "Qrels",
           "NuggetBanks", "NuggetizerNuggetBanks", "Ragtime26NuggetBanks"]:
    """Scrub a Report, Request, Document, Leaderboard, Qrels, NuggetBanks,
    NuggetizerNuggetBanks, or Ragtime26NuggetBanks, returning the same class.

    The union is spelled out rather than hidden behind an alias so that grepping
    this file for ``Report``, ``Qrels`` or ``NuggetBanks`` finds it.

    Rebuilding as the same class is the point: the result is a fixture the judge
    can load, and a transform that broke the schema fails here rather than at the
    point of use. Pydantic models and frozen dataclasses are both handled --
    ``Leaderboard`` and ``Qrels`` are dataclasses, the rest are pydantic.

    A ``Report`` is restricted in its entirety outside the permitted window --
    response text and citations, references, ranking, and the documents carried
    inside it, because what a system retrieved identifies it as surely as what it
    wrote. A ``Request`` is *not* restricted (a judge cannot be built blind to the
    question); it is accepted here only so a caller can scrub a workflow
    uniformly. A nugget bank built by response-grounded extraction is a digest of
    the responses it came from, so on restricted topics it is restricted content
    and must be read through here rather than directly.
    """
    if stats is not None:
        stats.records += 1
    return _rebuild(obj, chars, stats)


def _rebuild(obj: Any, chars: bool, stats: Optional[ScrubStats]) -> Any:
    """Scrub any object, keeping pydantic models and dataclasses as such."""
    if isinstance(obj, str):
        return scrub_string(obj, chars, stats)
    if obj is None or isinstance(obj, (int, float, bool)):
        return obj
    if hasattr(obj, "model_dump") and hasattr(type(obj), "model_validate"):
        return type(obj).model_validate(
            scrub_value(obj.model_dump(mode="json"), chars, stats))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Rebuild through the constructor so frozen dataclasses survive and
        # nested dataclasses stay dataclasses instead of collapsing to dicts.
        return type(obj)(**{f.name: _rebuild(getattr(obj, f.name), chars, stats)
                            for f in dataclasses.fields(obj) if f.init})
    if isinstance(obj, list):
        return [_rebuild(v, chars, stats) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_rebuild(v, chars, stats) for v in obj)
    if isinstance(obj, dict):
        out: Dict[Any, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and not is_schema_key(k):
                key: Any = scrub_string(k, chars, stats)
                if stats is not None:
                    stats.keys_scrubbed += 1
            else:
                key = k
            out[key] = _rebuild(v, chars, stats)
        return out
    return obj
