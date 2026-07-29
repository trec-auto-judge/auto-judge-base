"""Verification for a COLLECTION of reports (a submission), mirroring the Leaderboard/
Qrels `verification.py` pattern.

`LeaderboardVerification` verifies a whole leaderboard with fluent, chainable checks
(complete_measures, complete_topics, no_extra_topics, ...). `ReportVerification` is the
report analog: it verifies a list of reports against a track spec plus the topics file.

Two layers cooperate:
- `report_spec_verification` is the per-report RULE ENGINE (does one report obey the
  spec). `spec_compliant()` runs it over every report.
- this module adds the CROSS-REPORT checks a single report cannot see -- every expected
  topic present (`complete_topics`), no unexpected topics (`no_extra_topics`), no
  duplicate topic objects (`unique_topics`), and the exact-narrative copy rule
  (`exact_narratives`) -- exactly like Leaderboard's complete_topics / no_extra_topics.

Findings accumulate into `self.findings`, each tagged `error` or `warning`. Whether a
category is a warning is driven by `spec.smells` (the report analog of Leaderboard's
`warn` flag), overridable per run with `strict=True` (every finding becomes an error).
Call `.raise_first()` for fail-fast (Leaderboard-style), or read `.errors`/`.warnings`.
"""
import os
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from autojudge_base.track_spec import SPECS
from autojudge_base.report_spec_verification import findings as _report_findings


class ReportVerificationError(Exception):
    """Raised when report-collection verification fails (fail-fast)."""


@dataclass(frozen=True)
class Finding:
    """One verification finding. `origin` = (local_path, team_id, run_id) and `topic`
    identify the offending report for per-report (spec) findings; both are None for
    aggregate coverage findings. `label` is the short display tag for coverage findings
    (e.g. 'MISSING')."""
    category: str
    message: str
    severity: str                 # "error" | "warning"
    origin: Optional[tuple] = None
    topic: Optional[str] = None
    label: Optional[str] = None


# --- small pure helpers (also useful to callers presenting findings) ------------

def local_path(p) -> str:
    """Render a report's source path relative to the current working directory."""
    if p is None:
        return "?"
    try:
        return os.path.relpath(str(p))
    except ValueError:                 # different drive on Windows -> keep absolute
        return str(p)


def sortkey(x):
    """Sort ids numerically when possible, else lexically (numbers before strings)."""
    try:
        return (0, int(x))
    except ValueError:
        return (1, str(x))


def compress_ids(tids) -> str:
    """Compress integer-like ids into sorted ranges, e.g. '1001-1005, 1008'.
    Raises ValueError if any id is not an integer."""
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


def fmt_ids(ids) -> str:
    """Compact rendering of an id list: integer ranges, or a truncated plain list."""
    try:
        return compress_ids(ids)
    except ValueError:
        head = ", ".join(ids[:12])
        return head + (f", ... (+{len(ids) - 12} more)" if len(ids) > 12 else "")


class ReportVerification:
    """Fluent verifier for a collection of reports (a submission).

        ReportVerification(reports, "rag26", requests=reqs).all().raise_first()

    or read `.errors` / `.warnings` (lists of Finding) after running some checks. `spec`
    may be a TrackSpec, a track-id string, or None (structural-only). `requests` maps
    topic-id -> Request (supplies per-request length limits + the coverage cross-check).
    `strict` demotes nothing to a warning (every finding is a hard error).
    """

    _COV_LABEL = {"missing_topics": "MISSING", "new_topics": "NEW",
                  "duplicate_topics": "DUP", "narrative": "NARRATIVE"}

    def __init__(self, reports, spec=None, *, requests: Optional[dict] = None,
                 strict: bool = False):
        self.reports = list(reports)
        self._spec_arg = spec
        self.spec = SPECS.get(spec) if isinstance(spec, str) else spec  # resolved or None
        self.requests = requests or {}
        self.strict = strict
        self.findings: List[Finding] = []
        self.ok = 0                       # reports with no hard-error spec finding
        self._smells = () if strict else (self.spec.smells if self.spec is not None else ())

    # --- accessors ---
    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    def _add(self, category, message, *, origin=None, topic=None, label=None) -> None:
        severity = "warning" if category in self._smells else "error"
        self.findings.append(Finding(category, message, severity, origin, topic, label))

    def _expected(self) -> dict:
        return {tid: getattr(req, "title", None) for tid, req in self.requests.items()}

    # --- per-report spec rules (delegates to report_spec_verification) ---
    def spec_compliant(self) -> "ReportVerification":
        """Verify every report against the spec's rules (metadata, citations, references,
        docid pattern, length). Records each finding with its offending report."""
        for r in self.reports:
            md = r.metadata
            tid = str(md.topic_id)
            errs, warns = _report_findings(r, self._spec_arg, request=self.requests.get(tid))
            if self.strict:
                errs, warns = errs + warns, []
            if not errs:
                self.ok += 1
            origin = (local_path(getattr(r, "path", None)), md.team_id, md.run_id)
            for m in errs:
                self.findings.append(Finding("spec", m, "error", origin, tid))
            for m in warns:
                self.findings.append(Finding("spec", m, "warning", origin, tid))
        return self

    # --- cross-report coverage checks (a single report cannot see these) ---
    def complete_topics(self) -> "ReportVerification":
        """Every expected topic (from the topics file) has a report. No-op without a
        topics file (nothing to compare against)."""
        if not self.requests:
            return self
        missing = sorted(set(self._expected()) - {str(r.metadata.topic_id) for r in self.reports},
                         key=sortkey)
        if missing:
            self._add("missing_topics",
                      f"{len(missing)} topic(s) in the topics file have no report: {fmt_ids(missing)}",
                      label="MISSING")
        return self

    def no_extra_topics(self) -> "ReportVerification":
        """No report carries a topic id absent from the topics file. No-op without a
        topics file (every topic would look 'extra')."""
        if not self.requests:
            return self
        new = sorted({str(r.metadata.topic_id) for r in self.reports} - set(self._expected()),
                     key=sortkey)
        if new:
            self._add("new_topics",
                      f"{len(new)} report topic id(s) not in the topics file: {fmt_ids(new)}",
                      label="NEW")
        return self

    def unique_topics(self) -> "ReportVerification":
        """No topic id appears in more than one report object."""
        ids = [str(r.metadata.topic_id) for r in self.reports]
        dups = sorted((t for t, c in Counter(ids).items() if c > 1), key=sortkey)
        if dups:
            self._add("duplicate_topics",
                      f"{len(dups)} topic id(s) appear more than once: {fmt_ids(dups)}",
                      label="DUP")
        return self

    def exact_narratives(self) -> "ReportVerification":
        """For exact-narrative specs, each report's narrative copies the topic text."""
        if self.spec is None or not self.spec.require_exact_narrative:
            return self
        expected = self._expected()
        mism = sorted({str(r.metadata.topic_id) for r in self.reports
                       if str(r.metadata.topic_id) in expected
                       and r.metadata.narrative != expected[str(r.metadata.topic_id)]}, key=sortkey)
        if mism:
            self._add("narrative",
                      f"{len(mism)} report(s): narrative is not an exact copy of the topic text: {fmt_ids(mism)}",
                      label="NARRATIVE")
        return self

    def coverage(self) -> "ReportVerification":
        """Run all cross-report topic-coverage checks (requires `requests`)."""
        return self.complete_topics().no_extra_topics().unique_topics().exact_narratives()

    def all(self) -> "ReportVerification":
        """Per-report spec compliance plus cross-report coverage."""
        return self.spec_compliant().coverage()

    def raise_first(self) -> "ReportVerification":
        """Fail-fast: raise ReportVerificationError on the first hard error (Leaderboard
        style). Warnings (smells) never raise."""
        for f in self.findings:
            if f.severity == "error":
                raise ReportVerificationError(f.message)
        return self
