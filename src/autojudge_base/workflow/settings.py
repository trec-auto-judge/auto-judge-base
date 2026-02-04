"""
CLI settings utilities for AutoJudge workflow configuration.

Provides:
- KeyValueType: Click parameter type for key=value parsing
- create_cli_default_workflow: Default workflow for CLI usage
- apply_cli_overrides: Apply CLI --set/--nset/--jset to workflow
"""

from typing import Any, Iterable, Optional

import click

from .workflow import Workflow


class KeyValueType(click.ParamType):
    """Click parameter type for key=value pairs with automatic type inference."""

    name = "key=value"

    def convert(self, value, param, ctx):
        if "=" not in value:
            self.fail(f"Expected key=value format, got: {value}", param, ctx)
        key, val = value.split("=", 1)
        return (key, self._parse_value(val))

    def _parse_value(self, val: str):
        """Try to parse as int/float/bool, else keep as string."""
        # Try int
        try:
            return int(val)
        except ValueError:
            pass
        # Try float
        try:
            return float(val)
        except ValueError:
            pass
        # Try bool
        if val.lower() in ("true", "yes"):
            return True
        if val.lower() in ("false", "no"):
            return False
        return val


def create_cli_default_workflow(judge_uses_nuggets: bool) -> Workflow:
    """
    Create default Workflow for CLI usage (no workflow.yml).

    Args:
        judge_uses_nuggets: If True, creates nuggets then judges with them.
                           If False, judge only (no nugget creation).

    Returns:
        Workflow configured for CLI defaults.
    """
    if judge_uses_nuggets:
        return Workflow(
            create_nuggets=True,
            judge=True,
            nugget_depends_on_responses=False,
            judge_uses_nuggets=True,
            # force_recreate_nuggets=None (default) → uses prompt cache
            settings={"filebase": "{_name}"},
        )
    else:
        return Workflow(
            create_nuggets=False,
            judge=True,
            nugget_depends_on_responses=False,
            judge_uses_nuggets=False,
            # force_recreate_nuggets=None (default) → uses prompt cache
            settings={"filebase": "{_name}"},
        )


def apply_cli_workflow_overrides(
    wf: Workflow,
    settings_overrides: Iterable[tuple[str, Any]],
    nugget_settings_overrides: Iterable[tuple[str, Any]],
    judge_settings_overrides: Iterable[tuple[str, Any]],
    nugget_depends_on_responses: Optional[bool] = None,
    judge_uses_nuggets: Optional[bool] = None,
) -> None:
    """
    Apply CLI overrides to workflow in-place.

    Args:
        wf: Workflow to modify
        settings_overrides: --set key=value pairs for shared settings
        nugget_settings_overrides: --nset key=value pairs for nugget settings
        judge_settings_overrides: --jset key=value pairs for judge settings
        nugget_depends_on_responses: Override lifecycle flag (None = use workflow default)
        judge_uses_nuggets: Override lifecycle flag (None = use workflow default)
    """
    # Apply settings overrides
    for key, val in settings_overrides:
        wf.settings[key] = val
    for key, val in nugget_settings_overrides:
        wf.nugget_settings[key] = val
    for key, val in judge_settings_overrides:
        wf.judge_settings[key] = val

    # Apply lifecycle flag overrides (if provided)
    if nugget_depends_on_responses is not None:
        wf.nugget_depends_on_responses = nugget_depends_on_responses
    if judge_uses_nuggets is not None:
        wf.judge_uses_nuggets = judge_uses_nuggets