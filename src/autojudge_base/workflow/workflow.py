"""
Workflow declaration for AutoJudge nugget/judge pipelines.

Participants declare their workflow in workflow.yml to enable TIRA orchestration.
"""

import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator


# Built-in NuggetBanks type paths (for convenience)
NUGGET_BANKS_AUTOARGUE = "trec_auto_judge.nugget_data.NuggetBanks"
NUGGET_BANKS_NUGGETIZER = "trec_auto_judge.nugget_data.NuggetizerNuggetBanks"

# Default type
DEFAULT_NUGGET_BANKS_TYPE = NUGGET_BANKS_AUTOARGUE


class Workflow(BaseModel):
    """
    Workflow configuration loaded from workflow.yml.

    Controls which steps are executed:
    - create_nuggets: Whether to call create_nuggets() to generate/refine nuggets
    - judge: Whether to call judge() to produce leaderboard/qrels

    Settings are passed to AutoJudge methods as **kwargs:
    - settings: Shared settings for both phases (fallback)
    - nugget_settings: Override for create_nuggets()
    - judge_settings: Override for judge()

    Example workflow.yml:
        create_nuggets: true
        judge: true

        settings:
          filebase: "{_name}"
          top_k: 20

        nugget_settings:
          extraction_style: "thorough"

        judge_settings:
          threshold: 0.5

        variants:
          strict:
            threshold: 0.8

        sweeps:
          top-k-sweep:
            top_k: [10, 20, 50]
    """

    create_nuggets: Optional[bool] = None
    """Whether to call create_nuggets(). Derived from judge_uses_nuggets | qrels_uses_nuggets if not set."""

    create_qrels: Optional[bool] = None
    """Whether to call create_qrels(). Derived from judge_uses_qrels if not set."""

    judge: bool = True
    """Whether to call judge() to produce leaderboard."""

    judge_class: Optional[str] = None
    """Dotted import path for LeaderboardJudgeProtocol class.

    When specified, the judge is loaded dynamically from this path.
    If not specified, falls back to a single class that implements all protocols.
    """

    qrels_class: Optional[str] = None
    """Dotted import path for QrelsCreatorProtocol class.

    When specified, qrels creation uses this class.
    If not specified, falls back to judge_class (assumes it implements QrelsCreatorProtocol).
    """

    nugget_class: Optional[str] = None
    """Dotted import path for NuggetCreatorProtocol class.

    When specified, nugget creation uses this class.
    If not specified, falls back to judge_class (assumes it implements NuggetCreatorProtocol).
    """

    @model_validator(mode="after")
    def derive_creation_flags(self) -> "Workflow":
        """Derive create_nuggets and create_qrels from usage flags if not explicitly set."""
        if self.create_nuggets is None:
            # Need nuggets if judge or qrels creation uses them
            self.create_nuggets = self.judge_uses_nuggets or self.qrels_uses_nuggets
        if self.create_qrels is None:
            # Need qrels if judge uses them
            self.create_qrels = self.judge_uses_qrels
        return self

    nugget_banks_type: str = DEFAULT_NUGGET_BANKS_TYPE
    """Dotted import path for NuggetBanks container class."""

    nugget_input: Optional[str] = None
    """Path to existing nugget banks to load (for refinement or judge input)."""

    nugget_output: Optional[str] = None
    """Path to store created/refined nugget banks."""

    # Settings dicts passed to AutoJudge methods as **kwargs
    settings: dict[str, Any] = Field(default_factory=dict)
    """Shared settings passed to both create_nuggets() and judge()."""

    nugget_settings: dict[str, Any] = Field(default_factory=dict)
    """Settings passed to create_nuggets(), merged over shared settings."""

    judge_settings: dict[str, Any] = Field(default_factory=dict)
    """Settings passed to judge(), merged over shared settings."""

    # Named configurations and parameter sweeps
    variants: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Named configurations that override settings. Key = variant name."""

    sweeps: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Parameter sweeps. Values with lists expand to multiple configurations."""

    # Lifecycle flags
    nugget_depends_on_responses: bool = True
    """If True, pass responses to create_nuggets(). If False, pass None."""

    judge_uses_nuggets: bool = True
    """If True, pass nuggets to judge(). If False, pass None."""

    force_recreate_nuggets: Optional[bool] = None
    """If True, recreate nuggets even if file exists. If None, defaults to create_nuggets value. CLI can override."""

    augment_report: bool = False
    """If True, judge modifies Report.evaldata; save to {filebase}.responses.jsonl."""

    # Qrels lifecycle flags
    judge_uses_qrels: bool = True
    """If True, pass qrels to judge(). If False, pass None."""

    qrels_uses_nuggets: bool = True
    """If True, pass nuggets to create_qrels(). If False, pass None."""

    qrels_settings: dict[str, Any] = Field(default_factory=dict)
    """Settings passed to create_qrels(), merged over shared settings.

    Example:
        qrels_settings:
          grade_range: [0, 3]  # Valid relevance grades (min, max)
    """

    qrels_input: Optional[str] = None
    """Path to existing qrels to load (for refinement or judge input)."""

    qrels_output: Optional[str] = None
    """Path to store created qrels."""

    force_recreate_qrels: Optional[bool] = None
    """If True, recreate qrels even if file exists. If None, defaults to create_qrels value."""


@dataclass
class ResolvedConfiguration:
    """Resolved configuration for a single run (variant or sweep point)."""

    name: str
    """Variant/sweep name, or 'default' when running without variant."""

    settings: dict[str, Any]
    """Shared settings (fallback for all phases)."""

    nugget_settings: dict[str, Any]
    """Merged settings for create_nuggets()."""

    judge_settings: dict[str, Any]
    """Merged settings for judge()."""

    qrels_settings: dict[str, Any]
    """Merged settings for create_qrels()."""

    nugget_input_path: Optional[Path]
    """Resolved path for nugget input (from workflow.nugget_input or None)."""

    nugget_output_path: Optional[Path]
    """Resolved path for nugget output (workflow.nugget_output overrides filebase)."""

    judge_output_path: Optional[Path]
    """Resolved path for judge output (from judge_settings.filebase)."""

    qrels_input_path: Optional[Path]
    """Resolved path for qrels input (from workflow.qrels_input or None)."""

    qrels_output_path: Optional[Path]
    """Resolved path for qrels output (workflow.qrels_output overrides filebase)."""


# Reserved parameter names that would collide with AutoJudge protocol
RESERVED_PARAM_NAMES = frozenset({
    "rag_responses",
    "rag_topics",
    "llm_config",
    "nugget_banks",
})


def _validate_no_underscore_params(settings: dict[str, Any], context: str) -> None:
    """Raise error if any parameter starts with underscore (reserved for built-ins)."""
    for key in settings:
        if key.startswith("_"):
            raise ValueError(
                f"Parameter '{key}' in {context} starts with underscore. "
                "Parameters starting with '_' are reserved for built-in variables."
            )


def _validate_no_reserved_params(settings: dict[str, Any], context: str) -> None:
    """Raise error if any parameter collides with AutoJudge protocol parameters."""
    for key in settings:
        if key in RESERVED_PARAM_NAMES:
            raise ValueError(
                f"Parameter '{key}' in {context} is reserved. "
                f"These names collide with AutoJudge protocol: {sorted(RESERVED_PARAM_NAMES)}"
            )


def _validate_settings(settings: dict[str, Any], context: str) -> None:
    """Validate a settings dict for reserved names."""
    _validate_no_underscore_params(settings, context)
    _validate_no_reserved_params(settings, context)


def _validate_workflow(workflow: Workflow) -> None:
    """Validate workflow configuration on load."""
    _validate_settings(workflow.settings, "settings")
    _validate_settings(workflow.nugget_settings, "nugget_settings")
    _validate_settings(workflow.judge_settings, "judge_settings")
    _validate_settings(workflow.qrels_settings, "qrels_settings")

    for name, variant in workflow.variants.items():
        _validate_settings(variant, f"variants.{name}")
        if "nugget_settings" in variant:
            _validate_settings(variant["nugget_settings"], f"variants.{name}.nugget_settings")
        if "judge_settings" in variant:
            _validate_settings(variant["judge_settings"], f"variants.{name}.judge_settings")
        if "qrels_settings" in variant:
            _validate_settings(variant["qrels_settings"], f"variants.{name}.qrels_settings")

    for name, sweep in workflow.sweeps.items():
        _validate_settings(sweep, f"sweeps.{name}")
        if "nugget_settings" in sweep:
            _validate_settings(sweep["nugget_settings"], f"sweeps.{name}.nugget_settings")
        if "judge_settings" in sweep:
            _validate_settings(sweep["judge_settings"], f"sweeps.{name}.judge_settings")
        if "qrels_settings" in sweep:
            _validate_settings(sweep["qrels_settings"], f"sweeps.{name}.qrels_settings")


def _substitute_template(template: str, variables: dict[str, Any]) -> str:
    """
    Substitute {var} placeholders in template string.

    Args:
        template: String with {var} placeholders
        variables: Dict of variable values (including built-ins like _name)

    Returns:
        Substituted string
    """
    def replace(match: re.Match) -> str:
        var_name = match.group(1)
        if var_name not in variables:
            raise KeyError(f"Unknown variable '{{{var_name}}}' in template '{template}'")
        return str(variables[var_name])

    return re.sub(r"\{(\w+)\}", replace, template)


def _resolve_filebase(
    settings: dict[str, Any],
    variables: dict[str, Any],
    default_filebase: str = "{_name}",
) -> Optional[Path]:
    """
    Resolve filebase from settings to a Path.

    Args:
        settings: Settings dict (may contain 'filebase' key)
        variables: Variables for template substitution
        default_filebase: Default template if filebase not in settings

    Returns:
        Resolved Path, or None if filebase resolves to empty
    """
    filebase_template = settings.get("filebase", default_filebase)
    if not filebase_template:
        return None
    resolved = _substitute_template(filebase_template, variables)
    return Path(resolved) if resolved else None


def _merge_settings(
    base: dict[str, Any],
    *overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge multiple settings dicts, later ones override earlier."""
    result = dict(base)
    for override in overrides:
        result.update(override)
    return result


def _resolve_template_path(
    template: Optional[str],
    variables: dict[str, Any],
) -> Optional[Path]:
    """Resolve an optional template string to a Path."""
    if not template:
        return None
    resolved = _substitute_template(template, variables)
    return Path(resolved) if resolved else None


@dataclass
class _ResolvedPaths:
    """Internal container for resolved paths."""
    nugget_input_path: Optional[Path]
    nugget_output_path: Optional[Path]
    judge_output_path: Optional[Path]
    qrels_input_path: Optional[Path]
    qrels_output_path: Optional[Path]


def _resolve_output_paths(
    name: str,
    merged_nugget: dict[str, Any],
    merged_judge: dict[str, Any],
    merged_qrels: dict[str, Any],
    nugget_input_override: Optional[str] = None,
    nugget_output_override: Optional[str] = None,
    qrels_input_override: Optional[str] = None,
    qrels_output_override: Optional[str] = None,
) -> _ResolvedPaths:
    """
    Resolve nugget, qrels, and judge paths.

    Args:
        name: Configuration name (used as _name variable)
        merged_nugget: Merged nugget settings
        merged_judge: Merged judge settings
        merged_qrels: Merged qrels settings
        nugget_input_override: Explicit nugget input path (workflow.nugget_input)
        nugget_output_override: Explicit nugget output path (workflow.nugget_output)
        qrels_input_override: Explicit qrels input path (workflow.qrels_input)
        qrels_output_override: Explicit qrels output path (workflow.qrels_output)

    Returns:
        _ResolvedPaths with all path fields
    """
    nugget_vars = {"_name": name, **merged_nugget}

    # Nugget input: explicit override only (no filebase fallback)
    nugget_input_path = _resolve_template_path(nugget_input_override, nugget_vars)

    # Nugget output: explicit override, then filebase fallback
    if nugget_output_override:
        nugget_output_path = _resolve_template_path(nugget_output_override, nugget_vars)
    else:
        nugget_output_path = _resolve_filebase(merged_nugget, nugget_vars)

    # Qrels variables include nugget filebase
    qrels_vars = {
        "_name": name,
        "_nugget_filebase": str(nugget_output_path) if nugget_output_path else "",
        **merged_qrels,
    }

    # Qrels input: explicit override only
    qrels_input_path = _resolve_template_path(qrels_input_override, qrels_vars)

    # Qrels output: explicit override, then filebase fallback
    if qrels_output_override:
        qrels_output_path = _resolve_template_path(qrels_output_override, qrels_vars)
    else:
        qrels_output_path = _resolve_filebase(merged_qrels, qrels_vars)

    # Judge variables include both nugget and qrels filebases
    judge_vars = {
        "_name": name,
        "_nugget_filebase": str(nugget_output_path) if nugget_output_path else "",
        "_qrels_filebase": str(qrels_output_path) if qrels_output_path else "",
        **merged_judge,
    }
    judge_output_path = _resolve_filebase(merged_judge, judge_vars)

    return _ResolvedPaths(
        nugget_input_path=nugget_input_path,
        nugget_output_path=nugget_output_path,
        judge_output_path=judge_output_path,
        qrels_input_path=qrels_input_path,
        qrels_output_path=qrels_output_path,
    )


def resolve_default(workflow: Workflow) -> ResolvedConfiguration:
    """
    Resolve default configuration (no variant/sweep).

    Args:
        workflow: Workflow configuration

    Returns:
        ResolvedConfiguration with name='default'
    """
    name = "default"

    # Merge settings
    merged_nugget = _merge_settings(workflow.settings, workflow.nugget_settings)
    merged_judge = _merge_settings(workflow.settings, workflow.judge_settings)
    merged_qrels = _merge_settings(workflow.settings, workflow.qrels_settings)

    paths = _resolve_output_paths(
        name, merged_nugget, merged_judge, merged_qrels,
        nugget_input_override=workflow.nugget_input,
        nugget_output_override=workflow.nugget_output,
        qrels_input_override=workflow.qrels_input,
        qrels_output_override=workflow.qrels_output,
    )

    return ResolvedConfiguration(
        name=name,
        settings=workflow.settings,
        nugget_settings=merged_nugget,
        judge_settings=merged_judge,
        qrels_settings=merged_qrels,
        nugget_input_path=paths.nugget_input_path,
        nugget_output_path=paths.nugget_output_path,
        judge_output_path=paths.judge_output_path,
        qrels_input_path=paths.qrels_input_path,
        qrels_output_path=paths.qrels_output_path,
    )


def resolve_variant(workflow: Workflow, variant_name: str) -> ResolvedConfiguration:
    """
    Resolve a named variant configuration.

    Args:
        workflow: Workflow configuration
        variant_name: Name of variant to resolve

    Returns:
        ResolvedConfiguration for the variant

    Raises:
        KeyError: If variant_name not found in workflow.variants
    """
    if variant_name not in workflow.variants:
        raise KeyError(f"Variant '{variant_name}' not found. Available: {list(workflow.variants.keys())}")

    variant = workflow.variants[variant_name]

    # Extract variant's phase-specific settings if present
    variant_nugget = variant.get("nugget_settings", {})
    variant_judge = variant.get("judge_settings", {})
    variant_qrels = variant.get("qrels_settings", {})
    # Remove phase-specific keys from variant for shared settings
    variant_shared = {k: v for k, v in variant.items() if k not in ("nugget_settings", "judge_settings", "qrels_settings")}

    # Merge: base settings -> phase settings -> variant shared -> variant phase
    merged_nugget = _merge_settings(
        workflow.settings, workflow.nugget_settings, variant_shared, variant_nugget
    )
    merged_judge = _merge_settings(
        workflow.settings, workflow.judge_settings, variant_shared, variant_judge
    )
    merged_qrels = _merge_settings(
        workflow.settings, workflow.qrels_settings, variant_shared, variant_qrels
    )

    paths = _resolve_output_paths(
        variant_name, merged_nugget, merged_judge, merged_qrels,
        nugget_input_override=workflow.nugget_input,
        nugget_output_override=workflow.nugget_output,
        qrels_input_override=workflow.qrels_input,
        qrels_output_override=workflow.qrels_output,
    )

    return ResolvedConfiguration(
        name=variant_name,
        settings=_merge_settings(workflow.settings, variant_shared),
        nugget_settings=merged_nugget,
        judge_settings=merged_judge,
        qrels_settings=merged_qrels,
        nugget_input_path=paths.nugget_input_path,
        nugget_output_path=paths.nugget_output_path,
        judge_output_path=paths.judge_output_path,
        qrels_input_path=paths.qrels_input_path,
        qrels_output_path=paths.qrels_output_path,
    )


def resolve_sweep(workflow: Workflow, sweep_name: str) -> list[ResolvedConfiguration]:
    """
    Resolve a parameter sweep to multiple configurations.

    Args:
        workflow: Workflow configuration
        sweep_name: Name of sweep to resolve

    Returns:
        List of ResolvedConfiguration, one per sweep point (cartesian product)

    Raises:
        KeyError: If sweep_name not found in workflow.sweeps
    """
    if sweep_name not in workflow.sweeps:
        raise KeyError(f"Sweep '{sweep_name}' not found. Available: {list(workflow.sweeps.keys())}")

    sweep = workflow.sweeps[sweep_name]

    # Separate sweep params (lists) from fixed params
    sweep_params = {}
    fixed_params = {}
    for key, value in sweep.items():
        if isinstance(value, list) and key not in ("nugget_settings", "judge_settings", "qrels_settings"):
            sweep_params[key] = value
        else:
            fixed_params[key] = value

    # Extract phase-specific settings from sweep
    sweep_nugget = fixed_params.pop("nugget_settings", {})
    sweep_judge = fixed_params.pop("judge_settings", {})
    sweep_qrels = fixed_params.pop("qrels_settings", {})

    # Generate cartesian product of sweep params
    if not sweep_params:
        # No lists to expand, just one configuration
        param_combinations = [{}]
    else:
        keys = list(sweep_params.keys())
        values = [sweep_params[k] for k in keys]
        param_combinations = [dict(zip(keys, combo)) for combo in product(*values)]

    results = []
    for combo in param_combinations:
        # Merge: base -> phase -> sweep fixed -> sweep phase -> combo
        merged_nugget = _merge_settings(
            workflow.settings, workflow.nugget_settings, fixed_params, sweep_nugget, combo
        )
        merged_judge = _merge_settings(
            workflow.settings, workflow.judge_settings, fixed_params, sweep_judge, combo
        )
        merged_qrels = _merge_settings(
            workflow.settings, workflow.qrels_settings, fixed_params, sweep_qrels, combo
        )

        paths = _resolve_output_paths(
            sweep_name, merged_nugget, merged_judge, merged_qrels,
            nugget_input_override=workflow.nugget_input,
            nugget_output_override=workflow.nugget_output,
            qrels_input_override=workflow.qrels_input,
            qrels_output_override=workflow.qrels_output,
        )

        results.append(ResolvedConfiguration(
            name=sweep_name,
            settings=_merge_settings(workflow.settings, fixed_params, combo),
            nugget_settings=merged_nugget,
            judge_settings=merged_judge,
            qrels_settings=merged_qrels,
            nugget_input_path=paths.nugget_input_path,
            nugget_output_path=paths.nugget_output_path,
            judge_output_path=paths.judge_output_path,
            qrels_input_path=paths.qrels_input_path,
            qrels_output_path=paths.qrels_output_path,
        ))

    return results


def load_workflow(source: Union[str, Path]) -> Workflow:
    """
    Load workflow configuration from a YAML file.

    Args:
        source: Path to workflow.yml

    Returns:
        Workflow configuration

    Raises:
        ValueError: If workflow contains invalid parameters (e.g., starting with '_')

    Example workflow.yml:
        create_nuggets: true
        judge: true
    """
    path = Path(source)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    workflow = Workflow.model_validate(data)
    _validate_workflow(workflow)
    return workflow


def load_workflow_from_directory(directory: Union[str, Path]) -> Optional[Workflow]:
    """
    Load workflow.yml from a judge directory if it exists.

    Args:
        directory: Judge directory (e.g., trec25/judges/my-judge/)

    Returns:
        Workflow if workflow.yml exists, None otherwise
    """
    path = Path(directory) / "workflow.yml"
    if path.is_file():
        return load_workflow(path)
    return None


# Default workflow for judges that don't declare one (judge only, no nuggets)
DEFAULT_WORKFLOW = Workflow()


@dataclass
class LoadedJudgeComponents:
    """Container for loaded judge protocol implementations."""
    nugget_creator: Optional[object]  # NuggetCreatorProtocol
    qrels_creator: Optional[object]   # QrelsCreatorProtocol
    leaderboard_judge: Optional[object]  # LeaderboardJudgeProtocol


def load_judge_from_workflow(workflow: Workflow) -> LoadedJudgeComponents:
    """
    Load and instantiate judge classes from workflow configuration.

    Supports three separate class fields for modular judge composition:
    - nugget_class: NuggetCreatorProtocol implementation
    - qrels_class: QrelsCreatorProtocol implementation
    - judge_class: LeaderboardJudgeProtocol implementation

    If only judge_class is set, it's used for all three protocols (assumes
    the class implements all of them, like the combined AutoJudge protocol).

    Args:
        workflow: Workflow configuration

    Returns:
        LoadedJudgeComponents with instantiated protocol implementations

    Raises:
        ValueError: If no judge classes are specified
        ImportError: If a class cannot be imported
        TypeError: If a class cannot be instantiated

    Example workflow.yml (single class for all):
        judge_class: "trec25.judges.minimaljudge.MinimalJudge"

    Example workflow.yml (separate classes):
        nugget_class: "trec25.judges.shared.NuggetGenerator"
        qrels_class: "trec25.judges.umbrela.UmbrelaQrels"
        judge_class: "trec25.judges.myjudge.MyLeaderboard"
    """
    from ..utils import import_class

    # Check that at least one class is specified
    if not workflow.judge_class and not workflow.nugget_class and not workflow.qrels_class:
        raise ValueError(
            "No judge classes specified in workflow. "
            "Set at least one of: judge_class, nugget_class, qrels_class"
        )

    # Cache for instantiated classes (reuse if same class used for multiple protocols)
    _instances: dict[str, object] = {}

    def _get_instance(class_path: Optional[str]) -> Optional[object]:
        if not class_path:
            return None
        if class_path not in _instances:
            cls = import_class(class_path)
            _instances[class_path] = cls()
        return _instances[class_path]

    # Resolve each protocol's class (fall back to judge_class if not specified)
    nugget_class_path = workflow.nugget_class or workflow.judge_class
    qrels_class_path = workflow.qrels_class or workflow.judge_class
    judge_class_path = workflow.judge_class

    return LoadedJudgeComponents(
        nugget_creator=_get_instance(nugget_class_path),
        qrels_creator=_get_instance(qrels_class_path),
        leaderboard_judge=_get_instance(judge_class_path),
    )