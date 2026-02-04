from pathlib import Path
import sys
from typing import Optional, Tuple
from .io import load_runs_failsave
from .request import load_requests_from_irds, load_requests_from_file
from .llm import MinimaLlmConfig
from .llm_resolver import ModelPreferences, ModelResolver, ModelResolutionError
from .workflow import (
    load_workflow,
    resolve_default,
    resolve_variant,
    resolve_sweep,
    KeyValueType,
    create_cli_default_workflow,
    apply_cli_workflow_overrides,
)
from .judge_runner import run_judge
from .cli_default_group import DefaultGroup
import click
from . import AutoJudge


# Leaderboard format constants for CLI options
LEADERBOARD_FORMATS = ["trec_eval", "ir_measures", "tot", "ranking", "jsonl"]

LEADERBOARD_FORMAT_HELP = (
    "  trec_eval: measure topic value (3 cols, run from filename)\n"
    "  ir_measures: run topic measure value (4 cols)\n"
    "  tot: run measure topic value (4 cols)\n"
    "  ranking: topic Q0 doc_id rank score run (6 cols)"
)


def detect_header_interactive(path: Path, format: str, has_header: bool, label: str) -> bool:
    """
    Check if file has header and prompt user if detected but not specified.

    Args:
        path: Path to leaderboard file
        format: Format string (trec_eval, tot, ir_measures, ranking)
        has_header: User-specified header flag
        label: Label for prompt (e.g., "truth" or "eval")

    Returns:
        has_header value to use
    """
    if has_header:
        return True  # Already specified by user

    # jsonl format never has headers (each line is a complete JSON object)
    if format == "jsonl":
        return False

    if not path or not path.is_file():
        return False

    # Read first line and check if value column is numeric
    try:
        first_line = path.read_text().split("\n")[0].strip()
    except Exception:
        return False

    if not first_line:
        return False

    parts = first_line.split()
    if not parts:
        return False

    # Value is always last column for all formats
    value_col = parts[-1]

    try:
        float(value_col)
        return False  # Looks like data
    except ValueError:
        # Looks like header - prompt user
        print(f"First line of {label} leaderboard looks like header:", file=sys.stderr)
        print(f"  '{first_line}'", file=sys.stderr)
        response = input(f"Skip this header line? [Y/n]: ")
        return response.strip().lower() != 'n'


class ExpandedPath(click.Path):
    """Click Path that expands ~ to home directory."""

    def convert(self, value, param, ctx):
        # Expand ~ before validation
        if value is not None:
            value = str(Path(value).expanduser())
        return super().convert(value, param, ctx)


class ClickRagResponses(click.ParamType):
    name = "dir"

    def convert(self, value, param, ctx):
        path = Path(value).expanduser() if value else None
        if not path or not path.is_dir():
            self.fail(f"The directory {value} does not exist, so I can not load rag responses from this directory.", param, ctx)
        runs = load_runs_failsave(path)

        if len(runs) > 0:
            return runs

        self.fail(f"{value!r} contains no rag runs.", param, ctx)


def option_rag_responses():
    """Rag Run directory click option."""
    def decorator(func):
        func = click.option(
            "--rag-responses",
            type=ClickRagResponses(),
            required=True,
            help="The directory that contains the rag responses to evaluate."
        )(func)

        return func

    return decorator


class ClickIrDataset(click.ParamType):
    name = "ir-dataset"

    def fail_if_ir_datasets_is_not_installed(self, param, ctx, msg=""):
        try:
            import ir_datasets
            from ir_datasets import registry
        except:
            msg += " ir_datasets is not installed, so I can not try to load the data via ir_datasets. Please install ir_datasets to load data from there."
            self.fail(msg.strip(), param, ctx)

        try:
            import tira
            from tira.third_party_integrations import ir_datasets
        except:
            msg += " tira is not installed, so I can not try to load the data via tira ir_datasets integration. Please install tira to load data from there."
            self.fail(msg.strip(), param, ctx)

    def convert(self, value, param, ctx):
        self.fail_if_ir_datasets_is_not_installed(param, ctx)

        from ir_datasets import registry
        from tira.third_party_integrations import ir_datasets
        from .io import irds_from_dir, load_hf_dataset_config_or_none

        if value == "infer-dataset-from-context":
            candidate_files = set()
            if "rag_responses" in ctx.params:
                for r in ctx.params["rag_responses"]:
                    if r and r.path:
                        p = Path(r.path).parent
                        candidate_files.add(p / "README.md")
                        candidate_files.add(p.parent / "README.md")
                        candidate_files.add(p.parent.parent / "README.md")
  
            irds_config = None
            base_path = None
            for c in candidate_files:
                irds_config = load_hf_dataset_config_or_none(c, ["ir_dataset"])
                if irds_config:
                    base_path = c.parent
                    irds_config = irds_config["ir_dataset"]
                    break

            if not irds_config:
                raise ValueError("ToDo: Better error handling of wrong configurations")

            if "ir_datasets_id" in irds_config:
                return ir_datasets.load(irds_config["ir_datasets_id"])
            elif "directory" in irds_config:
                return irds_from_dir(str(base_path / irds_config["directory"]))
            else:
                raise ValueError("ToDo: Better error handling of incomplete configurations")

        if value and value in registry:
            return ir_datasets.load(value)

        if value and Path(value).is_dir() and (Path(value) / "queries.jsonl").is_file() and (Path(value) / "corpus.jsonl.gz").is_file():
            return irds_from_dir(value)

        if len(str(value).split("/")) == 2:
            return ir_datasets.load(value)
        else:
            raise ValueError("ToDo: Better error handling of incomplete configurations")


def option_ir_dataset():
    """Ir-dataset click option."""
    def decorator(func):
        func = click.option(
            "--ir-dataset",
            type=ClickIrDataset(),
            required=False,
            default="infer-dataset-from-context",
            help="The ir-datasets ID or a directory that contains the ir-dataset or TODO...."
        )(func)

        return func

    return decorator


class ClickRagTopics(ClickIrDataset):
    name = "file-or-ir-dataset"

    def fail_if_empty_or_return_otherwise(self, value, param, ctx, ret):
        if len(ret) == 0:
            self.fail(f"{value!r} contains 0 RAG topics.", param, ctx)
        else:
            return ret

    def convert(self, value, param, ctx):
        path = Path(value).expanduser() if value else None
        if path and path.is_file():
            try:
                ret = load_requests_from_file(path)
            except Exception as e:
                self.fail(f"The file {value} is not valid, no rag-topics could be loaded. {e}", param, ctx)
            return self.fail_if_empty_or_return_otherwise(value, param, ctx, ret)

        fail_msg = "The argument passed to --rag-topics is not a file."

        self.fail_if_ir_datasets_is_not_installed(param, ctx, msg=fail_msg)

        try:
            ds = super().convert(value, param, ctx)
        except:
            fail_msg += " The argument is also not a valid ir_datasets identifier that could be loaded."
            self.fail(fail_msg, param, ctx)

        ret = load_requests_from_irds(ds)
        return self.fail_if_empty_or_return_otherwise(value, param, ctx, ret)


def option_rag_topics():
    """Provide RAG topics CLI option."""
    def decorator(func):
        func = click.option(
            "--rag-topics",
            type=ClickRagTopics(),
            required=False,
            default="infer-dataset-from-context",
            help="The rag topics. Please either pass a local file that contains Requests in jsonl format (requires fields title and request_id). Alternatively, pass an ir-datasets ID to load the topics from ir_datasets."
        )(func)

        return func

    return decorator


def option_llm_config():
    """Optional llm-config.yml for model preferences."""
    def decorator(func):
        func = click.option(
            "--llm-config",
            type=ExpandedPath(exists=True, path_type=Path),
            required=False,
            default=None,
            help="Path to llm-config.yml (dev: base_url+model, submission: model_preferences). "
                 "Tip: CACHE_FORCE_REFRESH=1 bypasses prompt cache."
        )(func)
        return func
    return decorator


def option_submission():
    """Flag to enable submission mode (resolve model_preferences)."""
    def decorator(func):
        func = click.option(
            "--submission",
            is_flag=True,
            default=False,
            help="Submission mode: resolve model_preferences against organizer's available models"
        )(func)
        return func
    return decorator


class ClickNuggetBanksPath(click.ParamType):
    """Click parameter type for nugget banks path (file or directory)."""
    name = "file-or-dir"

    def convert(self, value, param, ctx):
        if value is None:
            return None

        path = Path(value).expanduser()

        if not path.exists():
            self.fail(f"Path {value} does not exist", param, ctx)

        if not path.is_file() and not path.is_dir():
            self.fail(f"Path {value} is neither a file nor directory", param, ctx)

        return path


def option_nugget_banks():
    """Optional nugget banks path CLI option (file or directory)."""
    def decorator(func):
        func = click.option(
            "--nugget-banks",
            type=ClickNuggetBanksPath(),
            required=False,
            default=None,
            help="Path to nugget banks file (JSON/JSONL) or directory."
        )(func)
        return func
    return decorator


def option_workflow():
    """Optional workflow.yml for declaring judge workflow."""
    def decorator(func):
        func = click.option(
            "--workflow",
            type=ExpandedPath(exists=True, path_type=Path),
            required=False,
            default=None,
            help="Path to workflow.yml declaring the judge's nugget/judge pipeline."
        )(func)
        return func
    return decorator


def _resolve_llm_config(
    llm_config_path: Optional[Path],
    submission: bool = False,
) -> MinimaLlmConfig:
    """
    Resolve LLM config from llm-config.yml or environment.

    Two modes:
    - Dev mode (default): Load direct config (base_url + model) from file or env.
      For judge developers testing with their local LLM.
    - Submission mode (--submission): Resolve model_preferences against available
      models provided by the organizer.

    Note: force_refresh is read from llm-config.yml or CACHE_FORCE_REFRESH env var.

    Args:
        llm_config_path: Path to llm-config.yml
        submission: If True, use submission mode (resolve model_preferences)
    """
    if submission:
        # Submission mode: resolve model_preferences against organizer's available models
        if llm_config_path is None:
            raise click.ClickException(
                "Submission mode requires --llm-config with model_preferences"
            )
        try:
            prefs = ModelPreferences.from_yaml(llm_config_path)
            resolver = ModelResolver.from_env()
            config = resolver.resolve(prefs)
            click.echo(f"Submission mode - resolved model: {config.model} from {config.base_url}", err=True)
            return config
        except ModelResolutionError as e:
            raise click.ClickException(str(e))
        except Exception as e:
            raise click.ClickException(f"Could not resolve model preferences from {llm_config_path}: {e}")

    # Dev mode: load config from YAML (with env as base)
    if llm_config_path is not None:
        try:
            config = MinimaLlmConfig.from_yaml(llm_config_path)
            click.echo(f"Dev mode - loaded config: {config.model} from {config.base_url}", err=True)
            return config
        except FileNotFoundError:
            click.echo(f"Warning: Config file not found: {llm_config_path}", err=True)
        # Note: from_yaml calls from_env() which raises RuntimeError if env vars missing
        # We let that propagate since it's a user error that needs fixing

    # Fallback to environment-based config
    return MinimaLlmConfig.from_env()


def _apply_llm_model_override(
    llm_config: MinimaLlmConfig,
    settings: dict,
    submission: bool,
) -> tuple[MinimaLlmConfig, dict]:
    """
    Apply llm_model from settings to llm_config and strip it from settings.

    Steps:
    1. Extract llm_model from settings
    2. Apply it to llm_config via with_model()
    3. In submission mode, validate the model is allowed by organizer
    4. Strip llm_model from settings before passing to AutoJudge

    Args:
        llm_config: Base LLM configuration
        settings: Settings dict (may contain 'llm_model')
        submission: Whether we're in submission mode

    Returns:
        Tuple of (updated_llm_config, settings_without_llm_model)
    """
    llm_model = settings.get("llm_model")
    stripped_settings = {k: v for k, v in settings.items() if k != "llm_model"}

    if not llm_model:
        return llm_config, stripped_settings

    # In submission mode, validate the model is allowed
    if submission:
        try:
            resolver = ModelResolver.from_env()
            available = resolver.available
            enabled_models = available.get_enabled_models()

            # Check if model is available (directly or via alias)
            canonical = available.resolve_alias(llm_model)
            if canonical not in available.models:
                click.echo(
                    f"Warning: llm_model '{llm_model}' is not available in submission mode. "
                    f"Available models: {enabled_models}. Ignoring llm_model setting.",
                    err=True,
                )
                return llm_config, stripped_settings

        except Exception as e:
            click.echo(
                f"Warning: Could not validate llm_model against available models: {e}. "
                f"Ignoring llm_model setting.",
                err=True,
            )
            return llm_config, stripped_settings

    # Apply the model override
    updated_config = llm_config.with_model(llm_model)
    click.echo(f"Model override from settings: {llm_model}", err=True)
    return updated_config, stripped_settings


def _sanitize_path_for_prefix(path: Optional[Path]) -> str:
    """Sanitize a path for use in batch state file names.

    Replaces slashes, tildes, and other unsafe characters with underscores.
    Returns empty string if path is None.
    """
    if path is None:
        return ""
    # Convert to string and replace unsafe characters
    s = str(path)
    # Replace common path separators and special chars
    for char in "/\\~:":
        s = s.replace(char, "_")
    # Remove leading/trailing underscores and collapse multiple underscores
    import re
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def execute_run_workflow(
    auto_judge=None,
    workflow: Optional[Path] = None,
    rag_responses=None,
    rag_topics=None,
    nugget_banks=None,
    llm_config: Optional[Path] = None,
    submission: bool = False,
    out_dir: Optional[Path] = None,
    filebase: Optional[str] = None,
    store_nuggets: Optional[Path] = None,
    variant: Optional[str] = None,
    sweep: Optional[str] = None,
    all_variants: bool = False,
    force_recreate_nuggets: Optional[bool] = None,
    create_nuggets: Optional[bool] = None,
    do_judge: Optional[bool] = None,
    settings_overrides: tuple = (),
    nugget_settings_overrides: tuple = (),
    judge_settings_overrides: tuple = (),
    qrels_settings_overrides: tuple = (),
    nugget_depends_on_responses: Optional[bool] = None,
    judge_uses_nuggets: Optional[bool] = None,
    create_qrels: Optional[bool] = None,
    judge_uses_qrels: Optional[bool] = None,
    qrels_uses_nuggets: Optional[bool] = None,
    qrels_input: Optional[Path] = None,
    qrels_output: Optional[Path] = None,
    augment_report: Optional[bool] = None,
    limit_topics: Optional[int] = None,
    limit_runs: Optional[int] = None,
    topic_ids: Optional[Tuple[str, ...]] = None,
    wf=None,
    # Modular protocol implementations (alternative to auto_judge)
    nugget_creator=None,  # NuggetCreatorProtocol
    qrels_creator=None,   # QrelsCreatorProtocol
    leaderboard_judge=None,  # LeaderboardJudgeProtocol
):
    """
    Core execution logic for running a judge workflow.

    This function is shared between:
    - auto_judge_to_click_command's run_cmd (auto_judge passed in, wf optional)
    - standalone run command (auto_judge loaded from wf.judge_class)

    Args:
        auto_judge: The AutoJudge instance to run
        wf: Pre-loaded Workflow (optional, loaded from workflow path if None)
        ... all other CLI args ...
    """
    # Load workflow if not provided
    if wf is None:
        if workflow:
            wf = load_workflow(workflow)
            click.echo(f"Loaded workflow: create_nuggets={wf.create_nuggets}, judge={wf.judge}", err=True)
        else:
            # No workflow file - require --nugget-judge flag
            if judge_uses_nuggets is None:
                raise click.UsageError(
                    "No --workflow file provided.\n\n"
                    "When running without workflow.yml, you must specify:\n"
                    "  --nugget-judge     (creates nuggets, then judges with them)\n"
                    "  --no-nugget-judge  (judge only, no nugget creation)\n"
                )
            wf = create_cli_default_workflow(judge_uses_nuggets)
            click.echo(f"Using CLI defaults: create_nuggets={wf.create_nuggets}, judge={wf.judge}", err=True)

    # Apply CLI overrides to workflow
    apply_cli_workflow_overrides(
        wf,
        settings_overrides,
        nugget_settings_overrides,
        judge_settings_overrides,
        nugget_depends_on_responses,
        judge_uses_nuggets,
    )

    # CLI --filebase overrides workflow settings.filebase
    if filebase:
        wf.settings["filebase"] = filebase
        click.echo(f"Filebase override: {filebase}", err=True)

    # If --limit-topics or --topics is set, prefix filebase with "tmp-" for test runs
    if (limit_topics is not None and limit_topics > 0) or (topic_ids is not None and len(topic_ids) > 0):
        current_filebase = wf.settings.get("filebase", "{_name}")
        wf.settings["filebase"] = f"tmp-{current_filebase}"
        click.echo(f"Limited topics mode: filebase changed to {wf.settings['filebase']}", err=True)

    # Validate mutually exclusive options
    options_set = sum([bool(variant), bool(sweep), all_variants])
    if options_set > 1:
        raise click.UsageError("--variant, --sweep, and --all-variants are mutually exclusive.")

    # Resolve configurations based on CLI options
    try:
        if variant:
            configs = [resolve_variant(wf, variant)]
        elif sweep:
            configs = resolve_sweep(wf, sweep)
        elif all_variants:
            if not wf.variants:
                raise click.UsageError("No variants defined in workflow.")
            configs = [resolve_variant(wf, name) for name in wf.variants]
        else:
            configs = [resolve_default(wf)]
    except KeyError as e:
        raise click.UsageError(str(e).strip("'\""))

    # Load base LLM config from file/env
    base_llm_config = _resolve_llm_config(llm_config, submission)

    # Convert rag_topics to list once (it's an iterable)
    topics_list = list(rag_topics)

    # Run each configuration
    for config in configs:
        click.echo(f"\n=== Running configuration: {config.name} ===", err=True)

        # Apply llm_model from settings, validate in submission mode, strip from settings
        effective_llm_config, clean_settings = _apply_llm_model_override(
            base_llm_config, config.settings, submission
        )

        # Determine output paths: --store-nuggets overrides, otherwise use resolved config
        nugget_output_path = store_nuggets or config.nugget_output_path
        judge_output_path = config.judge_output_path

        # Apply --out-dir prefix to output paths
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            if nugget_output_path:
                nugget_output_path = out_dir / nugget_output_path
            if judge_output_path:
                judge_output_path = out_dir / judge_output_path

        # Compute full batch prefix if llm_batch_prefix is set
        if effective_llm_config.parasail.llm_batch_prefix:
            resolved_filebase = clean_settings.get("filebase", "").replace("{_name}", config.name)
            effective_llm_config = _apply_batch_prefix(
                effective_llm_config,
                out_dir=out_dir,
                filebase=resolved_filebase,
                config_name=config.name,
            )

        if nugget_output_path:
            click.echo(f"Nugget output: {nugget_output_path}", err=True)
        if judge_output_path:
            click.echo(f"Judge output: {judge_output_path}", err=True)

        # CLI flags override workflow settings (None means use workflow default)
        effective_create_nuggets = create_nuggets if create_nuggets is not None else wf.create_nuggets
        effective_create_qrels = create_qrels if create_qrels is not None else wf.create_qrels
        effective_do_judge = do_judge if do_judge is not None else wf.judge

        # Determine qrels paths: CLI overrides, otherwise use resolved config
        qrels_input_path = qrels_input or config.qrels_input_path
        qrels_output_path = qrels_output or config.qrels_output_path

        # Apply --out-dir prefix to qrels paths
        if out_dir:
            if qrels_input_path:
                qrels_input_path = out_dir / qrels_input_path
            if qrels_output_path:
                qrels_output_path = out_dir / qrels_output_path

        # Merge qrels settings: config.qrels_settings with CLI overrides
        effective_qrels_settings = dict(config.qrels_settings)
        for key, value in qrels_settings_overrides:
            effective_qrels_settings[key] = value

        run_judge(
            auto_judge=auto_judge,
            rag_responses=rag_responses,
            rag_topics=topics_list,
            llm_config=effective_llm_config,
            nugget_banks_path=nugget_banks,
            judge_output_path=judge_output_path,
            nugget_output_path=nugget_output_path,
            do_create_nuggets=effective_create_nuggets,
            do_create_qrels=effective_create_qrels,
            do_judge=effective_do_judge,
            settings=clean_settings,
            nugget_settings=config.nugget_settings,
            judge_settings=config.judge_settings,
            qrels_settings=effective_qrels_settings,
            # Qrels paths
            qrels_input_path=qrels_input_path,
            qrels_output_path=qrels_output_path,
            # Lifecycle flags
            force_recreate_nuggets=force_recreate_nuggets if force_recreate_nuggets is not None else wf.force_recreate_nuggets,
            force_recreate_qrels=wf.force_recreate_qrels,
            nugget_depends_on_responses=wf.nugget_depends_on_responses,
            judge_uses_nuggets=wf.judge_uses_nuggets,
            judge_uses_qrels=judge_uses_qrels if judge_uses_qrels is not None else wf.judge_uses_qrels,
            qrels_uses_nuggets=qrels_uses_nuggets if qrels_uses_nuggets is not None else wf.qrels_uses_nuggets,
            augment_report=augment_report if augment_report is not None else wf.augment_report,
            config_name=config.name,
            limit_topics=limit_topics,
            limit_runs=limit_runs,
            topic_ids=topic_ids,
            # Modular protocol implementations
            nugget_creator=nugget_creator,
            qrels_creator=qrels_creator,
            leaderboard_judge=leaderboard_judge,
        )

        click.echo(f"Done configuration: {config.name}", err=True)

    click.echo(f"\nAll configurations complete.", err=True)


def options_run(workflow_required: bool = False):
    """
    Decorator that applies all common run command options.

    Args:
        workflow_required: If True, --workflow is required (for standalone CLI).
                          If False, --workflow is optional (for judge-specific CLI).
    """
    def decorator(func):
        # Apply options in reverse order (bottom-up)
        func = click.option("--limit-runs", type=int, default=None,
                          help="Limit to first N run_ids (for testing).")(func)
        func = click.option("--topics", "topic_ids", type=str, multiple=True, default=None,
                          help="Run only on these specific topic IDs (repeatable, for testing).")(func)
        func = click.option("--limit-topics", type=int, default=None,
                          help="Limit to first N topics (for testing).")(func)
        func = click.option("--augment-report/--no-augment-report", "augment_report",
                          default=None, help="Save modified Report.evaldata to {filebase}.responses.jsonl.")(func)
        func = click.option("--qrels-output", type=ExpandedPath(path_type=Path), default=None,
                          help="Path to store created qrels.")(func)
        func = click.option("--qrels-input", type=ClickNuggetBanksPath(), default=None,
                          help="Path to input qrels file (for refinement or judge input).")(func)
        func = click.option("--qrels-uses-nuggets/--no-qrels-uses-nuggets", default=None,
                          help="Override whether create_qrels receives nuggets.")(func)
        func = click.option("--judge-uses-qrels/--no-judge-uses-qrels", default=None,
                          help="Override whether judge receives qrels from create_qrels phase.")(func)
        func = click.option("--create-qrels/--no-create-qrels", default=None,
                          help="Override workflow create_qrels flag.")(func)
        func = click.option("--nugget-judge/--no-nugget-judge", "judge_uses_nuggets",
                          default=None, help="Judge uses nuggets (REQUIRED when --workflow omitted).")(func)
        func = click.option("--nugget-depends-on-responses/--no-nugget-depends-on-responses",
                          default=None, help="Override nugget_depends_on_responses lifecycle flag.")(func)
        func = click.option("--qset", "-Q", "qrels_settings_overrides", multiple=True, type=KeyValueType(),
                          help="Override qrels settings: --qset key=value (e.g., --qset grade_range=[0,3])")(func)
        func = click.option("--jset", "-J", "judge_settings_overrides", multiple=True, type=KeyValueType(),
                          help="Override judge settings: --jset key=value")(func)
        func = click.option("--nset", "-N", "nugget_settings_overrides", multiple=True, type=KeyValueType(),
                          help="Override nugget settings: --nset key=value")(func)
        func = click.option("--set", "-S", "settings_overrides", multiple=True, type=KeyValueType(),
                          help="Override shared settings: --set key=value")(func)
        func = click.option("--judge/--no-judge", "do_judge", default=None, help="Override workflow judge flag.")(func)
        func = click.option("--create-nuggets/--no-create-nuggets", default=None, help="Override workflow create_nuggets flag.")(func)
        func = click.option("--force-recreate-nuggets/--no-force-recreate-nuggets", default=None,
                          help="Recreate nuggets even if file exists. Default: same as create_nuggets.")(func)
        func = click.option("--all-variants", is_flag=True, help="Run all variants defined in workflow.yml.")(func)
        func = click.option("--sweep", type=str, help="Run a parameter sweep from workflow.yml (e.g., --sweep $name).", required=False)(func)
        func = click.option("--variant", type=str, help="Run a named variant from workflow.yml (e.g., --variant $name).", required=False)(func)
        func = click.option("--store-nuggets", type=ExpandedPath(path_type=Path), help="Override nugget output path.", required=False)(func)
        func = click.option("--filebase", type=str, help="Override workflow filebase (e.g., 'my-run').", required=False)(func)
        func = click.option("--out-dir", type=ExpandedPath(path_type=Path), help="Parent directory for all output files.", required=False)(func)
        func = option_submission()(func)
        func = option_llm_config()(func)
        func = option_nugget_banks()(func)
        func = option_rag_topics()(func)
        func = option_rag_responses()(func)
        func = click.option(
            "--workflow", "-w",
            type=ExpandedPath(exists=True, path_type=Path),
            required=workflow_required,
            default=None,
            help="Path to workflow.yml declaring the judge workflow settings."
        )(func)
        return func
    return decorator


def _apply_batch_prefix(
    llm_config: MinimaLlmConfig,
    out_dir: Optional[Path],
    filebase: str,
    config_name: str,
) -> MinimaLlmConfig:
    """
    Compute full batch prefix from llm_batch_prefix + out_dir + filebase + config_name.

    Creates a new config with the computed prefix set in parasail.prefix.

    Args:
        llm_config: LLM configuration with llm_batch_prefix set
        out_dir: Output directory (may be None)
        filebase: Filebase from workflow settings
        config_name: Configuration/variant name

    Returns:
        New config with parasail.prefix set to the computed value
    """
    from dataclasses import replace
    from .llm.llm_config import ParasailBatchConfig

    base_prefix = llm_config.parasail.llm_batch_prefix or ""
    out_dir_part = _sanitize_path_for_prefix(out_dir)

    # Build parts, filtering out empty strings
    parts = [p for p in [base_prefix, out_dir_part, filebase, config_name] if p]
    full_prefix = "_".join(parts)

    # Create new parasail config with computed prefix
    new_parasail = ParasailBatchConfig(
        llm_batch_prefix=llm_config.parasail.llm_batch_prefix,
        prefix=full_prefix,
        state_dir=llm_config.parasail.state_dir,
        poll_interval_s=llm_config.parasail.poll_interval_s,
        max_poll_hours=llm_config.parasail.max_poll_hours,
    )

    click.echo(f"Batch prefix: {full_prefix}", err=True)
    return replace(llm_config, parasail=new_parasail)


def auto_judge_to_click_command(auto_judge: AutoJudge, cmd_name: str):
    """
    Create a Click command group for an AutoJudge with subcommands:
    - nuggify: Create/refine nugget banks only
    - judge: Judge with existing nugget banks
    - run: Execute according to workflow.yml (DEFAULT)

    Invoking without a subcommand runs the 'run' command.
    """
    from .request import Request
    from .report import Report
    from typing import Iterable

    @click.group(cmd_name, cls=DefaultGroup, default_cmd_name="run")
    @click.pass_context
    def cli(ctx):
        """AutoJudge command group."""
        ctx.ensure_object(dict)
        ctx.obj["auto_judge"] = auto_judge

    @cli.command("judge")
    @option_rag_responses()
    @option_rag_topics()
    @option_nugget_banks()
    @option_llm_config()
    @option_submission()
    @click.option("--output", type=ExpandedPath(path_type=Path), help="Leaderboard output file.", required=True)
    def judge_cmd(
        rag_topics: Iterable[Request],
        rag_responses: Iterable[Report],
        nugget_banks,
        llm_config: Optional[Path],
        submission: bool,
        output: Path,
    ):
        """Judge RAG responses using existing nugget banks."""
        resolved_config = _resolve_llm_config(llm_config, submission)

        run_judge(
            auto_judge=auto_judge,
            rag_responses=rag_responses,
            rag_topics=list(rag_topics),
            llm_config=resolved_config,
            nugget_banks_path=nugget_banks,
            judge_output_path=output,
            do_create_nuggets=False,
            do_judge=True,
        )

    @cli.command("nuggify")
    @option_rag_responses()
    @option_rag_topics()
    @option_nugget_banks()
    @option_llm_config()
    @option_submission()
    @click.option("--store-nuggets", type=ExpandedPath(path_type=Path), help="Output nuggets file.", required=True)
    def nuggify_cmd(
        rag_responses: Iterable[Report],
        rag_topics: Iterable[Request],
        nugget_banks,
        llm_config: Optional[Path],
        submission: bool,
        store_nuggets: Path
    ):
        """Create or refine nugget banks based on RAG responses."""
        resolved_config = _resolve_llm_config(llm_config, submission)

        result = run_judge(
            auto_judge=auto_judge,
            rag_responses=rag_responses,
            rag_topics=list(rag_topics),
            llm_config=resolved_config,
            nugget_banks_path=nugget_banks,
            judge_output_path=None,
            nugget_output_path=store_nuggets,
            do_create_nuggets=True,
            do_judge=False,
        )

    @cli.command("run")
    @options_run(workflow_required=False)
    def run_cmd(**kwargs):
        """Run judge according to workflow.yml (default command)."""
        execute_run_workflow(auto_judge=auto_judge, **kwargs)

    @cli.command("batch-status")
    @option_llm_config()
    @click.option("--cancel", type=str, metavar="PREFIX", help="Cancel batch for PREFIX and delete local state.")
    @click.option("--cancel-remote", type=str, metavar="BATCH_ID", help="Cancel a remote batch by ID.")
    @click.option("--cancel-all", is_flag=True, help="Cancel ALL local batches.")
    def batch_status_cmd(
        llm_config: Optional[Path],
        cancel: Optional[str],
        cancel_remote: Optional[str],
        cancel_all: bool,
    ):
        """Show status of all Parasail batches.

        By default, lists all local batch state files and checks remote status
        for any active batches.
        """
        try:
            from .llm.batch import (
                batch_status_overview,
                cancel_batch,
                cancel_all_batches,
                cancel_all_local_batches,
            )
        except ImportError:
            raise click.ClickException(
                "Batch management requires minima-llm package. "
                "Install with: pip install minima-llm"
            )

        resolved_config = _resolve_llm_config(llm_config, submission=False)

        if cancel:
            cancel_all_batches(resolved_config, prefix=cancel)
        elif cancel_remote:
            cancel_batch(cancel_remote, resolved_config)
        elif cancel_all:
            cancel_all_local_batches(resolved_config)
        else:
            # Default: show overview of all batches
            batch_status_overview(resolved_config)

    return cli
