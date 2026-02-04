"""
JudgeRunner: Orchestrates AutoJudge execution with nugget lifecycle management.

Consolidates repeated functionality for nugget creation, saving, and judging.
"""

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import yaml

from .utils import get_git_info
from .nugget_data import (
    NuggetBanksProtocol,
    write_nugget_banks_generic,
)
from .qrels.qrels import Qrels, write_qrel_file
from .leaderboard.leaderboard import Leaderboard
from .llm import MinimaLlmConfig
from .report import Report
from .request import Request
from .workflow.paths import (
    resolve_nugget_file_path,
    resolve_qrels_file_path,
    resolve_leaderboard_file_path,
    resolve_config_file_path,
    resolve_responses_file_path,
    load_nugget_banks_from_path,
    load_qrels_from_path,
)


@dataclass
class JudgeResult:
    """Result of a judge run."""
    leaderboard: Optional[Leaderboard]
    qrels: Optional[Qrels]  # Final qrels (from judge() if available, else from create_qrels())
    nuggets: Optional[NuggetBanksProtocol]  # Final nuggets (created or loaded)


def run_judge(
    auto_judge=None,
    rag_responses: Iterable[Report] = None,
    rag_topics: Sequence[Request] = None,
    llm_config: MinimaLlmConfig = None,
    nugget_banks_path: Optional[Path] = None,
    judge_output_path: Optional[Path] = None,
    nugget_output_path: Optional[Path] = None,
    do_create_nuggets: bool = False,
    do_create_qrels: bool = False,
    do_judge: bool = True,
    # Settings dicts passed to AutoJudge methods as **kwargs
    settings: Optional[dict[str, Any]] = None,
    nugget_settings: Optional[dict[str, Any]] = None,
    judge_settings: Optional[dict[str, Any]] = None,
    qrels_settings: Optional[dict[str, Any]] = None,
    # Qrels paths
    qrels_input_path: Optional[Path] = None,
    qrels_output_path: Optional[Path] = None,
    # Lifecycle flags
    force_recreate_nuggets: Optional[bool] = None,
    force_recreate_qrels: Optional[bool] = None,
    nugget_depends_on_responses: bool = True,
    judge_uses_nuggets: bool = True,
    judge_uses_qrels: bool = True,
    qrels_uses_nuggets: bool = True,
    augment_report: bool = False,
    # Configuration name for reproducibility tracking
    config_name: str = "default",
    # Testing flags to limit scope
    limit_topics: Optional[int] = None,
    limit_runs: Optional[int] = None,
    topic_ids: Optional[Sequence[str]] = None,
    # Modular protocol implementations (alternative to auto_judge)
    nugget_creator=None,  # NuggetCreatorProtocol
    qrels_creator=None,   # QrelsCreatorProtocol
    leaderboard_judge=None,  # LeaderboardJudgeProtocol
) -> JudgeResult:
    """
    Execute judge workflow with nugget lifecycle management.

    Args:
        auto_judge: AutoJudge implementation
        rag_responses: RAG responses to evaluate
        rag_topics: Topics/queries to evaluate
        llm_config: LLM configuration
        nugget_banks_path: Path to input nugget banks (file or directory)
        judge_output_path: Leaderboard/qrels output path
        nugget_output_path: Path to store created/refined nuggets
        do_create_nuggets: If True, call create_nuggets()
        do_judge: If True, call judge()
        settings: Shared settings dict passed to both phases (fallback)
        nugget_settings: Settings dict passed to create_nuggets() (overrides settings)
        judge_settings: Settings dict passed to judge() (overrides settings)
        force_recreate_nuggets: If True, recreate even if file exists. If None (default),
            defaults to do_create_nuggets value (so fresh creation uses prompt cache, not stale files)
        nugget_depends_on_responses: If True, pass responses to create_nuggets()
        judge_uses_nuggets: If True, pass nuggets to judge()
        augment_report: If True, save modified Report.evaldata to {filebase}.responses.jsonl
        config_name: Variant/sweep name for reproducibility tracking (default: "default")
        limit_topics: If set, limit to first N topics (for testing)
        limit_runs: If set, limit to first N run_ids (for testing)
        topic_ids: If set, only run on these specific topic IDs (for testing)

    Returns:
        JudgeResult with leaderboard, qrels, and final nuggets
    """
    # Apply topic limit if specified (for testing)
    if limit_topics is not None and limit_topics > 0:
        rag_topics = list(rag_topics)[:limit_topics]
        limited_topic_ids = {t.request_id for t in rag_topics}
        rag_responses = [r for r in rag_responses if r.metadata.topic_id in limited_topic_ids]
        print(f"[judge_runner] Limited to first {limit_topics} topics: {sorted(limited_topic_ids)}", file=sys.stderr)

    # Apply explicit topic filter if specified (for testing)
    if topic_ids is not None and len(topic_ids) > 0:
        topic_ids_set = set(topic_ids)
        rag_topics = [t for t in rag_topics if t.request_id in topic_ids_set]
        rag_responses = [r for r in rag_responses if r.metadata.topic_id in topic_ids_set]
        print(f"[judge_runner] Filtered to explicit topics: {sorted(topic_ids_set)}", file=sys.stderr)

    # Apply run limit if specified (for testing)
    if limit_runs is not None and limit_runs > 0:
        # Ensure rag_responses is a list for multiple iterations
        rag_responses = list(rag_responses)
        # Get unique run_ids in order of first appearance
        seen_run_ids = []
        for r in rag_responses:
            if r.metadata.run_id not in seen_run_ids:
                seen_run_ids.append(r.metadata.run_id)
        limited_run_ids = set(seen_run_ids[:limit_runs])
        rag_responses = [r for r in rag_responses if r.metadata.run_id in limited_run_ids]
        print(f"[judge_runner] Limited to first {limit_runs} runs: {sorted(limited_run_ids)}", file=sys.stderr)

    # Default force_recreate_nuggets to do_create_nuggets if not explicitly set
    # This ensures fresh nugget creation uses prompt cache, not stale nugget files
    if force_recreate_nuggets is None:
        force_recreate_nuggets = do_create_nuggets

    # Get nugget_banks_type from nugget_creator or auto_judge (required for loading/saving nuggets)
    nugget_banks_type = getattr(nugget_creator, "nugget_banks_type", None) or getattr(auto_judge, "nugget_banks_type", None)

    # Resolve nugget output file path (add .nuggets.jsonl extension if needed)
    nugget_file_path = resolve_nugget_file_path(nugget_output_path) if nugget_output_path else None

    # Validate nugget_banks_type is available when needed for loading or saving
    needs_nugget_type = (
        (nugget_banks_path and nugget_banks_path.exists()) or  # Loading from input path
        (nugget_file_path and nugget_file_path.exists() and not force_recreate_nuggets) or  # Loading from output
        (do_create_nuggets and nugget_output_path)  # Will save nuggets (need type for future loading)
    )
    if needs_nugget_type and not nugget_banks_type:
        raise ValueError(
            "Cannot load/save nuggets: auto_judge does not define nugget_banks_type. "
            "Add nugget_banks_type class attribute to your AutoJudge implementation."
        )

    # Load input nuggets from path if provided
    input_nuggets: Optional[NuggetBanksProtocol] = None
    if nugget_banks_path and nugget_banks_path.exists() and nugget_banks_type:
        print(f"[judge_runner] Loading input nuggets: {nugget_banks_path}", file=sys.stderr)
        input_nuggets = load_nugget_banks_from_path(nugget_banks_path, nugget_banks_type)

    current_nuggets = input_nuggets
    current_qrels: Optional[Qrels] = None
    leaderboard = None

    # Default force_recreate_qrels to do_create_qrels if not explicitly set
    if force_recreate_qrels is None:
        force_recreate_qrels = do_create_qrels

    # Resolve qrels output file path (add .qrels extension if needed)
    qrels_file_path = resolve_qrels_file_path(qrels_output_path) if qrels_output_path else None

    # Load input qrels if provided
    if qrels_input_path and qrels_input_path.exists():
        print(f"[judge_runner] Loading input qrels: {qrels_input_path}", file=sys.stderr)
        current_qrels = load_qrels_from_path(qrels_input_path)

    _write_run_config(
        judge_output_path=judge_output_path,
        config_name=config_name,
        do_create_nuggets=do_create_nuggets,
        do_create_qrels=do_create_qrels,
        do_judge=do_judge,
        llm_model=llm_config.model,
        settings=settings,
        nugget_settings=nugget_settings,
        judge_settings=judge_settings,
        qrels_settings=qrels_settings,
    )


    # Step 1: Create or load nuggets
    if do_create_nuggets:
        # Check if output file exists and we can skip creation
        if nugget_file_path and nugget_file_path.exists() and not force_recreate_nuggets:
            print(f"[judge_runner] Loading existing nuggets (skipping creation): {nugget_file_path}", file=sys.stderr)
            current_nuggets = load_nugget_banks_from_path(nugget_file_path, nugget_banks_type)
        else:
            # Create nuggets
            nugget_kwargs = nugget_settings or settings or {}
            if nugget_kwargs:
                print(f"[judge_runner] create_nuggets settings: {nugget_kwargs}", file=sys.stderr)

            # Pass responses based on nugget_depends_on_responses flag
            responses_for_nuggets = rag_responses if nugget_depends_on_responses else None

            # Use nugget_creator if provided, otherwise fall back to auto_judge
            _nugget_creator = nugget_creator or auto_judge
            current_nuggets = _nugget_creator.create_nuggets(
                rag_responses=responses_for_nuggets,
                rag_topics=rag_topics,
                llm_config=llm_config,
                nugget_banks=input_nuggets,
                **nugget_kwargs,
            )
            # Verify created nuggets
            if current_nuggets is not None:
                # Verify type matches what auto_judge declared
                if nugget_banks_type and not isinstance(current_nuggets, nugget_banks_type):
                    print(
                        f"create_nuggets() returned {type(current_nuggets).__name__}, "
                        f"but auto_judge declares nugget_banks_type={nugget_banks_type.__name__}. "
                        f"Ensure create_nuggets() returns the declared type to avoid problems in nugget loading."
                        , sys.stderr
                    )
                topic_ids = [t.request_id for t in rag_topics]
                current_nuggets.verify(topic_ids)
                # Save immediately for crash recovery
                if nugget_file_path:
                    write_nugget_banks_generic(current_nuggets, nugget_file_path)
                    print(f"[judge_runner] Nuggets saved to: {nugget_file_path}", file=sys.stderr)

    # Step 2: Create or load qrels
    if do_create_qrels:
        # Check if output file exists and we can skip creation
        if qrels_file_path and qrels_file_path.exists() and not force_recreate_qrels:
            print(f"[judge_runner] Loading existing qrels (skipping creation): {qrels_file_path}", file=sys.stderr)
            current_qrels = load_qrels_from_path(qrels_file_path)
        else:
            # Create qrels
            qrels_kwargs = qrels_settings or settings or {}
            if qrels_kwargs:
                print(f"[judge_runner] create_qrels settings: {qrels_kwargs}", file=sys.stderr)

            # Pass nuggets based on qrels_uses_nuggets flag
            nuggets_for_qrels = current_nuggets if qrels_uses_nuggets else None

            # Use qrels_creator if provided, otherwise fall back to auto_judge
            _qrels_creator = qrels_creator or auto_judge
            current_qrels = _qrels_creator.create_qrels(
                rag_responses=rag_responses,
                rag_topics=rag_topics,
                llm_config=llm_config,
                nugget_banks=nuggets_for_qrels,
                **qrels_kwargs,
            )

            # Verify and save created qrels
            if current_qrels is not None:
                topic_ids = [t.request_id for t in rag_topics]
                current_qrels.verify(expected_topic_ids=topic_ids)
                # Save immediately for crash recovery
                if qrels_file_path:
                    write_qrel_file(qrel_out_file=qrels_file_path, qrels=current_qrels)
                    print(f"[judge_runner] Qrels saved to: {qrels_file_path}", file=sys.stderr)

    # Step 3: Judge leaderboard if requested
    if do_judge:
        judge_kwargs = dict(judge_settings or settings or {})

        # Inject resolved filebase from judge_output_path (replaces template like {_name})
        if judge_output_path:
            judge_kwargs["filebase"] = str(judge_output_path)

        if judge_kwargs:
            print(f"[judge_runner] judge settings: {judge_kwargs}", file=sys.stderr)

        # Pass nuggets based on judge_uses_nuggets flag
        nuggets_for_judge = current_nuggets if judge_uses_nuggets else None
        # Pass qrels based on judge_uses_qrels flag
        qrels_for_judge = current_qrels if judge_uses_qrels else None

        # Use leaderboard_judge if provided, otherwise fall back to auto_judge
        _leaderboard_judge = leaderboard_judge or auto_judge
        leaderboard = _leaderboard_judge.judge(
            rag_responses=rag_responses,
            rag_topics=rag_topics,
            llm_config=llm_config,
            nugget_banks=nuggets_for_judge,
            qrels=qrels_for_judge,
            **judge_kwargs,
        )

        # Write leaderboard output
        if judge_output_path:
            _write_leaderboard(
                leaderboard=leaderboard,
                rag_topics=rag_topics,
                judge_output_path=judge_output_path,
            )

            # Step 4: Save augmented responses if flag is set
            if augment_report:
                _write_augmented_responses(
                    rag_responses=rag_responses,
                    judge_output_path=judge_output_path,
                )

    return JudgeResult(
        leaderboard=leaderboard,
        qrels=current_qrels,
        nuggets=current_nuggets,
    )


def _write_leaderboard(
    leaderboard: Leaderboard,
    rag_topics: Sequence[Request],
    judge_output_path: Path,
) -> None:
    """Verify and write leaderboard."""
    topic_ids = [t.request_id for t in rag_topics]

    # Resolve output path from filebase
    leaderboard_path = resolve_leaderboard_file_path(judge_output_path)
    leaderboard.verify(expected_topic_ids=topic_ids, on_missing="fix_aggregate")
    leaderboard.write(leaderboard_path)
    print(f"[judge_runner] Leaderboard saved to: {leaderboard_path}", file=sys.stderr)


def _write_run_config(
    judge_output_path: Path,
    config_name: str,
    do_create_nuggets: bool,
    do_create_qrels: bool,
    do_judge: bool,
    llm_model: str,
    settings: Optional[dict[str, Any]],
    nugget_settings: Optional[dict[str, Any]],
    judge_settings: Optional[dict[str, Any]],
    qrels_settings: Optional[dict[str, Any]],
) -> None:
    """
    Write run configuration for reproducibility.

    Creates {filebase}.config.yml with:
    - Workflow parameters (name, flags, settings)
    - LLM model used
    - Git info (commit, dirty, remote)
    - Timestamp
    """
    config_path = resolve_config_file_path(judge_output_path)

    config: dict[str, Any] = {
        "name": config_name,
        "create_nuggets": do_create_nuggets,
        "create_qrels": do_create_qrels,
        "judge": do_judge,
        "llm_model": llm_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": get_git_info(),
    }

    # Only include non-empty settings
    if settings:
        config["settings"] = settings
    if nugget_settings:
        config["nugget_settings"] = nugget_settings
    if judge_settings:
        config["judge_settings"] = judge_settings
    if qrels_settings:
        config["qrels_settings"] = qrels_settings

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"[judge_runner] Config saved to: {config_path}", file=sys.stderr)


def _write_augmented_responses(
    rag_responses: Iterable[Report],
    judge_output_path: Path,
) -> None:
    """Write augmented responses (with evaldata) to {filebase}.responses.jsonl."""
    from .report import JsonlWriter

    responses_path = resolve_responses_file_path(judge_output_path)
    with JsonlWriter(responses_path) as writer:
        writer.write_many(rag_responses)
    print(f"[judge_runner] Augmented responses saved to: {responses_path}", file=sys.stderr)