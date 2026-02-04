"""CLI command to run an AutoJudge via workflow.yml."""
import click

from ..click_plus import options_run, execute_run_workflow
from ..workflow import load_workflow, load_judge_from_workflow


@click.command("run")
@options_run(workflow_required=True)
def run_workflow(**kwargs):
    """Run an AutoJudge according to workflow.yml.

    The workflow.yml can specify judge classes as dotted import paths:

        # Single class implementing all protocols (common case):
        judge_class: "trec25.judges.minimaljudge.minimal_judge.MinimalJudge"

        # Or separate classes for modular composition:
        nugget_class: "trec25.judges.shared.NuggetGenerator"
        qrels_class: "trec25.judges.umbrela.UmbrelaQrels"
        judge_class: "trec25.judges.myjudge.MyLeaderboard"

    Examples:

        trec-auto-judge run --workflow ./judges/myjudge/workflow.yml \\
            --rag-responses ./responses/ \\
            --rag-topics ./topics.jsonl \\
            --out-dir ./output/ \\
            --llm-config ./llm-config.yml

        trec-auto-judge run -w workflow.yml --variant strict ...

        trec-auto-judge run -w workflow.yml --sweep grid-search ...
    """
    workflow_path = kwargs["workflow"]

    # Load workflow
    wf = load_workflow(workflow_path)
    click.echo(f"Loaded workflow: create_nuggets={wf.create_nuggets}, judge={wf.judge}", err=True)

    # Load judge components from workflow (supports modular composition)
    if not wf.judge_class and not wf.nugget_class and not wf.qrels_class:
        raise click.UsageError(
            f"workflow.yml does not specify any judge classes.\n"
            f"Add 'judge_class: \"module.path.ClassName\"' to {workflow_path}"
        )

    try:
        components = load_judge_from_workflow(wf)
        # Log which classes were loaded
        loaded = []
        if wf.nugget_class:
            loaded.append(f"nugget_class={wf.nugget_class}")
        if wf.qrels_class:
            loaded.append(f"qrels_class={wf.qrels_class}")
        if wf.judge_class:
            loaded.append(f"judge_class={wf.judge_class}")
        click.echo(f"Loaded: {', '.join(loaded)}", err=True)
    except Exception as e:
        raise click.UsageError(f"Failed to load judge classes: {e}")

    # Execute with pre-loaded workflow and modular components
    execute_run_workflow(
        wf=wf,
        nugget_creator=components.nugget_creator,
        qrels_creator=components.qrels_creator,
        leaderboard_judge=components.leaderboard_judge,
        **kwargs,
    )