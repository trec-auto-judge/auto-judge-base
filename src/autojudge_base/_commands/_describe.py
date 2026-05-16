from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from urllib.request import urlopen

import click
import yaml

from ..workflow.paths import resolve_config_file_path, resolve_leaderboard_file_path

JUDGES_YML_URL = (
    "https://raw.githubusercontent.com/trec-auto-judge/"
    "auto-judge-starter-kit/refs/heads/main/judges.yml"
)
JUDGES_YML_PR_URL = (
    "https://github.com/trec-auto-judge/auto-judge-starter-kit/blob/main/judges.yml"
)
MISSING_MEASURE_DESCRIPTION = (
    "No description available. Please add it via a pull request to the file "
    f"{JUDGES_YML_PR_URL}"
)


@dataclass(frozen=True)
class JudgeMeasureDescription:
    name: str
    description: str


@dataclass(frozen=True)
class JudgeDescription:
    name: str
    description: str
    git_remotes: tuple[str, ...]
    measures: tuple[JudgeMeasureDescription, ...]


def resolve_describe_paths(path: Path) -> tuple[Path, Path]:
    if path.name.endswith(".eval.txt"):
        filebase = path.parent / path.name[: -len(".eval.txt")]
        leaderboard_path = path
    elif path.name.endswith(".eval"):
        filebase = path.parent / path.name[: -len(".eval")]
        leaderboard_path = path
    elif path.suffix == ".tsv":
        filebase = path.with_suffix("")
        leaderboard_path = path
    else:
        filebase = path
        leaderboard_path = resolve_leaderboard_file_path(path)

    return leaderboard_path, resolve_config_file_path(filebase)


def load_config_remote(config_path: Path) -> str:
    if not config_path.is_file():
        raise FileNotFoundError(f"Expected config next to leaderboard: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    remote = config.get("git", {}).get("remote") if isinstance(config, dict) else None
    if not remote:
        raise ValueError(f"Config {config_path} does not contain git.remote")

    return str(remote)


def fetch_judge_catalog(url: str = JUDGES_YML_URL) -> tuple[JudgeDescription, ...]:
    try:
        with urlopen(url, timeout=10) as response:
            payload = response.read().decode("utf-8")
    except OSError as exc:
        raise OSError(f"Failed to load judge descriptions from {url}: {exc}") from exc

    data = yaml.safe_load(payload) or {}
    judges = data.get("judges") if isinstance(data, dict) else None
    if not isinstance(judges, dict):
        raise ValueError(f"Invalid judge descriptions from {url}: missing 'judges' mapping")

    catalog = []
    for name, judge_data in judges.items():
        if not isinstance(judge_data, dict):
            raise ValueError(f"Invalid judge entry for {name!r} in {url}")

        measures_data = judge_data.get("measures") or []
        if not isinstance(measures_data, list):
            raise ValueError(f"Invalid measures list for {name!r} in {url}")

        catalog.append(
            JudgeDescription(
                name=str(name),
                description=str(judge_data.get("description", "")).strip(),
                git_remotes=tuple(str(remote) for remote in judge_data.get("git_remotes") or ()),
                measures=tuple(
                    JudgeMeasureDescription(
                        name=str(measure["name"]),
                        description=str(measure.get("description", "")).strip(),
                    )
                    for measure in measures_data
                    if isinstance(measure, dict) and "name" in measure
                ),
            )
        )

    return tuple(catalog)


def find_judge_description_by_remote(
    remote: str,
    catalog: tuple[JudgeDescription, ...],
) -> JudgeDescription:
    for judge in catalog:
        if remote in judge.git_remotes:
            return judge

    raise ValueError(f"No judge description found for git remote: {remote}")


def filter_judge_measures_for_leaderboard(
    leaderboard_path: Path,
    judge: JudgeDescription,
) -> JudgeDescription:
    measures_by_name = {measure.name: measure for measure in judge.measures}
    reported_measure_names: list[str] = []
    seen_measure_names: set[str] = set()

    def add_measure_name(measure_name: str) -> None:
        if measure_name and measure_name not in seen_measure_names:
            seen_measure_names.add(measure_name)
            reported_measure_names.append(measure_name)

    with leaderboard_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    obj = None
                if isinstance(obj, dict):
                    measure_name = obj.get("measure")
                    if isinstance(measure_name, str):
                        add_measure_name(measure_name)
                continue

            parts = line.split()
            if len(parts) >= 3:
                add_measure_name(parts[2])

    return JudgeDescription(
        name=judge.name,
        description=judge.description,
        git_remotes=judge.git_remotes,
        measures=tuple(
            measures_by_name.get(
                measure_name,
                JudgeMeasureDescription(
                    name=measure_name,
                    description=MISSING_MEASURE_DESCRIPTION,
                ),
            )
            for measure_name in reported_measure_names
        ),
    )


def format_judge_description(
    judge: JudgeDescription,
) -> str:
    lines = [
        f"Judge: {judge.name}",
    ]

    if judge.description:
        lines.extend(["", judge.description])

    if judge.measures:
        lines.extend(["", "Measures:"])
        lines.extend(
            f"- {measure.name}: {measure.description or MISSING_MEASURE_DESCRIPTION}"
            for measure in judge.measures
        )

    return "\n".join(lines)


def describe_leaderboard(
    leaderboard_path: Path,
    catalog: tuple[JudgeDescription, ...] | None = None,
) -> str:
    """Describe a leaderboard using the sibling config and remote judge metadata."""
    leaderboard_path, config_path = resolve_describe_paths(leaderboard_path)

    if not leaderboard_path.is_file():
        raise FileNotFoundError(f"Leaderboard not found: {leaderboard_path}")

    remote = load_config_remote(config_path)
    judge = find_judge_description_by_remote(remote, catalog or fetch_judge_catalog())
    judge = filter_judge_measures_for_leaderboard(leaderboard_path, judge)

    return judge


@click.command("describe")
@click.argument("leaderboard_path", type=click.Path(path_type=Path))
def describe_leaderboard_click(leaderboard_path: Path) -> None:
    """Describe a leaderboard using the sibling config and remote judge metadata."""
    try:
        judge = describe_leaderboard(leaderboard_path)
        click.echo(format_judge_description(judge))
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
