"""CLI command to list and validate available LLM models."""
import click
from pathlib import Path
from typing import Optional

from ..llm_resolver import AvailableModels, ModelPreferences, ModelResolver, ModelResolutionError


@click.command("list-models")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to available_models.yml (default: from env or ~/.autojudge/)"
)
@click.option(
    "--resolve", "-r",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Test resolution against a participant's llm-config.yml"
)
def list_models(config: Optional[Path], resolve: Optional[Path]):
    """List available LLM models and optionally test resolution.

    Examples:

        trec-auto-judge list-models

        trec-auto-judge list-models --config /path/to/available_models.yml

        trec-auto-judge list-models --resolve participant/llm-config.yml
    """
    # Load available models
    if config:
        available = AvailableModels.from_yaml(config)
        click.echo(f"Loaded config from: {config}")
    else:
        available = AvailableModels.from_env()
        click.echo("Loaded config from environment/defaults")

    click.echo()

    # Display enabled models
    enabled = available.get_enabled_models()
    if enabled:
        click.echo(click.style("Enabled models:", bold=True))
        for name in sorted(enabled):
            cfg = available.models[name]
            click.echo(f"  {name}")
            click.echo(f"    model_id: {cfg.model}")
            click.echo(f"    base_url: {cfg.base_url}")
    else:
        click.echo(click.style("No enabled models found!", fg="yellow"))

    # Display disabled models
    if available.disabled:
        click.echo()
        click.echo(click.style("Disabled models:", fg="bright_black"))
        for name in sorted(available.disabled):
            click.echo(f"  {name}")

    # Display default
    click.echo()
    click.echo(f"Default model: {available.default_model or '(none)'}")

    # Display aliases
    if available.aliases:
        click.echo()
        click.echo(click.style("Aliases:", bold=True))
        for alias, target in sorted(available.aliases.items()):
            click.echo(f"  {alias} -> {target}")

    # Test resolution if requested
    if resolve:
        click.echo()
        click.echo(click.style(f"Testing resolution against: {resolve}", bold=True))
        try:
            prefs = ModelPreferences.from_yaml(resolve)
            click.echo(f"  Preferences: {list(prefs.preferences)}")
            click.echo(f"  on_no_match: {prefs.on_no_match}")

            resolver = ModelResolver(available=available)
            resolved = resolver.resolve(prefs)

            click.echo()
            click.echo(click.style("Resolved:", fg="green", bold=True))
            click.echo(f"  model: {resolved.model}")
            click.echo(f"  base_url: {resolved.base_url}")
        except ModelResolutionError as e:
            click.echo()
            click.echo(click.style(f"Resolution failed: {e}", fg="red"))
            raise SystemExit(1)