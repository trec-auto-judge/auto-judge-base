"""
CLI entry point for autojudge-base.

Provides the `auto-judge` command with subcommands:
- run: Execute a judge workflow
"""

from click import group

from ._commands._run import run_workflow


@group()
def main():
    """TREC AutoJudge - Infrastructure for automated RAG evaluation."""
    pass


main.add_command(run_workflow)


if __name__ == '__main__':
    main()
