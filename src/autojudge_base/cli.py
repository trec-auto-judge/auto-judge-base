"""
CLI entry point for autojudge-base.

Provides the `auto-judge` command with subcommands:
- run: Execute a judge workflow
- export-corpus: Export corpus to archive
- list-models: List available models
"""

from click import group

from ._commands._run import run_workflow
from ._commands._export_corpus import export_corpus
from ._commands._list_models import list_models


@group()
def main():
    """TREC AutoJudge - Infrastructure for automated RAG evaluation."""
    pass


main.add_command(run_workflow)
main.command()(export_corpus)
main.add_command(list_models)


if __name__ == '__main__':
    main()
