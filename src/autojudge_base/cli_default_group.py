"""CLI utilities for Click-based commands."""

from typing import Optional

import click


class DefaultGroup(click.Group):
    """Click group that invokes a default subcommand when none is specified."""

    def __init__(self, *args, default_cmd_name: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_cmd_name = default_cmd_name

    def parse_args(self, ctx, args):
        # Show group help if --help is passed without subcommand
        if '--help' in args or '-h' in args:
            return super().parse_args(ctx, args)
        # If no args or first arg looks like an option, prepend default command
        if self.default_cmd_name and (not args or args[0].startswith('-')):
            args = [self.default_cmd_name] + list(args)
        return super().parse_args(ctx, args)

    def get_command(self, ctx, cmd_name):
        """Get command with helpful error message for invalid commands."""
        cmd = super().get_command(ctx, cmd_name)
        if cmd is None:
            available = ", ".join(sorted(self.commands.keys()))
            raise click.UsageError(
                f"Unknown command '{cmd_name}'.\n\n"
                f"Available commands: {available}\n\n"
                f"Common usage patterns:\n"
                f"  {ctx.info_name} run --rag-responses <dir> --rag-topics <file> --output <file>\n"
                f"  {ctx.info_name} judge --rag-responses <dir> --nugget-banks <file> --output <file>\n"
                f"  {ctx.info_name} nuggify --rag-responses <dir> --rag-topics <file> --store-nuggets <file>\n\n"
                f"Run '{ctx.info_name} --help' for more information."
            )
        return cmd

    def format_help(self, ctx, formatter):
        """Format help to include default command options."""
        super().format_help(ctx, formatter)

        # Append default command's options
        if self.default_cmd_name and self.default_cmd_name in self.commands:
            default_cmd = self.commands[self.default_cmd_name]
            formatter.write_paragraph()
            formatter.write_text(
                f"Default command: {self.default_cmd_name}"
            )
            formatter.write_paragraph()

            # Get default command's options
            with formatter.section("Default command options"):
                default_cmd.format_options(ctx, formatter)