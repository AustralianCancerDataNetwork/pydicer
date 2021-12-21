import argparse
import sys

from pydicer.cli.input import testinput_cli, filesystem_cli, pacs_cli, tcia_cli, web_cli
from pydicer.pipeline import run_test


def parse_sub_input():
    """function to parse the input command args"""
    parse_sub_command(
        "Run the Input module only",
        INPUT_TOOLS,
        "test",
    )


# Commands that can be run
MODULES = {"pipeline": run_test, "input": parse_sub_input}

# Sub command types for the Input command
INPUT_TOOLS = {
    "test": testinput_cli,
    "filesystem": filesystem_cli,
    "pacs": pacs_cli,
    "tcia": tcia_cli,
    "web": web_cli,
}

COMMANDS = str(list(MODULES.keys())).replace(", ", "|")


def parse_sub_command(desc, tools, default_choice):
    """Generic function to take in dynamic input and trigger the respective sub commands

    Args:
        desc (str): help description of what the sub command does
        tools (dict): dictionary of which sub command type can be run
        default_choice (str): default sub command type that will be run in the case no input is
        received from the user
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--type",
        help=f"Subcommand of the following: {COMMANDS}",
        default=default_choice,
        choices=tools,
    )

    args = parser.parse_args(sys.argv[2:4])
    tools[args.type](*sys.argv[4:])


def pydicer_cli():
    """
    Trigger pydicer CLI
    """

    parser = argparse.ArgumentParser(
        description="pydicer CLI (Command Line Interface)",
        usage=f"python -m pydicer.cli.run {COMMANDS}",
    )

    # Default to "pipeline" option without input
    parser.add_argument(
        "command",
        help=f"One of the following COMMANDS: {COMMANDS}",
    )

    args = parser.parse_args(sys.argv[1:2])
    MODULES[args.command]()


if __name__ == "__main__":
    pydicer_cli()
