"""
Command Line Interface tool to run pydicer pipeline or specific modules on their own

usage: python -m pydicer.cli.run ['pipeline'|'input']

pydicer CLI (Command Line Interface)

positional arguments:
  command     One of the following COMMANDS: ['pipeline'|'input']

optional arguments:
  -h, --help  show help message
"""
import argparse
from argparse import RawTextHelpFormatter
import sys

from pydicer.cli.contants import get_sub_help_mesg
from pydicer.cli.input import testinput_cli, pacs_cli, tcia_cli, web_cli, run_pipeline

# Sub command types for the Input command
INPUT_TOOLS = {
    "test": testinput_cli,
    "pacs": pacs_cli,
    "tcia": tcia_cli,
    "web": web_cli,
}

PIPELINE_TOOLS = {
    # "e2e": run_test, TODO This broke due to some changes. Either we need to fix or remove.
    "filesystem": run_pipeline,
    "test": run_pipeline,
    "pacs": run_pipeline,
    "tcia": run_pipeline,
    "web": run_pipeline,
}


def parse_sub_input(command):
    """function to parse the input command args"""
    parse_sub_command(command, "Run the Input module only", INPUT_TOOLS, "test", INPUT_COMMANDS)


def parse_sub_pipeline(command):
    """function to parse the pipeline command args"""
    parse_sub_command(
        command,
        "Run the pipeline with a specific input method",
        PIPELINE_TOOLS,
        "e2e",
        PIPELINE_COMMANDS,
    )


INPUT_COMMANDS = str(list(INPUT_TOOLS.keys())).replace(", ", "|")
PIPELINE_COMMANDS = str(list(PIPELINE_TOOLS.keys())).replace(", ", "|")
MODULES = {"pipeline": parse_sub_pipeline, "input": parse_sub_input}
COMMANDS = str(list(MODULES.keys())).replace(", ", "|")


def parse_sub_command(command, desc, tools, default_choice, help_commands):
    """Generic function to take in dynamic input and trigger the respective sub commands

    Args:
        desc (str): help description of what the sub command does
        tools (dict): dictionary of which sub command type can be run
        default_choice (str): default sub command type that will be run in the case no input is
        received from the user
    """
    parser = argparse.ArgumentParser(description=desc, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--type",
        help=get_sub_help_mesg(help_commands, command),
        default=default_choice,
        choices=tools,
    )

    args = parser.parse_args(sys.argv[2:4])
    if command == "pipeline":
        tools[args.type](args.type, *sys.argv[4:])
    else:
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
    MODULES[args.command](sys.argv[1])


if __name__ == "__main__":
    pydicer_cli()
