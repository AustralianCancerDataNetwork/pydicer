# Contributing

PyDicer welcomes any and all contributions in the way of new functionality, bug fixes or documentation. This document provides some guidance to developers who would like to contribute to the project.

## Git

Create a branch off of **main** while you make your changes or implement your new tool.
Once complete, head to  [GitHub to create a pull
request](https://github.com/australiancancerdatanetwork/pydicer/compare) to merge your changes
into the **main** branch. At this point the automated tests will run and maintainers will review
your submission before merging.

## Poetry

PyDicer uses poetry to manage dependencies. Instructions for installing poetry are available
[here](https://python-poetry.org/docs/#installation). Once installed, you can easily install the
libraries required to develop for PyDicer using the following command:

```bash
poetry install --with dev,docs --all-extras
```

This will automatically create a virtual environment managed by poetry. To run a script within this
environment, use the `poetry run` followed by what to run. For example, to run a test.py script:

```bash
poetry run python test.py
```

## VSC Devcontainer

You may setup a Visual Studio Code development container (Devcontainer) to ensure a standardised
development and testing environment, without the need to perform overhead installation. This
assumes that Docker and VSC are installed on your system.

To set this up, you may perform the VSC shortcut `ctrl + shift + p` (or `cmd + shift p` on Mac) and
select the `Reopen in devcontainer` option. This will create a Docker container with Python 3.9
and its dependencies installed, along with other tools we use for development (eg. git, pytest).

## Coding standards

Code in PyDicer must conform to Python's PEP-8 standards to ensure consistent formatting between contributors. To ensure this, pylint is used to check code conforms to these standards before a Pull Request can be merged. You can run pylint from the command line using the following command:

```bash
pylint pydicer
```

But a better idea is to ensure you are using a Python IDE which supports linting (such as [VSCode](https://code.visualstudio.com/docs/python/linting) or PyCharm). Make sure you resolve all suggestions from pylint before submitting your pull request.

If you're new to using pylint, you may like to [read this guide](https://docs.pylint.org/en/v2.11.1/tutorial.html).

## Automated tests

A test suite is included in PyDicer which ensures that code contributed to the repository functions as expected and continues to function as further development takes place. Any code submitted via a pull request should include appropriate automated tests for the new code.

pytest is used as a testing library. Running the tests from the command line is really easy:

```bash
pytest
```

Add your tests to the appropriate file in the `tests/` directory. See the [pytest documention](https://docs.pytest.org/en/6.2.x/getting-started.html) for more information.
