[![PyPI version](https://badge.fury.io/py/semeio.svg)](https://badge.fury.io/py/semeio)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/semeio)](https://img.shields.io/pypi/pyversions/semeio)
[![Actions Status](https://github.com/equinor/semeio/workflows/CI/badge.svg)](https://github.com/equinor/semeio/actions?query=workflow=CI)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# semeio

Semeio is a collection of forward models and workflows used in [ERT](https://github.com/equinor/ert). These are
exposing end points which is considered the API of semeio. If there are submodules that can be applied
more generally, or have use outside these forward models and workflows, please create an issue and it can be exposed in
the API.

# Installation

Semeio is available on [pypi](https://pypi.org/project/semeio/) and can be installed using `pip install semeio`.

```sh
# Install
pip install semeio
```

# Usage

Once installed semeio will automatically register its workflows and forward models with
[ERT](https://github.com/equinor/ert). Through the plugin hooks it will also add its own documentation to the [ERT](https://github.com/equinor/ert)
documentation. See the [ERT](https://github.com/equinor/ert) documentation for examples on
how to run workflows and forward models, and build the [ERT](https://github.com/equinor/ert) documentation to get
documentation for the workflows and forward models.

## Developing

We use uv to have one synchronized development environment for all packages.
See [installing uv](https://docs.astral.sh/uv/getting-started/installation/). We
recommend installing uv using your system's package manager, or into a small
dedicated virtual environment.

Once uv is installed, you can get a development environment by running:

```sh
git clone https://github.com/equinor/semeio
cd semeio
uv sync --all-groups --all-extras
```

# Run tests
To run the full test suite, do:

```sh
uv run pytest tests
```

[pre-commit](https://pre-commit.com/) is used to comply with the formatting standards.
The complete formatting tests can be run with:

```sh
uv run pre-commit run --all-files
```

Formatting use `ruff`, See `.pre-commit-config.yaml` for the
complete steps.

[pre-commit](https://pre-commit.com/) can also provide git hooks to run on every commit
to avoid committing with formatting errors. This will only run on the diff so is quite fast.
To configure this, run:

```sh
uv run pre-commit install
```

After this the hook will run on every commit.

If you would like to remove the hooks, run:

```sh
uv run pre-commit uninstall
```
