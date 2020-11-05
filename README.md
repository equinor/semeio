[![PyPI version](https://badge.fury.io/py/semeio.svg)](https://badge.fury.io/py/semeio)
[![Actions Status](https://github.com/equinor/semeio/workflows/CI/badge.svg)](https://github.com/equinor/semeio/actions?query=workflow=CI)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# semeio #

Semeio is a collection of jobs and workflow jobs used in [ERT](https://github.com/equinor/ert). These are
exposing end points which is considered the API of semeio. If there are submodules that can be applied
more generally, or have use outside these jobs and workflows, please create an issue and it can be exposed in
the API.

# Installation and usage

Semeio is available on [pypi](https://pypi.org/project/semeio/) and can be installed using `pip install semeio`.

```sh
# Install
pip install semeio
```

## Usage

Once installed semeio will automatically register its workflows and forward model jobs with
[ERT](https://github.com/equinor/ert). Through the plugin hooks it will also add its own documentation to the [ERT](https://github.com/equinor/ert)
documentation. See the [ERT](https://github.com/equinor/ert) documentation for examples on
how to run workflows and forward model jobs, and build the [ERT](https://github.com/equinor/ert) documentation to get
documentation for the workflows and forward model jobs.

## Run tests
[tox](https://tox.readthedocs.io/en/latest/) is used as the test facilitator,
to run the full test suite:

```sh
# Test
pip install tox
tox
```

or to run it for a particular Python version (in this case Python 3.7):

```sh
# Test
pip install tox
tox -e py37
```

or to run it for a the current Python version:

```sh
# Test
pip install tox
tox -e py
```

[pytest](https://docs.pytest.org/en/latest/) is used as the test runner, so for quicker
iteration it is possible to run:

```sh
# Test
pytest
```

this requires that the test dependencies from `test_requirements.txt` are installed.

```sh
# Install test requirements
pip install -r test_requirements.txt
```

[pre-commit](https://pre-commit.com/) is used to comply with the formatting standards.
The complete formatting tests can be run with:

```sh
pip install tox
tox -e style
```

Formatting tests include `black`, `flake8` and `pylint`, See `.pre-commit-config.yaml` for the
complete steps.

[pre-commit](https://pre-commit.com/) can also provide git hooks to run on every commit
to avoid commiting with formatting errors. This will only run on the diff so is quite fast.
To configure this, run:

```sh
pip install -r test_requirements.txt
pip install pre-commit
pre-commit install
```

After this the hook will run on every commit.

If you would like to remove the hooks, run:

```sh
pre-commit uninstall
```
