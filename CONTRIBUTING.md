# Contributing

The following is a set of guidelines for contributing to semeio.

## Ground Rules

1. We use ruff code formatting
1. We do our best to stick to pylint and flake8 code standards
1. All code must be testable and unit tested
1. Commit messages should follow the format as described here https://chris.beams.io/posts/git-commit/

## Pull Request Process

1. Work on your own fork of the main repo
1. Push your commits and make a draft pull request
1. Check that your pull request passes all test
1. When all tests have passed and your are happy with your changes, change your pull request to "ready for review"
   and ask for a code review
1. When your PR has been approved—rebase, squash and merge your changes

### Documentation

Because semeio automatically adds documentation to ert, if any changes to the documentation hooks are
performed, the ert documentation should be built to make sure no errors are introduced.

You can build the documentation after installation by running
```bash
git clone https://github.com/equinor/ert
cd ert
pip install -e .
pip install -r dev-requirements.txt
sphinx-build -n -v -E -W ./docs/rst/manual ./tmp/ert_docs
```
and then open the generated `./tmp/ert_docs/index.html` in a browser.
