import sys
import os

import pytest


@pytest.mark.skipif(sys.version_info < (3, 6), reason="requires Python3")
def test_code_style():
    from pathlib import Path
    import black
    from click.testing import CliRunner

    root = str(Path(__file__).parent.parent)

    runner = CliRunner()
    resp = runner.invoke(
        black.main,
        [
            "--check",
            os.path.join(root, "tests"),
            os.path.join(root, "semeio"),
            os.path.join(root, "setup.py"),
            "--exclude",
            "semeio/version.py",
        ],
    )

    assert (
        resp.exit_code == 0
    ), "Black would still reformat one or more files:\n{}".format(resp.output)
