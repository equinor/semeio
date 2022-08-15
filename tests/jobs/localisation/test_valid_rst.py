import pytest
import rstcheck_core.checker

from semeio.workflows.localisation.local_config_script import DESCRIPTION, EXAMPLES


@pytest.mark.parametrize("rst_text", [DESCRIPTION, EXAMPLES])
def test_valid_rst(rst_text):
    """
    Check that the documentation passed through the plugin system is
    valid rst
    """
    assert not list(rstcheck_core.checker.check_source(rst_text))
