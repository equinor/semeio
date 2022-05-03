import pytest
import rstcheck

from semeio.workflows.localisation.local_config_script import DESCRIPTION, EXAMPLES


@pytest.mark.parametrize("rst_text", [DESCRIPTION, EXAMPLES])
def test_valid_rst(rst_text):
    """
    Check that the documentation passed through the plugin system is
    valid rst
    """
    assert not list(rstcheck.check(rst_text))
