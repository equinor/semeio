# from semeio._docs_utils._json_schema_2_rst import _create_docs
# from semeio.workflows.localisation.localisation_config import LocalisationConfig
from semeio.workflows.localisation.local_config_script import DESCRIPTION, EXAMPLES
import rstcheck


def test_valid_rst():
    """
    Check that the documentation passed through the plugin system is
    valid rst
    """
    assert not list(rstcheck.check(DESCRIPTION))
    assert not list(rstcheck.check(EXAMPLES))
