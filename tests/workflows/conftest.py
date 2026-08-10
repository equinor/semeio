import pytest
from ert.config import ErtConfig


@pytest.fixture
def snake_oil_config(
    copy_snake_oil_case_storage,
):  # pylint: disable=unused-argument
    return ErtConfig.from_file("snake_oil.ert")
