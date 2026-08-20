from pathlib import Path

import pytest

from semeio.fmudesign._workbook import render_workbook_resource

SOURCE_DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def fmudesign_test_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    test_data = tmp_path_factory.mktemp("fmudesign_test_data")
    config_dir = test_data / "config"
    config_dir.mkdir()

    for spec_path in sorted((SOURCE_DATA / "config").glob("*.yaml")):
        render_workbook_resource(
            spec_path,
            config_dir / spec_path.with_suffix(".xlsx").name,
        )

    return test_data
