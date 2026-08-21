from pathlib import Path

import pytest

from semeio.fmudesign._workbook import load_workbook_spec, render_workbook

SOURCE_DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def fmudesign_test_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    test_data = tmp_path_factory.mktemp("fmudesign_test_data")
    config_dir = test_data / "config"
    config_dir.mkdir()

    generated_paths: set[Path] = set()
    for spec_path in sorted((SOURCE_DATA / "config").glob("*.yaml")):
        spec = load_workbook_spec(spec_path)
        destination = config_dir / spec_path.with_suffix(".xlsx").name
        output_paths = {
            destination,
            *(config_dir / filename for filename in spec.auxiliary_files),
        }
        duplicate_paths = generated_paths & output_paths
        if duplicate_paths:
            duplicates = ", ".join(sorted(path.name for path in duplicate_paths))
            raise ValueError(
                f"Workbook specifications create duplicate files: {duplicates}"
            )
        render_workbook(spec, destination)
        generated_paths.update(output_paths)

    return test_data
