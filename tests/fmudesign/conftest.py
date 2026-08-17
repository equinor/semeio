import csv
from pathlib import Path

import pytest
from xlsxwriter import Workbook  # type: ignore[import-untyped]
from xlsxwriter.worksheet import Worksheet  # type: ignore[import-untyped]

from tests.fmudesign.workbook_specs import WORKBOOK_SPECS, CellValue, WorkbookSpec

SOURCE_DATA = Path(__file__).parent / "data"


def _write_cell(worksheet: Worksheet, row: int, column: int, value: CellValue) -> None:
    if isinstance(value, str):
        worksheet.write_string(row, column, value)
    elif value is not None:
        worksheet.write_number(row, column, value)


def _write_workbook(path: Path, spec: WorkbookSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Workbook(path) as workbook:
        for sheet_name, rows in spec.items():
            worksheet = workbook.add_worksheet(sheet_name)
            for row_number, values in rows.items():
                for column, value in enumerate(values):
                    _write_cell(worksheet, row_number - 1, column, value)


def _parse_csv_cell(value: str) -> CellValue:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _write_design_summary_workbook(path: Path) -> None:
    with (SOURCE_DATA / "distributions/design.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = {
            row_number: tuple(_parse_csv_cell(value) for value in values)
            for row_number, values in enumerate(csv.reader(stream), start=1)
        }
    # The legacy Excel fixture uses the full parameter name; the CSV abbreviates it.
    rows[1] = (*rows[1][:-1], "RELP_GO_ILETOFTE")
    _write_workbook(path, {"DesignSheet01": rows})


@pytest.fixture(scope="session")
def fmudesign_test_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    test_data = tmp_path_factory.mktemp("fmudesign_test_data")
    config_dir = test_data / "config"
    for filename, spec in WORKBOOK_SPECS.items():
        _write_workbook(config_dir / filename, spec)
    _write_design_summary_workbook(test_data / "distributions/design.xlsx")
    return test_data
