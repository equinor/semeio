from datetime import date
from pathlib import Path

import openpyxl
import pytest

from semeio.fmudesign._workbook import render_workbook

REPOSITORY_ROOT = Path(__file__).parents[2]
EXCEL_SUFFIXES = {".xls", ".xlsx", ".xlsm", ".xlsb"}


def test_render_workbook(tmp_path):
    output_path = tmp_path / "example.xlsx"
    render_workbook(
        {
            "version": 1,
            "formats": {
                "header": {
                    "bold": True,
                    "font_color": "#FFFFFF",
                    "bg_color": "#1F4E78",
                    "align": "center",
                },
                "date": {"num_format": "yyyy-mm-dd"},
            },
            "worksheets": [
                {
                    "name": "general_input",
                    "rows": {
                        1: ["name", "value", "effective_date"],
                        2: ["rms_seeds", "00123", date(2026, 8, 20)],
                        4: ["distribution_seed", 42, None],
                    },
                    "cell_formats": {
                        "A1": "header",
                        "B1": "header",
                        "C1": "header",
                        "C2": "date",
                        "B3": "header",
                    },
                    "comments": {
                        "B2": {
                            "text": "Keep leading zeroes.",
                            "author": "FMU-design",
                        }
                    },
                    "column_widths": {"A": 24.0, "B:C": 18.0},
                    "row_heights": {1: 24.0},
                    "default_column_width": 9.0,
                    "default_row_height": 16.0,
                    "zoom": 145,
                    "orientation": "landscape",
                    "paper_size": 9,
                    "print_scale": 90,
                },
                {
                    "name": "empty_sheet",
                    "rows": {},
                },
            ],
        },
        output_path,
    )

    workbook = openpyxl.load_workbook(output_path)
    assert workbook.sheetnames == ["general_input", "empty_sheet"]

    worksheet = workbook["general_input"]
    assert worksheet["B2"].value == "00123"
    assert worksheet["C2"].value.date() == date(2026, 8, 20)
    assert worksheet["A4"].value == "distribution_seed"
    assert worksheet["B4"].value == 42
    assert worksheet["B3"].value is None
    assert worksheet["B3"].fill.fill_type == "solid"
    assert worksheet["B2"].comment.text == "Keep leading zeroes."
    assert worksheet["B2"].comment.author == "FMU-design"
    assert worksheet["A1"].font.bold
    assert worksheet["A1"].font.color.rgb == "FFFFFFFF"
    assert worksheet["A1"].fill.fgColor.rgb == "FF1F4E78"
    assert worksheet["A1"].alignment.horizontal == "center"
    assert worksheet.column_dimensions["A"].width == pytest.approx(24.7109375)
    assert worksheet.column_dimensions["D"].max == 16_384
    assert worksheet.column_dimensions["D"].width == pytest.approx(9.7109375)
    assert worksheet.row_dimensions[1].height == pytest.approx(24.0)
    assert worksheet.sheet_view.zoomScale == 145
    assert worksheet.sheet_format.defaultRowHeight == pytest.approx(16.0)
    assert worksheet.page_setup.orientation == "landscape"
    assert worksheet.page_setup.paperSize == 9
    assert worksheet.page_setup.scale == 90


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (
            {"version": 2, "formats": {}, "worksheets": []},
            "version",
        ),
        (
            {
                "version": 1,
                "formats": {},
                "worksheets": [{"name": "sheet", "rows": {}, "unknown": True}],
            },
            "unknown",
        ),
        (
            {
                "version": 1,
                "formats": {},
                "worksheets": [
                    {"name": "sheet", "rows": {}},
                    {"name": "sheet", "rows": {}},
                ],
            },
            "Worksheet names must be unique",
        ),
        (
            {
                "version": 1,
                "formats": {},
                "worksheets": [
                    {
                        "name": "sheet",
                        "rows": {},
                        "cell_formats": {"not-a-cell": "missing"},
                    }
                ],
            },
            "cell",
        ),
        (
            {
                "version": 1,
                "formats": {},
                "worksheets": [
                    {
                        "name": "sheet",
                        "rows": {1: ["x" * 32_768]},
                    }
                ],
            },
            "Failed to write cell A1",
        ),
    ],
)
def test_render_workbook_rejects_invalid_specs(tmp_path, spec, match):
    with pytest.raises(ValueError, match=match):
        render_workbook(spec, tmp_path / "invalid.xlsx")


def test_fmudesign_resources_do_not_contain_excel_workbooks():
    workbook_resources = [
        path.relative_to(REPOSITORY_ROOT)
        for root in (
            REPOSITORY_ROOT / "src/semeio/fmudesign",
            REPOSITORY_ROOT / "tests/fmudesign",
        )
        for path in root.rglob("*")
        if path.suffix.lower() in EXCEL_SUFFIXES
    ]

    assert workbook_resources == []
