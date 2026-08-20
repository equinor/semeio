from pathlib import Path

import openpyxl
import pytest
import yaml

from semeio.fmudesign import excel_to_dict
from semeio.fmudesign._workbook import render_workbook

REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLES_DIR = REPOSITORY_ROOT / "src/semeio/fmudesign/examples"
TEST_CONFIG_DIR = REPOSITORY_ROOT / "tests/fmudesign/data/config"
EXCEL_SUFFIXES = {".xls", ".xlsx", ".xlsm", ".xlsb"}

MINIMAL_SPEC = {
    "version": 1,
    "general": {
        "repeats": 1,
        "rms_seeds": "default",
    },
    "sensitivities": [
        {
            "name": "rms_seed",
            "type": "seed",
        }
    ],
    "defaults": {},
}


def test_render_workbook_from_fmudesign_concepts(tmp_path):
    output_path = tmp_path / "example.xlsx"
    created = render_workbook(
        {
            "version": 1,
            "general": {
                "repeats": 3,
                "rms_seeds": "default",
                "distribution_seed": 42,
            },
            "sensitivities": [
                {"name": "rms_seed", "type": "seed"},
                {
                    "name": "faults",
                    "type": "scenario",
                    "cases": ["west", "east"],
                    "parameters": {"FAULT_POSITION": [-1, 1]},
                },
                {
                    "name": "montecarlo",
                    "type": "distribution",
                    "realizations": 5,
                    "parameters": {
                        "PORO": {
                            "distribution": "normal",
                            "values": [0.2, 0.05],
                            "decimals": 3,
                            "correlation": "rock",
                            "dependency": "poro_map",
                        },
                        "PERM": {
                            "distribution": "lognormal",
                            "values": [4.0, 0.5],
                            "correlation": "rock",
                        },
                    },
                },
            ],
            "defaults": {
                "RMS_SEED": 1000,
                "FAULT_POSITION": 0,
                "PORO": 0.2,
                "PERM": 50,
                "NTG": 0.7,
            },
            "background": {
                "parameters": {
                    "NTG": {
                        "distribution": "uniform",
                        "values": [0.5, 0.9],
                        "decimals": 2,
                        "correlation": "background_rock",
                    }
                }
            },
            "correlations": {
                "rock": {
                    "parameters": ["PORO", "PERM"],
                    "matrix": [[1], [0.6, 1]],
                },
                "background_rock": {
                    "parameters": ["NTG"],
                    "matrix": [[1]],
                },
            },
            "dependencies": {
                "poro_map": {
                    "source": "PORO",
                    "values": [0.1, 0.2],
                    "targets": {"FACIES": ["sand", "shale"]},
                }
            },
            "auxiliary_files": {
                "seeds.xlsx": {
                    "rows": [[2000], [2001]],
                }
            },
            "include_instructions": True,
        },
        output_path,
    )

    assert created == [output_path, tmp_path / "seeds.xlsx"]

    parsed = excel_to_dict(output_path)
    assert parsed["repeats"] == 3
    assert parsed["distribution_seed"] == 42
    assert parsed["sensitivities"]["faults"]["cases"] == {
        "west": {"FAULT_POSITION": -1},
        "east": {"FAULT_POSITION": 1},
    }
    montecarlo = parsed["sensitivities"]["montecarlo"]
    assert montecarlo["numreal"] == 5
    assert montecarlo["parameters"]["PORO"] == [
        "normal",
        [0.2, 0.05],
        "rock",
    ]
    assert montecarlo["dependencies"]["PORO"] == {
        "from_values": ["0.1", "0.2"],
        "to_params": {"FACIES": ["sand", "shale"]},
    }
    assert parsed["background"]["parameters"]["NTG"] == [
        "uniform",
        [0.5, 0.9],
        "background_rock",
    ]

    workbook = openpyxl.load_workbook(output_path)
    assert workbook.sheetnames == [
        "general_input",
        "designinput",
        "defaultvalues",
        "background",
        "rock",
        "background_rock",
        "poro_map",
        "INFO",
    ]
    assert all(
        cell.comment is None
        for worksheet in workbook.worksheets
        for row in worksheet.iter_rows()
        for cell in row
    )

    designinput = workbook["designinput"]
    assert designinput["A1"].font.bold
    assert designinput["A1"].fill.fgColor.rgb == "FF1F4E78"
    assert designinput["Q1"].value == "dependencies"
    assert designinput.freeze_panes == "A2"

    seed_color = designinput["A2"].fill.fgColor.rgb
    faults_color = designinput["A3"].fill.fgColor.rgb
    montecarlo_color = designinput["A4"].fill.fgColor.rgb
    assert len({seed_color, faults_color, montecarlo_color}) == 3
    assert designinput["Q4"].fill.fgColor.rgb == montecarlo_color
    assert designinput["A5"].fill.fgColor.rgb == montecarlo_color

    assert workbook["INFO"]["A1"].value.startswith("FMU-design example")
    assert "share a color" in workbook["INFO"]["A2"].value

    seeds = openpyxl.load_workbook(tmp_path / "seeds.xlsx")
    assert [seeds.active.cell(row=row, column=1).value for row in (1, 2)] == [
        2000,
        2001,
    ]


def test_dependencies_column_is_omitted_when_unused(tmp_path):
    output_path = tmp_path / "without-dependencies.xlsx"
    render_workbook(MINIMAL_SPEC, output_path)

    worksheet = openpyxl.load_workbook(output_path)["designinput"]
    headers = [cell.value for cell in worksheet[1]]

    assert len(headers) == 16
    assert "dependencies" not in headers


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (
            {**MINIMAL_SPEC, "version": 2},
            "version",
        ),
        (
            {**MINIMAL_SPEC, "unknown": True},
            "unknown",
        ),
        (
            {
                **MINIMAL_SPEC,
                "sensitivities": [
                    {"name": "same", "type": "seed"},
                    {"name": "same", "type": "seed"},
                ],
            },
            "Sensitivity names must be unique",
        ),
        (
            {
                **MINIMAL_SPEC,
                "sensitivities": [
                    {
                        "name": "scenario",
                        "type": "scenario",
                        "cases": ["low", "high"],
                        "parameters": {"A": [1]},
                    }
                ],
            },
            "one value per case",
        ),
        (
            {
                **MINIMAL_SPEC,
                "sensitivities": [
                    {
                        "name": "distribution",
                        "type": "distribution",
                        "parameters": {
                            "A": {
                                "distribution": "normal",
                                "values": [0, 1],
                                "correlation": "missing",
                            }
                        },
                    }
                ],
            },
            "Unknown correlation",
        ),
        (
            {
                **MINIMAL_SPEC,
                "correlations": {
                    "invalid": {
                        "parameters": ["A", "B"],
                        "matrix": [[1], [0.5]],
                    }
                },
            },
            "lower-triangular",
        ),
        (
            {
                **MINIMAL_SPEC,
                "auxiliary_files": {
                    "../outside.xlsx": {
                        "rows": [[1]],
                    }
                },
            },
            "simple filenames",
        ),
        (
            {
                **MINIMAL_SPEC,
                "auxiliary_files": {
                    "invalid.xlsx": {
                        "rows": [[1]],
                    }
                },
            },
            "cannot replace the main workbook",
        ),
    ],
)
def test_render_workbook_rejects_invalid_specs(tmp_path, spec, match):
    with pytest.raises(ValueError, match=match):
        render_workbook(spec, tmp_path / "invalid.xlsx")


def test_example_sources_contain_domain_data_only():
    for directory in (EXAMPLES_DIR, TEST_CONFIG_DIR):
        for path in directory.glob("*.yaml"):
            spec = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert "formats" not in spec
            assert "worksheets" not in spec


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
