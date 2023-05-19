import shutil
from pathlib import Path

import pytest
import yaml

from semeio._exceptions.exceptions import ConfigurationError
from semeio.jobs.overburden_timeshift.ots import ots_load_params

TEST_NORNE_DIR = Path(__file__).parent / ".." / ".." / "test_data" / "norne"

# pylint: disable=duplicate-code


@pytest.fixture()
def ecl_files(tmpdir):
    for extension in ["INIT", "EGRID", "UNRST"]:
        shutil.copy(TEST_NORNE_DIR / f"NORNE_ATW2013.{extension}", tmpdir)


@pytest.mark.parametrize("vintages_export_file", ["ts.txt", None])
@pytest.mark.parametrize("horizon", ["horizon.irap", None])
@pytest.mark.parametrize("velocity_model", ["norne_vol.segy", None])
@pytest.mark.usefixtures("ecl_files")
def test_valid_config(tmpdir, velocity_model, horizon, vintages_export_file):
    conf = {
        "eclbase": "NORNE_ATW2013",
        "above": 100,
        "seabed": 300,
        "youngs": 0.5,
        "poisson": 0.3,
        "rfactor": 20,
        "mapaxes": False,
        "convention": 1,
        "output_dir": "ts",
        "vintages": {
            "ts_simple": [["1997-11-06", "1998-02-01"], ["1997-12-17", "1998-02-01"]],
            "dpv": [["1997-11-06", "1997-12-17"]],
        },
    }
    if velocity_model:
        conf.update({"velocity_model": velocity_model})
    if horizon:
        conf.update({"horizon": horizon})
    if vintages_export_file:
        conf.update({"vintages_export_file": vintages_export_file})
    with tmpdir.as_cwd():
        with open("ots_config.yml", "w", encoding="utf-8") as file:
            yaml.dump(conf, file, default_flow_style=False)
        ots_load_params("ots_config.yml")


@pytest.mark.parametrize(
    "extension, error_msg",
    [
        ["INIT", "Eclbase must have an INIT file"],
        ["EGRID", "Eclbase must have an EGRID file"],
        ["UNRST", "Eclbase must have a UNRST file"],
    ],
)
@pytest.mark.usefixtures("ecl_files")
def test_eclbase_config(tmpdir, extension, error_msg):
    conf = {
        "eclbase": "NORNE_ATW2013",
        "above": 100,
        "seabed": 300,
        "youngs": 0.5,
        "poisson": 0.3,
        "rfactor": 20,
        "mapaxes": False,
        "convention": 1,
        "output_dir": "ts",
        "horizon": "horizon.irap",
        "vintages_export_file": "ts.txt",
        "velocity_model": "norne_vol.segy",
        "vintages": {
            "ts_simple": [["1997-11-06", "1998-02-01"], ["1997-12-17", "1998-02-01"]],
            "dpv": [["1997-11-06", "1997-12-17"]],
        },
    }
    with tmpdir.as_cwd():
        with open("ots_config.yml", "w", encoding="utf-8") as file:
            yaml.dump(conf, file, default_flow_style=False)
        # Renaming a needed ecl file to simulate it not existing
        Path(f"NORNE_ATW2013.{extension}").rename("NOT_ECLBASE")
        with pytest.raises(ConfigurationError, match=error_msg):
            ots_load_params("ots_config.yml")


@pytest.mark.parametrize(
    "file_format",
    [
        "irap_ascii",
        "irapascii",
        "irap_txt",
        "irapasc",
        "irap_binary",
        "irapbinary",
        "irapbin",
        "irap",
        "gri",
        "zmap",
        "storm_binary",
        "petromod",
        "ijxyz",
    ],
)
@pytest.mark.usefixtures("ecl_files")
def test_valid_file_format(tmpdir, file_format):
    conf = {
        "eclbase": "NORNE_ATW2013",
        "above": 100,
        "seabed": 300,
        "youngs": 0.5,
        "poisson": 0.3,
        "rfactor": 20,
        "mapaxes": False,
        "convention": 1,
        "output_dir": "ts",
        "vintages": {
            "ts_simple": [["1997-11-06", "1998-02-01"], ["1997-12-17", "1998-02-01"]],
            "dpv": [["1997-11-06", "1997-12-17"]],
        },
    }
    conf.update({"file_format": file_format})
    with tmpdir.as_cwd():
        with open("ots_config.yml", "w", encoding="utf-8") as file:
            yaml.dump(conf, file, default_flow_style=False)
        ots_load_params("ots_config.yml")


@pytest.mark.usefixtures("ecl_files")
def test_invalid_file_format(tmpdir):
    conf = {
        "eclbase": "NORNE_ATW2013",
        "above": 100,
        "seabed": 300,
        "youngs": 0.5,
        "poisson": 0.3,
        "rfactor": 20,
        "mapaxes": False,
        "convention": 1,
        "output_dir": "ts",
        "vintages": {
            "ts_simple": [["1997-11-06", "1998-02-01"], ["1997-12-17", "1998-02-01"]],
            "dpv": [["1997-11-06", "1997-12-17"]],
        },
    }
    conf.update({"file_format": "not a file format"})
    with tmpdir.as_cwd():
        with open("ots_config.yml", "w", encoding="utf-8") as file:
            yaml.dump(conf, file, default_flow_style=False)
        with pytest.raises(ConfigurationError, match="valid file type is false"):
            ots_load_params("ots_config.yml")
