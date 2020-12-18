import shutil
import yaml
import pytest
from semeio._exceptions.exceptions import ConfigurationError
from semeio.jobs.overburden_timeshift.ots import ots_load_params
from pathlib import Path


TEST_NORNE_DIR = Path(__file__).parent / ".." / ".." / "test_data" / "norne"


@pytest.fixture()
def ecl_files(tmpdir):
    for extension in ["INIT", "EGRID", "UNRST"]:
        shutil.copy(TEST_NORNE_DIR / f"NORNE_ATW2013.{extension}", tmpdir)


@pytest.mark.parametrize("vintages_export_file", ["ts.txt", None])
@pytest.mark.parametrize("horizon", ["horizon.irap", None])
@pytest.mark.parametrize("velocity_model", ["norne_vol.segy", None])
def test_valid_config(
    tmpdir, monkeypatch, velocity_model, horizon, vintages_export_file, ecl_files
):
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
        with open("ots_config.yml", "w") as f:
            yaml.dump(conf, f, default_flow_style=False)
        ots_load_params("ots_config.yml")


@pytest.mark.parametrize(
    "extension, error_msg",
    [
        ["INIT", "Eclbase must have an INIT file"],
        ["EGRID", "Eclbase must have an EGRID file"],
        ["UNRST", "Eclbase must have a UNRST file"],
    ],
)
def test_eclbase_config(tmpdir, monkeypatch, ecl_files, extension, error_msg):
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
        with open("ots_config.yml", "w") as f:
            yaml.dump(conf, f, default_flow_style=False)
        # Renaming a needed ecl file to simulate it not existing
        Path(f"NORNE_ATW2013.{extension}").rename("NOT_ECLBASE")
        with pytest.raises(ConfigurationError, match=error_msg):
            ots_load_params("ots_config.yml")
