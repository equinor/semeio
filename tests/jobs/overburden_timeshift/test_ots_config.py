import yaml
import pytest
from semeio._exceptions.exceptions import ConfigurationError
from semeio.jobs.overburden_timeshift.ots import ots_load_params
from semeio.jobs.overburden_timeshift import ots
from unittest import mock
from datetime import datetime


@pytest.mark.parametrize("vintages_export_file", ["ts.txt", None])
@pytest.mark.parametrize("horizon", ["horizon.irap", None])
@pytest.mark.parametrize("velocity_model", ["norne_vol.segy", None])
def test_valid_config(
    tmpdir, monkeypatch, velocity_model, horizon, vintages_export_file
):
    dates = ["1997-11-06", "1997-12-17", "1998-02-01", "1998-02-01"]
    context_mock = mock.Mock(
        return_value=[datetime.strptime(x, "%Y-%m-%d").date() for x in dates]
    )
    monkeypatch.setattr(ots, "extract_ots_context", context_mock)
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


def test_invalid_config(tmpdir, monkeypatch):
    dates = []
    context_mock = mock.Mock(
        return_value=[datetime.strptime(x, "%Y-%m-%d").date() for x in dates]
    )
    monkeypatch.setattr(ots, "extract_ots_context", context_mock)
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
            with pytest.raises(ConfigurationError):
                ots_load_params("ots_config.yml")
