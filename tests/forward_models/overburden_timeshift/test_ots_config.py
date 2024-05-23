import shutil
from pathlib import Path

import pytest
import rstcheck_core.checker
from pydantic import ValidationError

from semeio._docs_utils._json_schema_2_rst import _create_docs
from semeio.forward_models.overburden_timeshift.ots_config import OTSConfig

TEST_NORNE_DIR = Path(__file__).parent / ".." / ".." / "test_data" / "norne"


@pytest.fixture(name="conf")
def conf_fixture():
    yield {
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


@pytest.fixture()
def ecl_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for extension in ["INIT", "EGRID", "UNRST"]:
        shutil.copy(TEST_NORNE_DIR / f"NORNE_ATW2013.{extension}", tmp_path)


@pytest.mark.parametrize("vintages_export_file", ["ts.txt", None])
@pytest.mark.parametrize("horizon", ["horizon.irap", None])
@pytest.mark.parametrize("velocity_model", ["norne_vol.segy", None])
@pytest.mark.usefixtures("ecl_files")
def test_valid_config(conf, velocity_model, horizon, vintages_export_file):
    if velocity_model:
        conf.update({"velocity_model": velocity_model})
    if horizon:
        conf.update({"horizon": horizon})
    if vintages_export_file:
        conf.update({"vintages_export_file": vintages_export_file})
    OTSConfig(**conf)


@pytest.mark.parametrize(
    "extension, error_msg",
    [
        ["INIT", r"missing required file\(s\): \['NORNE_ATW2013.INIT'\] "],
        ["EGRID", r"missing required file\(s\): \['NORNE_ATW2013.EGRID'\] "],
        ["UNRST", r"missing required file\(s\): \['NORNE_ATW2013.UNRST'\] "],
    ],
)
@pytest.mark.usefixtures("ecl_files")
def test_eclbase_config(extension, error_msg, conf):
    # Renaming a needed ecl file to simulate it not existing
    Path(f"NORNE_ATW2013.{extension}").rename("NOT_ECLBASE")
    with pytest.raises(ValidationError, match=error_msg):
        OTSConfig(**conf)


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
def test_valid_file_format(file_format, conf):
    conf.update({"file_format": file_format})
    OTSConfig(**conf)


@pytest.mark.usefixtures("ecl_files")
def test_invalid_file_format(conf):
    conf.update({"file_format": "not a file format"})
    with pytest.raises(ValidationError, match="file_format\n  Input should be"):
        OTSConfig(**conf)


@pytest.mark.usefixtures("ecl_files")
def test_missing_date(conf):
    conf["vintages"]["ts_simple"] = [["1997-11-06", "2024-01-03"]]
    with pytest.raises(ValidationError, match="Vintages with dates not found"):
        OTSConfig(**conf)


@pytest.mark.usefixtures("ecl_files")
def test_extra_date(conf):
    conf["vintages"]["ts_simple"] = [["1997-11-06", "1998-02-01", "1997-12-17"]]
    with pytest.raises(ValidationError, match="List should have at most 2 items"):
        OTSConfig(**conf)


def test_docs_generation(snapshot):
    docs = _create_docs(
        OTSConfig.model_json_schema(by_alias=False, ref_template="{model}")
    )
    snapshot.assert_match(
        docs,
        "rst_docs",
    )
    assert not list(rstcheck_core.checker.check_source(docs))
