"""Test STEA forward model towards the HTTP STEA library

"""
from pathlib import Path
import sys
import pytest
from ert.shared.main import main

from stea import SteaKeys

base_stea_config = """
project-id: 56892
project-version: 1
config-date: 2018-11-01 00:00:00
ecl-profiles:
  FOPT:
    ecl-key: FOPT
ecl-case: {}
results:
  - NPV
"""

base_ert_config = """
RUNPATH realization-%d/iter-%d
ECLBASE {ecl_base}
QUEUE_SYSTEM LOCAL
NUM_REALIZATIONS 1

FORWARD_MODEL STEA(<CONFIG>=<CONFIG_PATH>/stea_conf.yml{fm_options})
"""


ECLDIR_NORNE = Path(__file__).absolute().parent.parent.parent / "test_data" / "norne"


@pytest.mark.usefixtures("setup_tmpdir")
@pytest.mark.parametrize(
    "ecl_case, ecl_base, fm_options",
    [
        (ECLDIR_NORNE / "NORNE_ATW2013", "random_eclbase", ""),
        ("a_random_ecl_case", ECLDIR_NORNE / "NORNE_ATW2013", ",<ECL_CASE>=<ECLBASE>"),
    ],
)
def test_stea_fm_single_real(monkeypatch, httpserver, ecl_case, ecl_base, fm_options):
    httpserver.expect_request(
        "/foobar/api/v1/Alternative/56892/1/summary",
        query_string="ConfigurationDate=2018-11-01T00:00:00",
    ).respond_with_json(
        {
            "AlternativeId": 1,
            "AlternativeVersion": 1,
            "Profiles": [{"Id": "FOPT", "Unit": "SM3"}],
        }
    )
    httpserver.expect_request(
        "/foobar/api/v1/Calculate/", method="POST"
    ).respond_with_json(
        {
            SteaKeys.KEY_VALUES: [
                {SteaKeys.TAX_MODE: SteaKeys.CORPORATE, SteaKeys.VALUES: {"NPV": 30}}
            ]
        },
    )
    server = httpserver.url_for("/foobar")
    # Mock a forward model configuration file
    Path("stea_conf.yml").write_text(
        base_stea_config.format(ecl_case) + f"\nstea_server: {server}"
    )

    # Write an ERT config file
    Path("config.ert").write_text(
        base_ert_config.format(ecl_base=ecl_base, fm_options=fm_options)
    )

    monkeypatch.setattr(sys, "argv", ["ert", "test_run", "config.ert", "--verbose"])
    main()

    assert Path("realization-0/iter-0/NPV_0").read_text()[0:3] == "30\n"
    assert Path("realization-0/iter-0/stea_response.json").exists()
