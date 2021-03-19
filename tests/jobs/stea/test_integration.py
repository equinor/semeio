"""Test STEA forward model towards the HTTP STEA library

Only works on-prem.
"""
from pathlib import Path
import sys
import subprocess
import shutil
import requests

import pytest

try:
    import stea  # noqa

    HAVE_STEA = True
except ImportError:
    HAVE_STEA = False


def on_prem():
    try:
        stea_server = "https://ws2291.statoil.net"  # From stea/stea_input.py
        requests.get(stea_server, verify=False)
        return True
    except requests.exceptions.ConnectionError:
        return False


ECLDIR_NORNE = Path(__file__).absolute().parent.parent.parent / "test_data" / "norne"


@pytest.mark.skipif(
    not on_prem(), reason="Integration with STEA server can only be tested on-prem"
)
@pytest.mark.skipif(
    sys.platform == "darwin",
    reason=(
        "Forward models have same name as executable and "
        "libres will find wrong exe on case insensitive system "
        "bug report here: https://github.com/equinor/libres/issues/1053"
    ),
)
@pytest.mark.skipif(HAVE_STEA is False, reason="STEA library not installed")
def test_stea_fm_single_real(tmpdir):
    tmpdir.chdir()
    Path("realization-0/iter-0/eclipse/model").mkdir(parents=True)

    print(ECLDIR_NORNE)
    for filename in ECLDIR_NORNE.glob("NORNE_ATW2013.*"):
        print(filename)
        shutil.copy(
            filename, Path("realization-0/iter-0/eclipse/model") / filename.name
        )

    # Mock a forward model configuration file
    Path("realization-0/iter-0/stea_conf.yml").write_text(
        """
project-id: 56892
project-version: 1
config-date: 2018-11-01 00:00:00
ecl-profiles:
  FOPT:
    ecl-key: FOPT
ecl-case: eclipse/model/NORNE_ATW2013
results:
  - NPV
  - IRR
  - CEI
"""
    )

    # Write an ERT config file
    Path("config.ert").write_text(
        """
RUNPATH realization-%d/iter-%d
ECLBASE eclipse/model/NORNE_ATW2013 -- NB: no realization suffix
QUEUE_SYSTEM LOCAL
NUM_REALIZATIONS 1

FORWARD_MODEL STEA(<CONFIG>=stea_conf.yml)
"""
    )

    # pylint: disable=subprocess-run-check
    # (assert on the return code further down)
    result = subprocess.run(
        ["ert", "test_run", "config.ert"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdouterr = result.stdout.decode() + result.stderr.decode()
    print(stdouterr)
    # Print any stderr files from the forward model:
    for stderrfile in Path("realization-0").glob("iter-*/*stderr*"):
        if stderrfile.stat().st_size > 0:
            print(stderrfile)
            print(stderrfile.read_text())

    assert Path("realization-0/iter-0/NPV_0").read_text()[0:3] == "146"
    assert Path("realization-0/iter-0/stea_response.json").exists()
