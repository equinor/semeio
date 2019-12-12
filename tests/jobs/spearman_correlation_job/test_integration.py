import os
import shutil
import sys

import pandas as pd
import pytest
from res.enkf import EnKFMain, ResConfig

from ert_shared.libres_facade import LibresFacade
from semeio.jobs.spearman_correlation_job import job as spearman
from tests.jobs.correlated_observations_scaling.conftest import TEST_DATA_DIR

if sys.version_info >= (3, 3):
    from unittest.mock import Mock
else:
    from mock import Mock


def test_spearman_correlation(monkeypatch):
    df = pd.DataFrame(data=[[7, 8, 9], [10, 11, 12]], index=[1, 2], columns=[0, 1, 2])
    tuples = list(zip(*[df.columns.to_list(), df.columns.to_list()]))
    df.columns = pd.MultiIndex.from_tuples(tuples, names=["key_index", "data_index"])

    facade = Mock()
    mock_data = Mock()
    mock_data.get_simulated_data.return_value = df
    measured_data = Mock(return_value=mock_data)
    scal_job = Mock()
    monkeypatch.setattr(spearman, "scaling_job", scal_job)
    monkeypatch.setattr(spearman, "MeasuredData", measured_data)
    spearman._spearman_correlation(facade, ["A_KEY"], 0.1, False)

    assert measured_data.called_once_with(facade, ["A_KEY"])
    assert scal_job.called_once()


@pytest.mark.skipif(TEST_DATA_DIR is None, reason="no libres test-data")
@pytest.mark.usefixtures("setup_tmpdir")
def test_main_entry_point_gen_data(monkeypatch):
    scal_job = Mock()
    monkeypatch.setattr(spearman, "scaling_job", scal_job)

    test_data_dir = os.path.join(TEST_DATA_DIR, "local", "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("snake_oil.ert")

    ert = EnKFMain(res_config)
    facade = LibresFacade(ert)

    spearman.spearman_job(facade, 1.0, False)

    assert scal_job.call_count == 71
