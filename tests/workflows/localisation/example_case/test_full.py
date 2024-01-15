"""
Test of full workflow using non-adaptive localisation
- Gaussian fields are simulated
- Response variables are upscaled values of simulated gaussian fields
- Some selected positions within the upscaled grid are used to extract
  synthetic observations.
- Localisation config file is generated.
- Running ERT using the generated localisation file.
- The ERT forward model (simulating and upscaling gaussian fields
  is done by the script 'sim_fields.py')
- For iteration > 0, the updated field parameter is imported from ERT
  and upscaled. Predictions of observed values are extracted and
  saved to ERT using GEN_DATA keyword.
- The final intial and updated ensemble is used when comparing with
  reference values for the field. One selected realization is used
  when comparing ERT result with reference result.
- The test depends on test example configurations saved in yml file.
  Parameters in this test example configuration is modified by the test functions.
- Reference case realisations are created previously and are data checked
  into git as ascii grdecl files.
- Any changes in test example configuration may require a separate file for
  the reference file in GRDECL format.
- The reference case when replaced must be visualized ot QC the reference.
"""
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import xtgeo
import yaml
from ert import LibresFacade
from ert.__main__ import ert_parser
from ert.cli import ENSEMBLE_SMOOTHER_MODE
from ert.cli.main import run_cli
from ert.shared.plugins.plugin_manager import ErtPluginContext
from ert.storage import open_storage

from scripts.common_functions import (
    Settings,
    initialize_case,
    example_cases,
)

# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=missing-function-docstring
# pylint: disable=too-many-statements


@pytest.mark.parametrize(
    "new_settings",
    [
        pytest.param(example_cases("A")),
        pytest.param(example_cases("B")),
        pytest.param(example_cases("C")),
        pytest.param(example_cases("A2")),
        pytest.param(example_cases("D")),
        pytest.param(example_cases("E")),
        pytest.param(example_cases("F")),
        pytest.param(example_cases("G")),
    ],
)
# pylint: disable=too-many-branches
def test_that_localization_works_with_different_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, new_settings: Dict[str, Any]
):
    monkeypatch.chdir(tmp_path)
    print(f"Setup test case on tmp_path:  {tmp_path}")
    settings = Settings()
    settings.reset_to_default()
    settings.update(new_settings)

    # Write yml file for the sim_fields.py script
    main_settings_dict = settings.to_dict()
    with open("example_config.yml", "w", encoding="utf-8") as file:
        file.write(
            yaml.safe_dump(main_settings_dict, indent=4, default_flow_style=False)
        )

    (tmp_path / "init_files").mkdir()
    (tmp_path / "reference_files").mkdir()

    # Make available data to generate grid, region parameters and scaling factor
    polygon_file = None
    if settings.case_name == "A2":
        polygon_file = "init_files/polygons.txt"
        shutil.copy(
            Path(__file__).parent / settings.model_size.polygon_file,
            polygon_file,
        )

    if settings.case_name == "E":
        region_polygon_file = "init_files/region_polygons.txt"
        shutil.copy(
            Path(__file__).parent / settings.localisation.region_polygons,
            region_polygon_file,
        )

    if settings.case_name == "F":
        scaling_factor_param_file = "init_files/scaling_factor.grdecl"
        shutil.copy(
            Path(__file__).parent / settings.localisation.scaling_file,
            scaling_factor_param_file,
        )

    if settings.case_name == "G":
        scaling_factor_param_file = "init_files/scaling_factor_rms_origo.grdecl"
        shutil.copy(
            Path(__file__).parent / settings.localisation.scaling_file,
            scaling_factor_param_file,
        )

    # Make ERT template file available
    ert_config_template_file = "init_files/sim_field_template.ert"
    shutil.copy(
        Path(__file__).parent / settings.response.ert_config_template_file,
        ert_config_template_file,
    )

    initialize_case(settings)

    parser = ArgumentParser(prog="test_main")
    if settings.case_name == "A":
        reference_file_initial = "FieldParam_real5_iter0_A_local.grdecl"
        reference_file_updated = "FieldParam_real5_iter1_A_local.grdecl"
        grid_file_name = "GRID_STANDARD.EGRID"
    elif settings.case_name == "B":
        reference_file_initial = "FieldParam_real5_iter0_A_local.grdecl"
        reference_file_updated = "FieldParam_real5_iter1_A_local.grdecl"
        grid_file_name = "GRID_STANDARD.EGRID"
    elif settings.case_name == "C":
        reference_file_initial = "FieldParam_real5_iter0_C_local.grdecl"
        reference_file_updated = "FieldParam_real5_iter1_C_local.grdecl"
        grid_file_name = "GRID_RMS_ORIGO.EGRID"
    elif settings.case_name == "A2":
        reference_file_initial = "FieldParam_real5_iter0_A2_local.grdecl"
        reference_file_updated = "FieldParam_real5_iter1_A2_local.grdecl"
        grid_file_name = "GRID_WITH_ACTNUM.EGRID"
    elif settings.case_name == "D":
        # Scaling factor =1 and one correlation group
        # The result should be identical to not running localisation like
        # case A without localisation
        reference_file_initial = "FieldParam_real5_iter0_D.grdecl"
        reference_file_updated = "FieldParam_real5_iter1_D.grdecl"
        grid_file_name = "GRID_STANDARD.EGRID"
    elif settings.case_name == "E":
        reference_file_initial = "FieldParam_real5_iter0_E_local.grdecl"
        reference_file_updated = "FieldParam_real5_iter1_E_local.grdecl"
        grid_file_name = "GRID_STANDARD.EGRID"
    elif settings.case_name == "F":
        reference_file_initial = "FieldParam_real5_iter0_F_local.grdecl"
        reference_file_updated = "FieldParam_real5_iter1_F_local.grdecl"
        grid_file_name = "GRID_STANDARD.EGRID"
    elif settings.case_name == "G":
        reference_file_initial = "FieldParam_real5_iter0_G_local.grdecl"
        reference_file_updated = "FieldParam_real5_iter1_G_local.grdecl"
        grid_file_name = "GRID_RMS_ORIGO.EGRID"

    reference_path_initial = "reference_files/" + reference_file_initial
    reference_path_updated = "reference_files/" + reference_file_updated
    shutil.copy(
        Path(__file__).parent / "init_files" / reference_file_initial,
        reference_path_initial,
    )
    shutil.copy(
        Path(__file__).parent / "init_files" / reference_file_updated,
        reference_path_updated,
    )
    shutil.copy(Path(__file__).parent / "localisation.wf", "localisation.wf")
    Path("scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(
        Path(__file__).parent / "scripts" / "FM_SIM_FIELD", "scripts/FM_SIM_FIELD"
    )
    shutil.copy(
        Path(__file__).parent / "scripts" / "sim_fields.py", "scripts/sim_fields.py"
    )
    shutil.copy(
        Path(__file__).parent / "scripts" / "common_functions.py",
        "scripts/common_functions.py",
    )
    ert_config_file = settings.response.ert_config_file
    parsed = ert_parser(
        parser,
        [
            ENSEMBLE_SMOOTHER_MODE,
            "--current-case",
            "es_prior",
            "--target-case",
            "es_posterior",
            ert_config_file,
        ],
    )
    with ErtPluginContext() as _:
        run_cli(parsed)

        facade = LibresFacade.from_config_file(ert_config_file)

        grid_file = xtgeo.grid_from_file(grid_file_name, fformat="egrid")
        es_prior_expected = xtgeo.gridproperty_from_file(
            reference_path_initial,
            fformat="grdecl",
            name="FIELDPAR",
            grid=grid_file,
        )
        es_prior_expected.mask_undef()
        es_posterior_expected = xtgeo.gridproperty_from_file(
            reference_path_updated,
            fformat="grdecl",
            name="FIELDPAR",
            grid=grid_file,
        )
        es_posterior_expected.mask_undef()
        dims = es_prior_expected.dimensions
        with open_storage(facade.enspath) as storage:
            realization = 5
            es_prior = storage.get_ensemble_by_name("es_prior")
            es_prior_xdata = es_prior.load_parameters("FIELDPAR").sel(
                realizations=realization
            )
            es_prior_values_from_storage = es_prior_xdata["values"].values

            es_posterior = storage.get_ensemble_by_name("es_posterior")
            es_posterior_xdata = es_posterior.load_parameters("FIELDPAR").sel(
                realizations=realization
            )
            es_posterior_values_from_storage = es_posterior_xdata["values"].values

        # Write to file for easier QC of differences
        es_prior_from_storage = xtgeo.GridProperty(
            ncol=dims[0],
            nrow=dims[1],
            nlay=dims[2],
            name="prior_storage",
            values=es_prior_values_from_storage,
        )
        es_prior_from_storage.mask_undef()
        es_prior_from_storage.to_file("prior_storage.roff", fformat="roff")

        # Write to file for easier QC of differences
        es_posterior_from_storage = xtgeo.GridProperty(
            ncol=dims[0],
            nrow=dims[1],
            nlay=dims[2],
            name="posterior_storage",
            values=es_posterior_values_from_storage,
        )
        es_posterior_from_storage.mask_undef()
        es_posterior_from_storage.to_file("posterior_storage.roff", fformat="roff")

        # Check with reference
        assert np.allclose(
            es_prior_expected.values3d,
            np.round(es_prior_values_from_storage, 4),
            atol=1e-4,
            equal_nan=True,
        )

        assert np.allclose(
            es_posterior_expected.values3d,
            np.round(es_posterior_values_from_storage, 4),
            atol=1e-4,
            equal_nan=True,
        )
