import sys

from ert import (
    ForwardModelStepDocumentation,
    ForwardModelStepJSON,
    ForwardModelStepPlugin,
    ForwardModelStepValidationError,
)

from .scripts.design2params import description as design2params_description
from .scripts.design_kw import description as design_kw_description
from .scripts.design_kw import design_kw as DesignKWScript
from .scripts.fm_pyscal import description as pyscal_description
from .scripts.gendata_rft import description as gendata_rft_description
from .scripts.overburden_timeshift import description as ots_description
from .scripts.replace_string import description as replace_string_description


class Design2Params(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="DESIGN2PARAMS",
            command=[
                "design2params",
                "<IENS>",
                "<xls_filename>",
                "<designsheet>",
                "<defaultssheet>",
            ],
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            category="utility.templating",
            source_package="semeio",
            source_function_name="Design2Params",
            description=design2params_description,
        )


class DesignKW(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="DESIGN_KW",
            command=[
                "design_kw",
                "<template_file>",
                "<result_file>",
                "--validate=<VALIDATE>",
            ],
            default_mapping={"<VALIDATE>": "false"},
        )

    def validate_pre_experiment(self, fm_step_json: ForwardModelStepJSON) -> None:
        if self.private_args.get("<VALIDATE>", "").lower() == "true":
            try:
                template_file_name = self.private_args["<template_file>"]
            except KeyError:
                raise ForwardModelStepValidationError(
                    "Argument list is missing <template_file>"
                ) from None
            DesignKWScript.validate_configuration(template_file_name=template_file_name)

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            category="utility.templating",
            source_package="semeio",
            source_function_name="DesignKW",
            description=design_kw_description,
        )


class GenDataRFT(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="GENDATA_RFT",
            command=[
                "gendata_rft",
                "--eclbase",
                "<ECLBASE>",
                "--trajectory_path",
                "<PATH_TO_TRAJECTORY_FILES>",
                "--well_and_time_file",
                "<WELL_AND_TIME_FILE>",
                "--zonemap",
                "<ZONEMAP>",
                "--csvfile",  # output
                "<CSVFILE>",
                "--outputdirectory",
                "<OUTPUTDIRECTORY>",
            ],
            default_mapping={
                "<ZONEMAP>": "ZONEMAP_NOT_PROVIDED",
                "<CSVFILE>": "gendata_rft.csv",
                "<OUTPUTDIRECTORY>": ".",
            },
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            category="utility.transformation",
            source_package="semeio",
            source_function_name="GenDataRFT",
            examples="""
Setup a file with well-names and associated date of RFT data in a file called
e.g. ``well_date_rft.txt``::

    -- well YYYY-MM-DD report_step
    A-1     2000-02-01 0

A directory with trajectory files must be prepared, which must contain one
file for each well mentioned in the file above. A file in this directory
could look like::

    -- utmx    utmy      depth_MD depth_TVD  zone
    462608.57 5934210.96 1674.44  1624.38    Upper -- cell 29 28 2

(add more lines for more points).

A zonemap-file is a text-file with k-index and zone-name pr line, e.g a file
named ``layer_zone_table.txt``::

    1 Upper
    2 Upper
    3 Lower
    4 Lower

In the ert config, after running the Eclipse (or similar) forward model, add::

    DEFINE RFT_INPUT <CONFIG_PATH>/../input/observations/rft
    FORWARD_MODEL MAKE_DIRECTORY(<DIRECTORY>=gendata_rft)
    FORWARD_MODEL GENDATA_RFT(<PATH_TO_TRAJECTORY_FILES>=<RFT_INPUT>/rft/, \
        <WELL_AND_TIME_FILE>=<RFT_INPUT>/well_date_rft.txt, \
        <ZONEMAP>=<RFT_INPUT>/layer_zone_table.txt, \
        <OUTPUTDIRECTORY>=gendata_rft)

For assisted history matching, add ``GEN_DATA`` statements to the ert config::

    GEN_DATA A-1 RESULT_FILE:gendata_rft/RFT_A-1_%d REPORT_STEPS:0

**Migrating to RFT_OBSERVATION**

ert can read RFT files directly using ``RFT_OBSERVATION`` in the observation
configuration. This replaces ``GENDATA_RFT`` forward model, the RFT
``GEN_DATA`` statements, the trajectory files, the well-and-time file and the
corresponding ``GENERAL_OBSERVATION`` entries.

The following shows how a setup observing pressure in well ``PROD`` is
converted.

*Before* ert config::

    FORWARD_MODEL GENDATA_RFT( \\
        <PATH_TO_TRAJECTORY_FILES>=<CONFIG_PATH>/resources/trajectories, \\
        <WELL_AND_TIME_FILE>=<CONFIG_PATH>/resources/well_date_rft.txt, \\
        <ZONEMAP>=<CONFIG_PATH>/resources/layer_zone_table.txt \\
    )
    GEN_DATA PROD_RFT RESULT_FILE:RFT_PROD_%d REPORT_STEPS:1
    OBS_CONFIG obs.txt

*Before* ``obs.txt``::

    GENERAL_OBSERVATION PROD_PRESSURE
    {
        DATA = PROD_RFT;
        OBS_FILE = prod_pressure.txt;
        RESTART = 1;
    };

*Before* ``well_date_rft.txt``::

    -- well date       report_step
    PROD 2015-01-01 1

*Before* ``trajectories/PROD.txt``::

    -- utmx utmy depth_MD depth_TVD zone
    9500.0 9500.0 8335.0 8335.0 Zone1
    9500.0 9500.0 8360.0 8360.0 Zone1
    9500.0 9500.0 8400.0 8400.0 Zone1

*Before* ``prod_pressure.txt``::

    4380.123 100.0
    4390.123 100.0
    4395.5   100.0

*After*, ert config::

    ZONEMAP <CONFIG_PATH>/resources/layer_zone_table.txt
    OBS_CONFIG obs.txt

*After* ``obs.txt``::

    RFT_OBSERVATION PROD_PRESSURE_1 {
        WELL=PROD;
        DATE=2015-01-01;
        PROPERTY=PRESSURE;
        VALUE=4380.123;
        ERROR=100.0;
        EAST=9500.0;
        NORTH=9500.0;
        TVD=8335.0;
        ZONE=Zone1;
    };

Repeat the block for each point (``TVD`` 8360.0 and 8400.0 with values
4390.123 and 4395.5), or use one CSV file with all observations::

    RFT_OBSERVATION PROD_PRESSURE {
        CSV=<CONFIG_PATH>/resources/prod_pressure.csv;
    };

where ``prod_pressure.csv`` contains::

    "WELL_NAME", "DATE", "ZONE", "PRESSURE", "ERROR", "TVD", "NORTH", "EAST"
    "PROD", "2015-01-01", "Zone1", 4380.123, 100.0, 8335.0, 9500.0, 9500.0
    "PROD", "2015-01-01", "Zone1", 4390.123, 100.0, 8360.0, 9500.0, 9500.0
    "PROD", "2015-01-01", "Zone1", 4395.5, 100.0, 8400.0, 9500.0, 9500.0

The ``layer_zone_table.txt`` file is unchanged. For other properties, use
``PROPERTY=SWAT`` (and a ``SWAT`` column in the CSV), etc.

The CSV can be created with the ``create_rft_ertobs`` job from fmu.tools.rms
(run inside RMS), which writes ``rft_ertobs.csv`` to its ``exportdir``. The
``.txt``, ``.obs`` and well-date files it also writes are only needed for
``GENDATA_RFT``. See https://equinor.github.io/fmu-tools/create_rft_ertobs.html

For visualization, the ``EXPORT_RFT`` workflow job in ert replaces the CSV from
``GENDATA_RFT`` combined with ``merge_rft_ertobs``.

Differences to be aware of: there are no ``-1`` padded values and no report
steps, and observations that cannot be matched to an RFT value are deactivated.
Observations whose location is outside the declared zone are deactivated with
a warning. See the ert documentation for ``RFT_OBSERVATION`` and ``ZONEMAP``.

""",
            description=gendata_rft_description,
        )


class OTS(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="OTS",
            command=["overburden_timeshift", "-c", "<CONFIG>"],
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            category="modelling.surface",
            source_package="semeio",
            source_function_name="OTS",
            description=ots_description,
        )


class Pyscal(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="PYSCAL",
            command=[
                "fm_pyscal",
                "<PARAMETER_FILE>",
                "<RESULT_FILE>",
                "<SHEET_NAME>",
                "<INT_PARAM_WO_NAME>",
                "<INT_PARAM_GO_NAME>",
                "<SLGOF>",
                "<FAMILY>",
                "<DELTA_S>",
            ],
            default_mapping={
                "<RESULT_FILE>": "relperm.inc",
                "<SHEET_NAME>": "__NONE__",
                "<INT_PARAM_WO_NAME>": "__NONE__",
                "<INT_PARAM_GO_NAME>": "__NONE__",
                "<SLGOF>": "SGOF",
                "<FAMILY>": "1",
                "<DELTA_S>": "__NONE__",
            },
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            category="modelling.reservoir",
            source_package="semeio",
            source_function_name="Pyscal",
            description=pyscal_description,
            examples="""
.. code-block:: none

  FORWARD_MODEL PYSCAL(<PARAMETER_FILE>=scalinput.xlsx, \
    <RESULT_FILE>=eclipse/include/props/relperm.inc, \
    <SHEETNAME>=alternativerecommendation)
  FORWARD_MODEL PYSCAL(<PARAMETER_FILE>=scalinput.xlsx, \
    <RESULT_FILE>=eclipse/include/props/relperm.inc, \
    <INT_PARAM_WO_NAME>=RELPERM_INTERP)
  FORWARD_MODEL PYSCAL(<PARAMETER_FILE>=scalinput.xlsx, \
    <RESULT_FILE>=eclipse/include/props/relperm.inc, \
    <INT_PARAM_WO_NAME>=RELPERM_INTERP_WO, \
    <INT_PARAM_GO_NAME>=RELPERM_INTERP_GO)
  FORWARD_MODEL PYSCAL(<PARAMETER_FILE>=scalinput.xlsx, \
    <RESULT_FILE>=eclipse/include/props/relperm.inc, \
    <FAMILY>=2) -- for Eclipse family 2 output
  FORWARD_MODEL PYSCAL(<PARAMETER_FILE>=scalinput.xlsx, \
    <RESULT_FILE>=eclipse/include/props/relperm.inc, \
    <DELTA_S>=0.0025) -- for higher saturation step detail

""",
        )


class InsertNoSim(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="INSERT_NOSIM",
            command=[
                "sed",
                "-i",
                "",
                r"s/^RUNSPEC.*/RUNSPEC\nNOSIM/",
                "<ECLBASE>.DATA",
            ]
            if sys.platform == "darwin"
            else [
                "sed",
                "-i",
                r"s/^RUNSPEC.*/RUNSPEC\nNOSIM/",
                "<ECLBASE>.DATA",
            ],
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            source_package="semeio",
            source_function_name="InsertNoSim",
            description="""
Inserts a NOSIM for every RUNSPEC occurrence in the file
""",
        )


class RemoveNoSim(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="REMOVE_NOSIM",
            command=["sed", "-i", "", "/^NOSIM/d", "<ECLBASE>.DATA"]
            if sys.platform == "darwin"
            else ["sed", "-i", "/^NOSIM/d", "<ECLBASE>.DATA"],
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            source_package="semeio",
            source_function_name="RemoveNoSim",
            description="Remove all NOSIM lines from <ECLBASE>.DATA file",
        )


class ReplaceString(ForwardModelStepPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="REPLACE_STRING",
            command=["replace_string", "-o", "<FROM>", "-n", "<TO>", "-f", "<FILE>"],
        )

    @staticmethod
    def documentation() -> ForwardModelStepDocumentation | None:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            source_package="semeio",
            source_function_name="ReplaceString",
            description=replace_string_description,
            examples="""
            | REPLACE_STRING(<FROM>=hello,<TO>=world,<FILE>=some_file.txt
            """,
        )
