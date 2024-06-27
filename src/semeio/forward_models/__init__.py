from typing import Optional

from ert import (
    ForwardModelStepDocumentation,
    ForwardModelStepJSON,
    ForwardModelStepPlugin,
)

from .scripts.design2params import description as design2params_description
from .scripts.design_kw import description as design_kw_description


class Design2Params(ForwardModelStepPlugin):
    def __init__(self):
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

    def validate_pre_realization_run(
        self, fm_step_json: ForwardModelStepJSON
    ) -> ForwardModelStepJSON:
        return fm_step_json

    def validate_pre_experiment(self, fm_step_json: ForwardModelStepJSON) -> None:
        return fm_step_json

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        return ForwardModelStepDocumentation(
            category="utility.templating",
            source_package="semeio",
            source_function_name="Design2Params",
            description=design2params_description,
        )


class DesignKW(ForwardModelStepPlugin):
    def __init__(self):
        super().__init__(
            name="DESIGN_KW",
            command=["design_kw", "<template_file>", "<result_file>"],
        )

    def validate_pre_realization_run(
        self, fm_step_json: ForwardModelStepJSON
    ) -> ForwardModelStepJSON:
        return fm_step_json

    def validate_pre_experiment(self, fm_step_json: ForwardModelStepJSON) -> None:
        pass

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        return ForwardModelStepDocumentation(
            category="utility.templating",
            source_package="semeio",
            source_function_name="DesignKW",
            description=design_kw_description,
        )


from .scripts.gendata_rft import description as gendata_rft_description


class GenDataRFT(ForwardModelStepPlugin):
    def __init__(self):
        super().__init__(
            name="GENDATA_RFT",
            command=[
                "gendata_rft",
                "-e",
                "<ECL_BASE>",
                "-t",
                "<PATH_TO_TRAJECTORY_FILES>",
                "-w",
                "<WELL_AND_TIME_FILE>",
                "-z",
                "<ZONEMAP>",
                "-c",
                "<CSVFILE>",
                "-o",
                "<OUTPUTDIRECTORY>",
            ],
            default_mapping={
                "<ZONEMAP>": "ZONEMAP_NOT_PROVIDED",
                "<CSVFILE>": "gendata_rft.csv",
                "<OUTPUTDIRECTORY>": ".",
            },
        )

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
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

In the ert config, after running the Eclipse (or similiar) forward model, add::

    DEFINE RFT_INPUT <CONFIG_PATH>/../input/observations/rft
    FORWARD_MODEL MAKE_DIRECTORY(<DIRECTORY>=gendata_rft)
    FORWARD_MODEL GENDATA_RFT(<PATH_TO_TRAJECTORY_FILES>=<RFT_INPUT>/rft/, <WELL_AND_TIME_FILE>=<RFT_INPUT>/well_date_rft.txt, <ZONEMAP>=<RFT_INPUT>/layer_zone_table.txt, <OUTPUTDIRECTORY>=gendata_rft)

For assisted history matching, add ``GEN_DATA`` statements to the ert config::

    GEN_DATA A-1 RESULT_FILE:gendata_rft/RFT_A-1_%d INPUT_FORMAT:ASCII REPORT_STEPS:0

""",
            description=gendata_rft_description,
        )


from .scripts.overburden_timeshift import description as ots_description


class OTS(ForwardModelStepPlugin):
    def __init__(self):
        super().__init__(
            name="OTS",
            command=["overburden_timeshift", "-c", "<CONFIG>"],
        )

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        return ForwardModelStepDocumentation(
            category="modelling.surface",
            source_package="semeio",
            source_function_name="OTS",
            description=ots_description,
        )


from .scripts.fm_pyscal import description as pyscal_description


class Pyscal(ForwardModelStepPlugin):
    def __init__(self):
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
            ],
            default_mapping={
                "<RESULT_FILE>": "relperm.inc",
                "<SHEET_NAME>": "__NONE__",
                "<INT_PARAM_WO_NAME>": "__NONE__",
                "<INT_PARAM_GO_NAME>": "__NONE__",
                "<SLGOF>": "SGOF",
                "<FAMILY>": "1",
            },
        )

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        return ForwardModelStepDocumentation(
            category="modelling.reservoir",
            source_package="semeio",
            source_function_name="Pyscal",
            description=pyscal_description,
            examples="""
.. code-block:: none

  FORWARD_MODEL PYSCAL(<PARAMETER_FILE>=scalinput.xlsx, <RESULT_FILE>=eclipse/include/props/relperm.inc, <SHEETNAME>=alternativerecommendation)
  FORWARD_MODEL PYSCAL(<PARAMETER_FILE>=scalinput.xlsx, <RESULT_FILE>=eclipse/include/props/relperm.inc, <INT_PARAM_WO_NAME>=RELPERM_INTERP)
  FORWARD_MODEL PYSCAL(<PARAMETER_FILE>=scalinput.xlsx, <RESULT_FILE>=eclipse/include/props/relperm.inc, <INT_PARAM_WO_NAME>=RELPERM_INTERP_WO, <INT_PARAM_GO_NAME>=RELPERM_INTERP_GO)
  FORWARD_MODEL PYSCAL(<PARAMETER_FILE>=scalinput.xlsx, <RESULT_FILE>=eclipse/include/props/relperm.inc, <FAMILY>=2) -- for Eclipse family 2 output

""",
        )


class InsertNoSim(ForwardModelStepPlugin):
    def __init__(self):
        super().__init__(
            name="INSERT_NOSIM",
            command=[
                "sed",
                "-i",
                "s/^RUNSPEC.*/RUNSPEC\\\\nNOSIM/",
                "<ECLBASE>.DATA",
            ],
        )

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            source_package="semeio",
            source_function_name="InsertNoSim",
            description="""
Inserts a NOSIM for every RUNSPEC occurrence in the file
""",
        )


class RemoveNoSim(ForwardModelStepPlugin):
    def __init__(self):
        super().__init__(
            name="REMOVE_NOSIM",
            command=["sed", "-i", "/^NOSIM/d", "<ECLBASE>.DATA"],
        )

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            source_package="semeio",
            source_function_name="RemoveNoSim",
            description="Remove all NOSIM lines from <ECLBASE>.DATA file",
        )


from .scripts.replace_string import description as replace_string_description


class ReplaceString(ForwardModelStepPlugin):
    def __init__(self):
        super().__init__(
            name="REPLACE_STRING",
            command=["replace_string", "-o", "<FROM>", "-n", "<TO>", "-f", "<FILE>"],
        )

    @staticmethod
    def documentation() -> Optional[ForwardModelStepDocumentation]:
        return ForwardModelStepDocumentation(
            category="utility.file_system",
            source_package="semeio",
            source_function_name="ReplaceString",
            description=replace_string_description,
            examples="""
            | REPLACE_STRING(<FROM>=hello,<TO>=world,<FILE>=some_file.txt
            """,
        )


__all__ = ["Design2Params"]
