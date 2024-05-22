from ert import ForwardModelStepJSON, ForwardModelStepPlugin


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


class OTS(ForwardModelStepPlugin):
    def __init__(self):
        super().__init__(
            name="OTS",
            command=["overburden_timeshift", "-c", "<CONFIG>"],
        )


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


class RemoveNoSim(ForwardModelStepPlugin):
    def __init__(self):
        super().__init__(
            name="REMOVE_NOSIM",
            command=["sed", "-i", "/^NOSIM/d", "<ECLBASE>.DATA"],
        )


class ReplaceStringConfig(ForwardModelStepPlugin):
    def __init__(self):
        super().__init__(
            name="REPLACE_STRING",
            command=["replace_string", "-o", "<FROM>", "-n", "<TO>", "-f", "<FILE>"],
        )


__all__ = ["Design2Params"]
