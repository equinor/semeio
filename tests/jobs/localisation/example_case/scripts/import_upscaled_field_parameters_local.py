"""
Import field parameters into RMS (Must be included as python job
in RMS workflow and edited to fit your scratch directory)
Variable project is defined when running within RMS,
but not outside since it refers to Roxar API
"""
# pylint: disable=import-error, undefined-variable
from import_fields_to_rms import import_from_scratch_directory

SCRATCH = "/scratch/fmu/olia/sim_field_local/"
CASE_NAME = "local"

PRJ = project  # noqa: 821
GRID_MODEL_NAME = "UpscaleGrid"
FIELD_NAMES = [
    "Upscaled",
]
ITERATION = 3
import_from_scratch_directory(
    PRJ, GRID_MODEL_NAME, FIELD_NAMES, CASE_NAME, SCRATCH, ITERATION
)
