"""
Import field parameters into RMS (Must be included as python job
in RMS workflow and edited to fit your scratch directory)
Variable project is defined when running within RMS,
but not outside since it refers to Roxar API.
"""

# pylint: disable=import-error, undefined-variable
from import_fields_to_rms import import_from_scratch_directory

PRJ = project  # noqa: 821
GRID_MODEL_NAME = "GRID"
FIELD_NAMES = [
    "FieldParam",
]
FILE_FORMAT = "ROFF"
ITERATION = 3

# The different cases
CASE_NAME = "original"
SCRATCH = "/scratch/fmu/olia/sim_field/"
import_from_scratch_directory(
    PRJ, GRID_MODEL_NAME, FIELD_NAMES, CASE_NAME, SCRATCH, FILE_FORMAT, ITERATION
)
CASE_NAME = "local"
SCRATCH = "/scratch/fmu/olia/sim_field_local/"
import_from_scratch_directory(
    PRJ, GRID_MODEL_NAME, FIELD_NAMES, CASE_NAME, SCRATCH, FILE_FORMAT, ITERATION
)
