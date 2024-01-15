#!/usr/bin/env python
"""
Script converting a grid parameter file from ROFF to GRDECL
"""

import sys
from pathlib import PurePath

import xtgeo


def main(argv):
    """
    Convert a grid parameter file from ROFF to GRDECL
    and can also take as input a GRDECL file and write
    to another GRDECL file with fixed format
    """
    input_file = argv[1]
    output_file = argv[2]
    grid_file = argv[3]
    param_name = "FIELDPAR"
    grid_obj = xtgeo.grid_from_file(grid_file, fformat="egrid")
    # Check extension to find input format
    suffix = PurePath(input_file).suffix
    if suffix.upper() == ".ROFF":
        print(f"Read file: {input_file} in ROFF format")
        field_obj = xtgeo.gridproperty_from_file(
            input_file, fformat="roff", name=param_name, grid=grid_obj
        )
        field_obj.mask_undef()
        print(f"Write file: {output_file} in GRDECL format")
        field_obj.to_file(output_file, fformat="grdecl", fmt="%20.6f")
    elif suffix.upper() == ".GRDECL":
        # Ensure the grdecl file use a fixed format for the values
        # to more easily compare it with a reference
        print(f"Read file: {input_file} in GRDECL format")
        field_obj = xtgeo.gridproperty_from_file(
            input_file, fformat="grdecl", name=param_name, grid=grid_obj
        )
        field_obj.mask_undef()
        print(f"Write file: {output_file} in GRDECL format")
        field_obj.to_file(output_file, fformat="grdecl", fmt="%20.6f")


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 4:
        print(
            "Usage: roff_to_grdecl.py  <input_roff_file or input grdecl_file> "
            "<output_grdecl_file>  <grid_file EGRID format> "
        )
    else:
        main(args)
