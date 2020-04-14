#!/usr/bin/env python

import sys
from semeio.jobs.csv_export1.design_matrix_reader import DesignMatrixReader
from semeio.jobs.csv_export1.ert_csv_ensemble import ErtCSVEnsemble


def export(*args):
    if len(args) < 2:
        print(
            "Usage: Arguments to csv_export1.py are: path_file export_file"
            " DesignMatrix:design_matrix_file(optional) DateInterval:date_interval(optional)"
            "summary_key...summary_key (optional)"
        )
    elif len(args) >= 2:
        ens = ErtCSVEnsemble(args[0])
        ens_export_file = args[1]
        date_interval = None
        ens.loadParameters()

        if len(args) >= 3:
            index = 2

            if "DesignMatrix:" in args[index]:
                design_matrix_arg = args[index].split(":")
                design_matrix_file = design_matrix_arg[1]
                design_matrix = DesignMatrixReader.loadDesignMatrix(design_matrix_file)
                ens.loadDesignMatrix(design_matrix)
                index += 1

            if "DateInterval:" in args[index]:
                date_interval_arg = args[index].split(":")
                date_interval = date_interval_arg[1]
                index += 1

            for key in args[index:]:
                ens.addSummaryPattern(key)

        ens.dump(ens_export_file, date_interval)


if __name__ == "__main__":
    export(*sys.argv[1:])
