import pandas as pd
from fmu import ensemble


def csv_exporter(runpathfile, time_index, outputfile, column_keys=None):
    """Export CSV data (summary and parameters) from an EnsembleSet

    The EnsembleSet is described by a runpathfile which must exists
    and point to realizations"""
    ensemble_set = ensemble.EnsembleSet(
        name="ERT EnsembleSet for CSV_EXPORT2", runpathfile=runpathfile
    )
    summary = ensemble_set.load_smry(time_index=time_index, column_keys=column_keys)
    parameters = ensemble_set.parameters
    summary_parameters = pd.merge(summary, parameters)
    summary_parameters.to_csv(outputfile, index=False)
