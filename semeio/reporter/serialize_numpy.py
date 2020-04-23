import numpy as np
import pandas as pd


def serialize_numpy(key, val):
    if not isinstance(val, np.ndarray):
        return False

    pd.DataFrame(val).to_csv("{}.csv".format(key), header=False, index=False)
    return True
