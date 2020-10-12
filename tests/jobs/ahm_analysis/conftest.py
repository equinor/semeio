import pytest
import cwrap

from ecl import EclDataType
from ecl.eclfile import EclKW


@pytest.fixture()
def grid_prop():
    def wrapper(prop_name, value, grid_size, fname):
        prop = EclKW(prop_name, grid_size, EclDataType.ECL_FLOAT)
        prop.assign(value)
        with cwrap.open(fname, "w") as f:
            prop.write_grdecl(f)

    return wrapper
