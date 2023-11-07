import os

import numpy as np
import segyio
from resdata import ResDataType
from resdata.grid import Grid
from resdata.resfile import FortIO, ResdataKW, openFortIO
from segyio import TraceField


def create_init(grid, case):
    poro = ResdataKW("PORO", grid.getNumActive(), ResDataType.RD_FLOAT)
    porv = ResdataKW("PORV", grid.getGlobalSize(), ResDataType.RD_FLOAT)

    with openFortIO(f"{case}.INIT", mode=FortIO.WRITE_MODE) as file:
        poro.fwrite(file)
        porv.fwrite(file)


def create_restart(grid, case, rporv=None):
    # pylint: disable=invalid-name
    with openFortIO(f"{case}.UNRST", mode=FortIO.WRITE_MODE) as file:
        seq_hdr = ResdataKW("SEQNUM", 1, ResDataType.RD_FLOAT)
        seq_hdr[0] = 1
        p = ResdataKW("PRESSURE", grid.getNumActive(), ResDataType.RD_FLOAT)
        p.assign(1)

        for i, _ in enumerate(p):
            p[i] = 10

        header = ResdataKW("INTEHEAD", 67, ResDataType.RD_INT)
        header[64] = 1
        header[65] = 1
        header[66] = 2000

        seq_hdr.fwrite(file)
        header.fwrite(file)
        p.fwrite(file)

        if rporv:
            rp = ResdataKW("RPORV", grid.getNumActive(), ResDataType.RD_FLOAT)
            for idx, val in enumerate(rporv):
                rp[idx] = val
            rp.fwrite(file)

        seq_hdr[0] = 2
        header[66] = 2010
        for i, _ in enumerate(p):
            p[i] = 20

        seq_hdr.fwrite(file)
        header.fwrite(file)
        p.fwrite(file)

        if rporv:
            rp = ResdataKW("RPORV", grid.getNumActive(), ResDataType.RD_FLOAT)
            for idx, val in enumerate(rporv):
                rp[idx] = val
            rp.fwrite(file)

        seq_hdr[0] = 3
        header[66] = 2011
        for i, _ in enumerate(p):
            p[i] = 25

        seq_hdr.fwrite(file)
        header.fwrite(file)
        p.fwrite(file)

        if rporv:
            rp = ResdataKW("RPORV", grid.getNumActive(), ResDataType.RD_FLOAT)
            for idx, val in enumerate(rporv):
                rp[idx] = val
            rp.fwrite(file)


def create_segy_file(name, spec, trace=None, il=None, xl=None, cdp_x=None, cdp_y=None):
    # pylint: disable=too-many-arguments,invalid-name

    if trace is None:
        trace = np.empty(len(spec.samples), dtype=np.single)
        for i in range(len(spec.samples)):
            trace[i] = 1500 + i

    int_val = [50, 150]
    if il is None:
        il = int_val
    if xl is None:
        xl = int_val
    if cdp_x is None:
        cdp_x = int_val
    if cdp_y is None:
        cdp_y = int_val

    with segyio.create(name, spec) as file:
        scalar = 1
        trno = 0
        for i, i_cdp_x in enumerate(cdp_x):
            for j, j_cdp_y in enumerate(cdp_y):
                file.header[trno] = {
                    TraceField.CDP_Y: j_cdp_y,
                    TraceField.CDP_X: i_cdp_x,
                    TraceField.CROSSLINE_3D: xl[i],
                    TraceField.INLINE_3D: il[j],
                    TraceField.SourceGroupScalar: scalar,
                }
                file.trace[trno] = trace + 0.5 * trno
                trno += 1


def mock_segy(
    rd_grid,
    relative_resolution_scale=(1.0, 1.0, 1.0),
    relative_size_scale=(1.0, 1.0),
    relative_position_shift=(0, 0),
    segy_filename="vel_vol.segy",
):
    # pylint: disable=too-many-locals,invalid-name

    # get the surface top_corners
    nx = rd_grid.get_nx()
    ny = rd_grid.get_ny()
    nz = rd_grid.get_nz()
    top_corners = np.empty(shape=(nx + 1, ny + 1, 3), dtype=np.float32)
    for i in range(nx + 1):
        for j in range(ny + 1):
            top_corners[i, j] = rd_grid.getNodePos(i, j, 0)

    # get the bounding box of the surface
    min_x = np.min(top_corners[:, :, 0])
    max_x = np.max(top_corners[:, :, 0])
    min_y = np.min(top_corners[:, :, 1])
    max_y = np.max(top_corners[:, :, 1])

    # we scale and shift wrt the mid-point of the surface
    mid_point = 0.5 * (max_x + min_x), 0.5 * (max_y + min_y)
    extent_x = max_x - min_x
    extent_y = max_y - min_y
    scale_extent_x = extent_x * relative_size_scale[0]
    scale_extent_y = extent_y * relative_size_scale[1]

    # compute scaled and shifted bounding box
    min_x = mid_point[0] - scale_extent_x / 2 + extent_x * relative_position_shift[0]
    max_x = mid_point[0] + scale_extent_x / 2 + extent_x * relative_position_shift[0]
    min_y = mid_point[1] - scale_extent_y / 2 + extent_y * relative_position_shift[1]
    max_y = mid_point[1] + scale_extent_y / 2 + extent_y * relative_position_shift[1]

    # get the new resolution
    nx, ny, nz = tuple(
        np.multiply(relative_resolution_scale, [nx, ny, nz]).astype(np.int32)
    )

    # traces are currently hardcoded going linearly up 1500-4800 samplez into nz steps
    spec = segyio.spec()
    spec.format = 5
    spec.sorting = 2
    spec.samples = range(0, nz * 4, 4)
    spec.ilines = range(ny)
    spec.xlines = range(nx)
    trace = np.linspace(1500, 4800, nz).astype(np.int32)
    xl = np.linspace(min_x, max_x, nx).astype(np.int32)
    il = np.linspace(min_y, max_y, ny).astype(np.int32)
    cdp_x = np.linspace(min_x, max_x, nx).astype(np.int32)
    cdp_y = np.linspace(min_y, max_y, ny).astype(np.int32)

    # for more complex segy volume ie. traces with might use porosity
    # ie. poro = init.iget_named_kw("PORO")
    create_segy_file(
        segy_filename, spec=spec, trace=trace, il=il, xl=xl, cdp_x=cdp_x, cdp_y=cdp_y
    )


if __name__ == "__main__":
    grid_path = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "../../../test-data/norne")
    )
    grid_file = os.path.join(grid_path, "NORNE_ATW2013.EGRID")
    norne_grid = Grid(grid_file, apply_mapaxes=True)

    mock_segy(
        norne_grid,
        relative_resolution_scale=(1, 1, 1),
        relative_size_scale=(1, 1),
        relative_position_shift=(0, 0),
        segy_filename="vol_vel.segy",
    )
