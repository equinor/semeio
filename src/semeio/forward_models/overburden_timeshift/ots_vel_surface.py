import numpy as np
import segyio
from scipy.interpolate import CloughTocher2DInterpolator
from segyio import TraceField


class OTSVelSurface:
    def __init__(self, res_surface, vcube):
        """
        Create a surface where the timeshift can be calculated.

        Read a reservoir
        """

        self._x = None
        self._y = None
        self._z = None

        self._nx = None
        self._ny = None

        self._z3d = None
        self._dt = None

        self._map_reservoir_surface_to_velocity(res_surface, vcube)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def nx(self):
        return self._nx

    @property
    def ny(self):
        return self._ny

    @property
    def z3d(self):
        return self._z3d

    @property
    def dt(self):
        return self._dt

    def __len__(self):
        return len(self.x)

    def __str__(self):
        s = "x: " + str(self.x)
        s += "\ny: " + str(self.y)
        s += "\nz: " + str(self.z)
        s += "\ndt=" + str(self.dt)
        s += "\nz3d=" + str(self.z3d)
        return s

    def _read_velocity(self, vcube, cell_corners):
        """
        Read velocity from segy file. Upscale.
        :param vcube:
        :param cell_corners:
        :return:
        """

        with segyio.open(vcube) as f:
            x = np.empty(f.tracecount, dtype=np.float64)
            y = np.empty(f.tracecount, dtype=np.float64)
            dt = segyio.dt(f) / 1e6
            nt = len(f.samples)

            for i, h in enumerate(f.header):
                scalar = h[TraceField.SourceGroupScalar]
                if scalar < 0:
                    scalar = -1.0 / scalar
                x[i] = h[TraceField.CDP_X] * scalar
                y[i] = h[TraceField.CDP_Y] * scalar

            ny = len(f.xlines)
            nx = len(f.ilines)

            # memory issue might happen if volume becomes too large
            traces = f.trace.raw[:]

            if f.sorting == segyio.TraceSortingFormat.INLINE_SORTING:
                x = x.reshape(ny, nx)
                y = y.reshape(ny, nx)
                traces = traces.reshape(ny, nx, nt)
                x = np.transpose(x)
                y = np.transpose(y)
                traces = np.transpose(traces, (1, 0, 2))
            elif f.sorting == segyio.TraceSortingFormat.CROSSLINE_SORTING:
                x = x.reshape(nx, ny)
                y = y.reshape(nx, ny)
                traces = traces.reshape(nx, ny, nt)
            else:
                raise RuntimeError

        x, y, traces, nt, dt = self._upscale_velocity(
            cell_corners, x, y, traces, nt, dt
        )

        return x, y, traces, nt, dt

    @staticmethod
    def _upscaling_size_stepping(res_corners, axis, vel_axis):
        """
        Upscales axis of velocity.

        :param res_corners: All corner points of reservoir model, along @axis
        :param axis: Gives which axis is being upscaled. 0 for x, 1 for y
        :param vel_axis: axis of model, x or y given by @axis
        :return:
        """
        res_n = res_corners.shape[axis]
        # average extent of the grid in axis direction divided by resolution
        # gives us average extent of one voxel
        size = np.mean(np.max(res_corners, axis) - np.min(res_corners, axis)) / res_n
        # the number of grid voxels that can be fit into the segy volume
        # defined by CDP_X and CDP_Y
        nn = np.ceil(
            np.mean(np.max(vel_axis, axis) - np.min(vel_axis, axis)) / size
        ).astype(int)

        n = vel_axis.shape[axis]
        ups = np.floor((n - 1) / nn).astype(int)
        # always only upscaling
        if ups < 1:
            ups = 1
            nn = n

        return nn, ups

    def _upscale_velocity(self, res_corners, x_vel, y_vel, traces, nt, dt):
        """resample to a new grid size based on the grid size"""

        nxx, upsx = self._upscaling_size_stepping(res_corners[:, :, 0], 0, x_vel)

        nyy, upsy = self._upscaling_size_stepping(res_corners[:, :, 1], 1, y_vel)

        ntt = int(np.floor(nt / (16e-3 / dt)))
        upst = int(np.floor((nt - 1) / ntt))
        if upst == 0:
            upst = 1
            ntt = nt
        dt *= upst
        # skip positions if needed
        x = x_vel[0 : (nxx - 1) * upsx + 1 : upsx, 0 : (nyy - 1) * upsy + 1 : upsy]
        y = y_vel[0 : (nxx - 1) * upsx + 1 : upsx, 0 : (nyy - 1) * upsy + 1 : upsy]
        # upscale traces if needed, i.e., ntt > nt
        # skip samples if needs, ups >= 2
        traces_upscaled = traces[
            0 : (nxx - 1) * upsx + 1 : upsx,
            0 : (nyy - 1) * upsy + 1 : upsy,
            0 : (ntt - 1) * upst + 1 : upst,
        ]

        return x, y, traces_upscaled, nt, dt

    def _map_reservoir_surface_to_velocity(self, res_surface, vcube):
        """
        Interpolates reservoir top surface to velocity grid
        """
        # downsample segy if segy resultion of CDP and
        # sample rate is higher than of the Eclgrid
        x, y, traces, _, self._dt = self._read_velocity(vcube, res_surface.cell_corners)
        # this is some clever integration of traces using
        # averaging between two samples?
        vel_t_int = np.zeros_like(traces)
        vel_t_int[:, :, 1:] = (traces[:, :, 0:-1] + traces[:, :, 1:]) / 2
        # cumulative sum of each trace and reshape to 2D
        # array of size (num_trace, num_samples)
        self._z3d = np.cumsum(vel_t_int * self._dt / 2, 2).reshape(
            -1, vel_t_int.shape[-1]
        )
        # this creates interpolation functions over existing surface / grid
        ip = CloughTocher2DInterpolator((res_surface.x, res_surface.y), res_surface.z)
        # interpolate over our vel field given by pos x and y
        z = ip(x, y)
        # z gives a new depth for segy CDP pos given the grid interface surface

        # So far we have downsampled the segy file to correspond
        # with the resolution of the grid.
        # The x,y values corresponds to the remaining segy positions.
        # furthermore the z values are the interpolated values that
        # match the top layer of the active cells in the grid file.
        self._nx, self._ny = x.shape
        self._x = x.reshape(-1)
        self._y = y.reshape(-1)
        self._z = z.reshape(-1)
