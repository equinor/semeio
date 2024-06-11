import numpy as np


class OTSResSurface:
    def __init__(self, grid, above=0):
        """
        Create a surface from a reservoir grid.

        The surface will be located given meters above most shallow active cell

        :param grid: Reservoir grid
        :param above: Scalar: meters above most shallow active cell
        """

        self._grid = grid
        self._x = None
        self._y = None
        self._z = None
        self._nx = None
        self._ny = None

        self._calculate_surface(grid, above)

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

    def __len__(self):
        return len(self.x)

    def __str__(self):
        return f"x: {self.x}\ny: {self.y}\nz: {self.z}"

    @property
    def cell_corners(self):
        return self._get_top_corners()

    def _calculate_surface(self, grid, above):
        # calculate average from top face vecrtices
        # from unstructured grid as an interface between active and inactive cells
        nx, ny, nz = grid.getNX(), grid.getNY(), grid.getNZ()

        x = np.empty(shape=(nx * ny), dtype=np.float32)
        y = np.empty(shape=(nx * ny), dtype=np.float32)
        z = np.empty(shape=(nx * ny), dtype=np.float32)
        for i in range(nx):
            for j in range(ny):
                _k = nz - 1
                for k in range(nz):
                    if grid.active(ijk=(i, j, k)):
                        _k = k
                        break
                ijk = (i, j, _k)

                top = [grid.getCellCorner(c, ijk=ijk) for c in range(0, 4)]
                pos = i * ny + j
                x[pos] = sum(val[0] for val in top) / 4.0
                y[pos] = sum(val[1] for val in top) / 4.0
                z[pos] = sum(val[2] for val in top) / 4.0

        z -= above

        self._x = x
        self._y = y
        self._z = z
        self._nx = nx
        self._ny = ny

    def _get_top_corners(self):
        grid = self._grid
        nx = grid.getNX() + 1
        ny = grid.getNY() + 1
        top_corners = np.empty(shape=(nx, ny, 3), dtype=np.float32)
        for i in range(nx):
            for j in range(ny):
                # getNodePos should mostly return top-right vertex position of the node
                top_corners[i, j] = grid.getNodePos(i, j, 0)

        return top_corners
