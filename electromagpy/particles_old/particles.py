# distutils: language=c++
# Particles that respond to electromagnetic fields
import cython
from cython import double
import numpy as np
from cython.cimports.libcpp.vector import vector

from collections.abc import Sequence


@cython.cclass
class _Particle:
    """
    Charged particle that responds to electromagnetic fields
    """

    def __init__(
            self, q: float, m: float,
            r: Sequence = (0.0, 0.0, 0.0),
            v: Sequence = (0.0, 0.0, 0.0)
    ):

        self._q = q
        self._m = m

        self._r = [r[i] for i in range(3)]
        self._v = [v[i] for i in range(3)]

        self._PE = float('nan')
        self._F = [float('nan'), float('nan'), float('nan')]

    @property
    def q(self):
        return self._q

    @property
    def m(self):
        return self._m

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, r):
        for i in range(3):
            self._r[i] = r[i]

        self._PE = float('nan')
        self._F = [float('nan'), float('nan'), float('nan')]

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, v):

        for i in range(3):
            self._v[i] = v[i]

        self._PE = float('nan')
        self._F = [float('nan'), float('nan'), float('nan')]

    @property
    def KE(self) -> float:
        """Kinetic energy"""
        return 0.5 * self.m * (self.v[0] ** 2 + self.v[1] ** 2 + self.v[2] ** 2)

    @property
    def L(self) -> np.ndarray:
        """Angular momentum vector"""
        return self.m * np.linalg.cross(self.r, self.v)

    @property
    def PE(self) -> float:
        """Potential energy"""
        return self._PE

    @PE.setter
    def PE(self, PE):
        """Should only be set by interacting field"""
        self._PE = PE

    @property
    def F(self):
        """Force vector"""
        return self._F

    @F.setter
    def F(self, F):
        """Should only be set by interacting field"""
        self._F = F


class Particle(_Particle):
    """Python wrapper for _Particle C class"""
    pass
