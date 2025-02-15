# distutils: language=c++
# Particles that respond to electromagnetic fields
import cython
from cython import ulong, double, void
import numpy as np
# from cython.cimports import numpy as np
from cython.cimports.electromagpy.fields.field import dot, cross, _Field
from cython.cimports.libcpp.vector import vector
from electromagpy.fields.electrostatic import Vacuum

from collections.abc import Iterable


@cython.cclass
class _Particle:
    """
    Charged particle that responds to electromagnetic fields
    """

    _q: double
    _m: double
    # field: _Field

    # t: double
    _r: vector[double]
    _v: vector[double]

    # # pre-initialized force and acceleration vectors
    # _F: vector[double]
    # _a1: vector[double]  # there are two acceleration vectors used
    # _a2: vector[double]  # by the velocity-Verlet algorithm

    # def __cinit__(self):
    #     self.t = 0.0
    #     self._F = [0.0, 0.0, 0.0]
    #     self._a1 = [0.0, 0.0, 0.0]
    #     self._a2 = [0.0, 0.0, 0.0]

    def __init__(
            self, q: float, m: float,
            # field: _Field = _Field(),
            r: Iterable = (0.0, 0.0, 0.0),
            v: Iterable = (0.0, 0.0, 0.0)
    ):

        self._q = q
        self._m = m

        # self.field = field

        self._r = list(r)
        self._v = list(v)

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

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, v):
        for i in range(3):
            self._v[i] = v[i]

    # @cython.cfunc
    # def eval_F(self) -> void:
    #
    #     self.field.eval_E(self.r, self.t)
    #     self.field.eval_B(self.r, self.t)
    #
    #     self._F[0] = self.q * (self.field._E[0] + cross(0, self.v, self.field._B))
    #     self._F[1] = self.q * (self.field._E[1] + cross(1, self.v, self.field._B))
    #     self._F[2] = self.q * (self.field._E[2] + cross(2, self.v, self.field._B))

    # @cython.ccall
    # def F(self) -> vector[double]:
    #
    #     self.eval_F()
    #
    #     return self._F

    # @cython.cfunc
    # def step(self, dt: double) -> void:
    #     """
    #     Update particle position based on forces and velocity, using velocity Verlet
    #     integration
    #     """
    #
    #     # Update the force
    #     self.eval_F()
    #
    #     # Update the acceleration
    #     self._a1[0] = self._F[0] / self.m
    #     self._a1[1] = self._F[1] / self.m
    #     self._a1[2] = self._F[2] / self.m
    #
    #     # update the coordinates
    #     self.r[0] += self.v[0] * dt + 0.5 * self._a1[0] * dt * dt
    #     self.r[1] += self.v[1] * dt + 0.5 * self._a1[1] * dt * dt
    #     self.r[2] += self.v[2] * dt + 0.5 * self._a1[2] * dt * dt
    #
    #     # update the force again, at the new coordinate
    #     self.eval_F()
    #
    #     # update the acceleration again
    #     self._a1[0] = self._F[0] / self.m
    #     self._a1[1] = self._F[1] / self.m
    #     self._a1[2] = self._F[2] / self.m
    #
    #     # update the velocity
    #     self.v[0] += 0.5 * (self._a1[0] + self._a2[0]) * dt
    #     self.v[1] += 0.5 * (self._a1[1] + self._a2[1]) * dt
    #     self.v[2] += 0.5 * (self._a1[2] + self._a2[2]) * dt
    #
    #     # update the time
    #     self.t += dt
    #
    # @cython.ccall
    # def trace(self, duration: double, dt: double):
    #     """
    #     Evolve the position and velocity of the particle over the given duration,
    #     broken into N time steps
    #     """
    #
    #     N: ulong = cython.cast(ulong, duration / dt)
    #
    #     trajectory = np.empty((N, 3), dtype=np.double)
    #     traj_view = cython.declare(double[:, :], trajectory)
    #
    #     for i in range(N):
    #         self.step(dt)
    #
    #         traj_view[i, 0] = self.r[0]
    #         traj_view[i, 1] = self.r[1]
    #         traj_view[i, 2] = self.r[2]
    #
    #     return np.asarray(traj_view)


class Particle(_Particle):
    pass
    # @property
    # def L(self) -> np.ndarray:
    #     """Angular momentum vector"""
    #     return self.m * np.linalg.cross(self.r, self.v)
    #
    # @property
    # def KE(self) -> float:
    #     """Kinetic energy"""
    #     return 0.5 * self.m * np.dot(self.v, self.v)
    #
    # @property
    # def PE(self) -> float:
    #     """Potential energy"""
    #     return self.q * self.field.V(self.r, self.t)
    #
    # @property
    # def TE(self) -> float:
    #     """Total energy"""
    #     return self.KE + self.PE
