# Particles that respond to electromagnetic fields
import cython
import numpy as np
from electromagpy.fields.field import dot, cross, _Field, Field
from electromagpy.fields.electrostatic import Vacuum

from collections.abc import Iterable


@cython.cclass
class _Particle:
    """
    Charged particle that responds to electromagnetic fields
    """

    q: cython.double
    m: cython.double

    t: cython.double
    r: cython.double[3]
    v: cython.double[3]

    #pre-initialized electric and magnetic field vector
    _E: cython.double[3]
    _B: cython.double[3]

    # pre-initialized acceleration and force vectors
    _F: cython.double[3]
    _a1: cython.double[3]  # there are two acceleration vectors used
    _a2: cython.double[3]  # by the velocity-Verlet algorithm

    def __init__(
            self, q: float, m: float, field: _Field,
            r: Iterable, v: Iterable
    ):

        self.q = q
        self.m = m

        self.field = field
        self.r = list(r)
        self.v = list(v)

    @cython.cfunc
    def _eval_F(self) -> cython.void:

        self.field._eval_E(self.r, self.t)
        self.field._eval_B(self.r, self.t)

        self._F[0] = self.q * (self.field._E[0] + cross(0, self.v, self.field._B))
        self._F[1] = self.q * (self.field._E[1] + cross(1, self.v, self.field._B))
        self._F[2] = self.q * (self.field._E[2] + cross(2, self.v, self.field._B))

    @cython.cfunc
    def F(self) -> cython.double[:]:

        self._eval_F()

        F_view = cython.declare(cython.double[:], self._F)

        return F_view

    @cython.cfunc
    def update(self, dt: cython.double) -> cython.void:
        """
        Update particle position based on forces and velocity, using velocity Verlet
        integration
        """

        # Update the force
        self._eval_F()

        # Update the acceleration
        self._a1[0] = self._F[0] / self.m
        self._a1[1] = self._F[1] / self.m
        self._a1[2] = self._F[2] / self.m

        # update the coordinates
        self.r[0] += self.v[0] * dt + 0.5 * self._a1[0] * dt * dt
        self.r[1] += self.v[1] * dt + 0.5 * self._a1[1] * dt * dt
        self.r[2] += self.v[2] * dt + 0.5 * self._a1[2] * dt * dt

        # update the force again, at the new coordinate
        self._eval_F()

        # update the acceleration again
        self._a2[0] = self._F[0] / self.m
        self._a2[1] = self._F[1] / self.m
        self._a2[2] = self._F[2] / self.m

        # update the velocity
        self.v[0] += 0.5 * (self._a1[0] + self._a2[0]) * dt
        self.v[1] += 0.5 * (self._a1[1] + self._a2[1]) * dt
        self.v[2] += 0.5 * (self._a1[2] + self._a2[2]) * dt

        # update the time
        self.t += dt

    @cython.cfunc
    def evolve(self, duration: cython.double, dt: cython.double) -> cython.void:
        """
        Evolve the position and velocity of the particle over the given duration with
        time step dt
        """

        t: cython.double = 0.0

        while t < duration:
            self.update(dt)
            t += dt


class Particle:

    def __init__(
            self, q: float, m: float, field: Field = None,
            r: Iterable = None, v: Iterable = None
    ):

        if field is None:
            field = Vacuum()

        self._field = field

        if r is None:
            r = [0.0, 0.0, 0.0]
        else:
            r = [float(ri) for ri in r]

        if v is None:
            v = [0.0, 0.0, 0.0]
        else:
            v = [float(vi) for vi in v]

        self._particle = _Particle(q, m, field._field, r, v)

    @property
    def q(self) -> float:
        return self._particle.q

    @q.setter
    def q(self, q: float):
        self._particle.q = q

    @property
    def m(self) -> float:
        return self._particle.m

    @m.setter
    def m(self, m: float):
        self._particle.m = m

    @property
    def field(self) -> Field:
        return self._field

    @field.setter
    def field(self, field: Field):
        self._field = field
        self._particle.field = field._field

    @property
    def r(self) -> np.ndarray:
        return np.array([ri for ri in self._particle.r], dtype=np.float64)

    @r.setter
    def r(self, r: Iterable):
        self._particle.r[0] = r[0]
        self._particle.r[1] = r[1]
        self._particle.r[2] = r[2]

    @property
    def v(self) -> np.ndarray:
        return np.array([vi for vi in self._particle.v], dtype=np.float64)

    @v.setter
    def v(self, v: Iterable):
        self._particle.v[0] = v[0]
        self._particle.v[1] = v[1]
        self._particle.v[2] = v[2]

    @property
    def F(self) -> np.ndarray:

        return self._particle.F

    @property
    def L(self) -> np.ndarray:
        """Angular momentum vector"""
        return self.m * np.linalg.cross(self.r, self.v)

    @property
    def KE(self) -> float:
        """Kinetic energy"""
        return 0.5 * self.m * np.dot(self.v, self.v)

    @property
    def PE(self) -> float:
        """Potential energy"""
        return self.q * self.field.V(self.r, self.t)

    @property
    def TE(self) -> float:
        """Total energy"""
        return self.KE + self.PE
