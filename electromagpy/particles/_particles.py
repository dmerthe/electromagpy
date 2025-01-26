# Particles that respond to electromagnetic fields
import cython
import numpy as np
from electromagpy.fields import dot, cross, Field, Vacuum
from electromagpy.fields._fields import _Field


@cython.cclass
class _Particle:
    """
    Charged particle that responds to electromagnetic fields
    """

    q: cython.float
    m: cython.float

    t: cython.float
    r: cython.float[3]
    v: cython.float[3]

    #pre-initialized electric and magnetic field vector
    _E: cython.float[3]
    _B: cython.float[3]

    # pre-initialized acceleration and force vectors
    _F: cython.float[3]
    _a1: cython.float[3]
    _a2: cython.float[3]

    def __cinit__(
            self, q: cython.float, m: cython.float, field: _Field,
            r: cython.double[3], v: cython.double[3]
    ):

        self.q = q
        self.m = m

        self.field = field
        self.r = r
        self.v = v

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
    def update(self, dt: cython.float) -> cython.void:
        """
        Update particle position based on forces and velocity, using velocity Verlet
        integration
        """

        self._eval_F()

        self._a1[0] = self._F[0] / self.m
        self._a1[1] = self._F[1] / self.m
        self._a1[2] = self._F[2] / self.m

        self.r[0] += self.v[0] * dt + 0.5 * self._a1[0] * dt * dt
        self.r[1] += self.v[1] * dt + 0.5 * self._a1[1] * dt * dt
        self.r[2] += self.v[2] * dt + 0.5 * self._a1[2] * dt * dt

        self._eval_F()

        self._a2[0] = self._F[0] / self.m  # update based on new position
        self._a2[1] = self._F[1] / self.m
        self._a2[2] = self._F[2] / self.m

        self.v[0] += 0.5 * (self._a1[0] + self._a2[0]) * dt
        self.v[1] += 0.5 * (self._a1[1] + self._a2[1]) * dt
        self.v[2] += 0.5 * (self._a1[2] + self._a2[2]) * dt

        self.t += dt

    @cython.cfunc
    def evolve(self, duration: cython.float, dt: cython.float) -> cython.void:
        """
        Evolve the position and velocity of the particle over the given duration with
        time step dt
        """

        # TODO: add error estimation
        t0: cython.double = self.t

        while self.t < t0 + duration:
            self.update(dt)


class Particle:

    def __init__(self, q, m, field=None, r=None, v=None):

        if field is None:
            field = Vacuum()

        self._field = field

        if r is None:
            r = [0.0, 0.0, 0.0]

        if v is None:
            v = [0.0, 0.0, 0.0]

        self._particle = _Particle(q, m, field._field, r, v)

    @property
    def q(self):
        return self._particle.q

    @q.setter
    def q(self, q):
        self._particle.q = q

    @property
    def m(self):
        return self._particle.m

    @m.setter
    def m(self, m):
        self._particle.m = m

    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, field):
        self._field = field
        self._particle.field = field._field

    @property
    def r(self):
        return np.array([ri for ri in self._particle.r], dtype=np.float64)

    @r.setter
    def r(self, r):
        self._particle.r[0] = r[0]
        self._particle.r[1] = r[1]
        self._particle.r[2] = r[2]

    @property
    def v(self):
        return np.array([vi for vi in self._particle.v], dtype=np.float64)

    @v.setter
    def v(self, v):
        self._particle.v[0] = v[0]
        self._particle.v[1] = v[1]
        self._particle.v[2] = v[2]

    @property
    def L(self) -> cython.float[3]:
        """Angular momentum vector"""
        return self.m * cross(self.r, self.v)

    @property
    def KE(self) -> cython.float:
        """Kinetic energy"""
        return 0.5 * self.m * dot(self.v, self.v)

    @property
    def PE(self) -> cython.float:
        """Potential energy"""
        return self.q * self.field.V(self.r, self.t)

    @property
    def TE(self) -> cython.float:
        """Total energy"""
        return self.KE + self.PE
