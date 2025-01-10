# Particles that respond to electromagnetic fields
import cython
from fields import Field, Vacuum, dot, cross


class Particle:
    """
    Charged particle that responds to electromagnetic fields
    """

    q: cython.float = 0.0
    m: cython.float = 0.0

    t: cython.float = 0.0
    r: cython.float[3] = [0.0, 0.0, 0.0]
    v: cython.float[3] = [0.0, 0.0, 0.0]

    field = Vacuum()

    # pre-initialized acceleration vectors
    _a1: cython.float[3] = [0.0, 0.0, 0.0]
    _a2: cython.float[3] = [0.0, 0.0, 0.0]

    # pre-initialized field vectors
    _E: cython.float[3] = [0.0, 0.0, 0.0]
    _B: cython.float[3] = [0.0, 0.0, 0.0]

    def __init__(self, q: cython.float, m: cython.float, field=None, r=None, v=None):

        self.q = q
        self.m = m

        if field is not None:
            if isinstance(field, Field):
                self.field = field
            else:
                raise ValueError('field must be an instance of fields.Field')

        if r is not None:
            self.r = r

        if v is not None:
            self.v = v

    def force(self) -> cython.float[3]:

        self._E = self.field.E(self.r, self.t)
        self._B = self.field.B(self.r, self.t)

        return self.q * self._E + self.q * cross(self.v, self._B)

    def update(self, dt: cython.float) -> cython.void:
        """
        Update particle position based on forces and velocity, using velocity Verlet
        integration
        """

        self._a1 = self.force() / self.m

        self.r += self.v * dt + 0.5 * self._a1 * dt * dt

        self._a2 = self.force() / self.m  # update based on new position

        self.v += 0.5 * (self._a1 + self._a2) * dt

        self.t += dt

    def evolve(self, duration: cython.float, dt: cython.float) -> cython.void:
        """
        Evolve the position and velocity of the particle over the given duration with
        time step dt
        """

        # TODO: add error estimation

        while self.t < self.t + duration:
            self.update(dt)

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
