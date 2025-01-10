# Physical bodies that generate electromagnetic fields
import cython
from math import log, sqrt, pi


def dot(v1: cython.float[3], v2: cython.float[3]) -> cython.float:
    """Compute the dot product of two 3-vectors"""
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def cross(v1: cython.float[3], v2: cython.float[3]) -> cython.float[3]:
    """Compute the cross product of two 3-vectors"""

    v3: cython.float[3]

    v3[0] = v1[1] * v2[2] - v1[2] * v2[1]
    v3[1] = v1[2] * v2[0] - v1[0] * v2[2]
    v3[2] = v1[0] * v2[1] - v1[1] * v2[0]

    return v3


class Field:
    """
    Base class for all electromagnetic fields.

    Establishes the electric potential (V), electric field (E), magnetic
    potential (A) and magnetic field (B).
    """

    _V: cython.float = 0.0
    _E: cython.float[3] = [0.0, 0.0, 0.0]
    _A: cython.float[3] = [0.0, 0.0, 0.0]
    _B: cython.float[3] = [0.0, 0.0, 0.0]

    def V(self, r: cython.float[3], t: cython.float):
        return self._V

    def E(self, r: cython.float[3], t: cython.float):
        return self._E

    def A(self, r: cython.float[3], t: cython.float):
        return self._A

    def B(self, r: cython.float[3], t: cython.float):
        return self._B


class Vacuum(Field):
    """Region with zero electromagnetic field"""
    pass


class Orbitrap(Field):
    """
    Electrostatic orbitrap, can be defined by its bias (anode-cathode), rc
    (inner/cathode radius), ra (outer/anode radius) and Ra (radius of
    curvature of the anode at z = 0), or by the traditional orbitrap potential
    parameters, k, Rm and C.
    """

    def __init__(self, k: cython.float, Rm: cython.float, C: cython.float):

        self.k = k
        self.Rm = Rm
        self.C = C

    def V(self, r: cython.float[3], t: cython.float):

        rr2 = r[0] * r[0] + r[1] * r[1]
        rr = sqrt(rr2)  # cylindrical radius

        return 0.5*self.k * ((r[2]*r[2] - 0.5 * rr2) + self.Rm*self.Rm*log(rr / self.Rm)) + self.C

    def E(self, r: cython.float[3], t: cython.float):

        rr2 = r[0]*r[0] + r[1]*r[1]

        factor = 0.5 * self.k * (1.0 - self.Rm*self.Rm / rr2)

        self._E[0] = r[0] * factor
        self._E[1] = r[1] * factor
        self._E[2] = -self.k * r[2]

        return self._E

    @staticmethod
    def params_from_geom(
            bias: cython.float, rc: cython.float, ra: cython.float, Ra: cython.float
    ) -> cython.float[3]:
        """
        Calculate the orbitrap parameters (k, Rm, C) from the electrical bias and
        geometrical parameters of the electrodes.

        :param bias: electrical bias (anode-cathode)
        :param rc: radius of the cathode in the z = 0 plane
        :param ra: radius of the anode in the z = 0 plane
        :param Ra: radius of curvature in the rz-plane at z = 0
        """

        Rm = sqrt(ra ** 2 + 2 * ra * Ra)
        k = 4 * bias / (2 * Rm ** 2 * log(ra / rc) - ra ** 2 + rc ** 2)
        C = (k / 2) * (ra ** 2 / 2 + Rm ** 2 * log(Rm / ra))

        return k, Rm, C

    def calc_axial_freq(self, q: cython.float, m: cython.float) -> cython.float:
        """
        Calculate the axial oscillation frequency of a particle
        """

        return sqrt(self.k * q / m) / (2 * pi)

    def calc_circ_orb(
            self, q: cython.float, m: cython.float, Lz: cython.float) -> cython.float:
        """
        Calculate the equivalent circular orbit at z = 0 for a particle based on
        azimuthal angular momentum
        """

        w0: cython.float = self.calc_axial_freq(q, m)

        Lhat = Lz / (m*self.Rm*w0**2)

        return self.Rm * sqrt(0.5-0.5*sqrt(1.0-8.0*Lhat**2))
