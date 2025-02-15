# distutils: language=c++
# cython extension module for electrostatic fields

import cython
from cython import double, void
from math import sqrt, log, pi
from cython.cimports.libcpp.cmath import log as clog
from cython.cimports.libcpp.vector import vector
from cython.cimports.electromagpy.fields.field import _Field, dot

@cython.cclass
class _Vacuum(_Field):
    """Region with zero electromagnetic field"""
    pass


class Vacuum(_Vacuum):
    pass

@cython.cclass
class _UniformE(_Field):
    """Constant uniform magnetic field"""

    def __init__(self, Ex: double, Ey: double, Ez: double):

        self._E[0] = Ex
        self._E[1] = Ey
        self._E[2] = Ez

    @cython.cfunc
    def eval_V(self, r: vector[double], t: double) -> void:

        self._V = -dot(self._E, r)


class UniformE(_UniformE):
    pass


@cython.cclass
class _Orbitrap(_Field):
    """
    Electrostatic orbitrap, can be defined by its bias (anode-cathode), rc
    (inner/cathode radius), ra (outer/anode radius) and Ra (radius of
    curvature of the anode at z = 0), or by the traditional orbitrap potential
    parameters, k, Rm and C.
    """

    k: double
    Rm: double
    C: double

    def __init__(self, k: double, Rm: double, C: double):

        self.k = k
        self.Rm = Rm
        self.C = C

    @cython.cfunc
    def eval_V(self, r: vector[double], t: double) -> void:

        rr2: double = r[0] * r[0] + r[1] * r[1]  # cylindrical radius squared
        Rm2: double = self.Rm * self.Rm  # Rm squared

        self._V = 0.5 * self.k * (
                (r[2]*r[2] - 0.5 * rr2) + 0.5 * Rm2 * clog(rr2 / Rm2)
        ) + self.C

    @cython.cfunc
    def eval_E(self, r: vector[double], t: double) -> void:

        rr2: double = r[0]*r[0] + r[1]*r[1]

        factor: double = 0.5 * self.k * (1.0 - self.Rm*self.Rm / rr2)

        self._E[0] = r[0] * factor
        self._E[1] = r[1] * factor
        self._E[2] = -self.k * r[2]


class Orbitrap(_Orbitrap):

    @staticmethod
    def params_from_geom(bias: float, rc: float, ra: float, Ra: float) -> list:
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

        return [k, Rm, C]

    def calc_axial_freq(self, q: float, m: float) -> float:
        """
        Calculate the axial oscillation frequency of a particle of charge q and mass m
        in an orbitrap
        """

        return sqrt(self.k * q / m) / (2 * pi)

    def calc_circ_orb(self, q: float, m: float, Lz: float) -> float:
        """
        Calculate the circular orbit at z = 0 for a particle of charge q and mass m and
        azimuthal angular momentum Lz
        """

        w0 = self.calc_axial_freq(q, m)

        Lhat = Lz / (m*self.Rm*w0**2)

        return self.Rm * sqrt(0.5-0.5*sqrt(1.0-8.0*Lhat**2))
