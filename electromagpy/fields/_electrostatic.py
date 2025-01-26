# cython extension module for electrostatic fields

import cython
from _fields import _Field, py_field_factory


@cython.cclass
class _Vacuum(_Field):
    """Region with zero electromagnetic field"""
    pass


Vacuum = py_field_factory(_Vacuum)


# class Orbitrap(Field):
#     """
#     Electrostatic orbitrap, can be defined by its bias (anode-cathode), rc
#     (inner/cathode radius), ra (outer/anode radius) and Ra (radius of
#     curvature of the anode at z = 0), or by the traditional orbitrap potential
#     parameters, k, Rm and C.
#     """
#
#     def __init__(self, k: cython.float, Rm: cython.float, C: cython.float):
#
#         self.k = k
#         self.Rm = Rm
#         self.C = C
#
#     def V(self, r: cython.float[3], t: cython.float):
#
#         rr2 = r[0] * r[0] + r[1] * r[1]
#         rr = sqrt(rr2)  # cylindrical radius
#
#         return 0.5*self.k * ((r[2]*r[2] - 0.5 * rr2) + self.Rm*self.Rm*log(rr / self.Rm)) + self.C
#
#     def E(self, r: cython.float[3], t: cython.float):
#
#         rr2 = r[0]*r[0] + r[1]*r[1]
#
#         factor = 0.5 * self.k * (1.0 - self.Rm*self.Rm / rr2)
#
#         self._E[0] = r[0] * factor
#         self._E[1] = r[1] * factor
#         self._E[2] = -self.k * r[2]
#
#         return self._E
#
#     @staticmethod
#     def params_from_geom(
#             bias: cython.float, rc: cython.float, ra: cython.float, Ra: cython.float
#     ) -> cython.float[3]:
#         """
#         Calculate the orbitrap parameters (k, Rm, C) from the electrical bias and
#         geometrical parameters of the electrodes.
#
#         :param bias: electrical bias (anode-cathode)
#         :param rc: radius of the cathode in the z = 0 plane
#         :param ra: radius of the anode in the z = 0 plane
#         :param Ra: radius of curvature in the rz-plane at z = 0
#         """
#
#         Rm = sqrt(ra ** 2 + 2 * ra * Ra)
#         k = 4 * bias / (2 * Rm ** 2 * log(ra / rc) - ra ** 2 + rc ** 2)
#         C = (k / 2) * (ra ** 2 / 2 + Rm ** 2 * log(Rm / ra))
#
#         return k, Rm, C
#
#     def calc_axial_freq(self, q: cython.float, m: cython.float) -> cython.float:
#         """
#         Calculate the axial oscillation frequency of a particle
#         """
#
#         return sqrt(self.k * q / m) / (2 * pi)
#
#     def calc_circ_orb(
#             self, q: cython.float, m: cython.float, Lz: cython.float) -> cython.float:
#         """
#         Calculate the equivalent circular orbit at z = 0 for a particle based on
#         azimuthal angular momentum
#         """
#
#         w0: cython.float = self.calc_axial_freq(q, m)
#
#         Lhat = Lz / (m*self.Rm*w0**2)
#
#         return self.Rm * sqrt(0.5-0.5*sqrt(1.0-8.0*Lhat**2))


