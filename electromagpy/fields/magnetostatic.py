# distutils: language=c++
# Extension module for magnetostatic fields

import cython
from cython import double
from cython.cimports.libcpp.numbers import pi
from cython.cimports.libcpp.cmath import sqrt as csqrt
from cython.cimports.scipy.special.cython_special import ellipk as K, ellipe as E
from cython.cimports.libcpp.vector import vector
from cython.cimports.electromagpy.fields.field import _Field, dot, cross

from math import sqrt

mu0: double = pi*4.0e-7


@cython.cclass
class _UniformB(_Field):
    """Constant uniform magnetic field"""

    def __init__(self, Bx: double, By: double, Bz: double):

        self._B[0] = Bx
        self._B[1] = By
        self._B[2] = Bz

    @cython.ccall
    def A(self, r: vector[double], t: double) -> vector[double]:
        self._A[0] = 0.5 * cross(0, self._B, r)
        self._A[1] = 0.5 * cross(1, self._B, r)
        self._A[2] = 0.5 * cross(2, self._B, r)

        return self._A


class UniformB(_UniformB):
    pass


@cython.cclass
class _CurrentLoop(_Field):
    """
    Field of a steady circular current loop

    Magnetic potential and field are taken from W.R. Smythe's "Static and Dynamic
    Electricity", starting from page 270.
    """

    I: double
    a: double

    r0: vector[double]
    n: vector[double]

    def __init__(
            self,
            I: double, radius: double,
            center: vector[double] = (0.0, 0.0, 0.0),
            normal: vector[double] = (0.0, 0.0, 1.0)
    ):

        self.I = I
        self.a = radius

        self.r0 = center

        self.n = normal

        norm_n: double = sqrt(self.n[0]**2 + self.n[1]**2 + self.n[2]**2)
        self.n[0] = self.n[0] / norm_n
        self.n[1] = self.n[1] / norm_n
        self.n[2] = self.n[2] / norm_n

    @cython.ccall
    def A(self, r: vector[double], t: double) -> vector[double]:
        """Taken from page 932 of Arfken, Weber and Harris"""

        # compute displacement of position from loop center
        xp: double = r[0] - self.r0[0]
        yp: double = r[1] - self.r0[1]
        zp: double = r[2] - self.r0[2]

        rp2: double = xp*xp + yp*yp + zp*zp  # magnitude squared
        rp: double = csqrt(rp2)  # magnitude

        # compute cylindrical coordinates oriented to the loop
        zpp: double = xp*self.n[0] + yp*self.n[1] + zp*self.n[2]
        rho: double = csqrt(rp2 - zpp*zpp)
        print(rho, rp2, zpp)
        if rho == 0.0:

            self._A[0] = 0.0
            self._A[1] = 0.0
            self._A[2] = 0.0

            return self._A

        # compute argument k
        k2: double = 4.0 * self.a * rho / ((self.a + rho)*(self.a + rho) + zpp*zpp)
        k: double = csqrt(k2)

        # compute azimuthal component in loop-oriented cylindrical coordinates
        Athpp: double = (4.0e-7*self.I/k) * csqrt(self.a/rho) * ((1.0-0.5*k2)*K(k2) - E(k2))

        # compute loop-oriented azimuthal unit vector in original cartesian basis
        thpphat0: double = (self.n[1]*zp - self.n[2]*yp) / rp
        thpphat1: double = (self.n[2]*xp - self.n[0]*zp) / rp
        thpphat2: double = (self.n[0]*yp - self.n[1]*xp) / rp

        # impute magnetic potential components
        self._A[0] = Athpp * thpphat0
        self._A[1] = Athpp * thpphat1
        self._A[2] = Athpp * thpphat2

        return self._A

    @cython.ccall
    def B(self, r: vector[double], t: double) -> vector[double]:

        # compute displacement of position from loop center
        xp: double = r[0] - self.r0[0]
        yp: double = r[1] - self.r0[1]
        zp: double = r[2] - self.r0[2]

        rp2: double = xp * xp + yp * yp + zp * zp  # magnitude squared
        rp: double = csqrt(rp2)  # magnitude

        # compute cylindrical coordinates oriented to the loop
        zpp: double = xp * self.n[0] + yp * self.n[1] + zp * self.n[2]
        rho: double = csqrt(rp2 - zpp * zpp)

        if rho == 0.0:
            # compute on-axis B

            Bzpp: double = 0.5*mu0*self.a*self.a*self.I / csqrt(self.a*self.a + zpp*zpp)**3

            self._B[0] = Bzpp * self.n[0]
            self._B[1] = Bzpp * self.n[1]
            self._B[2] = Bzpp * self.n[2]

            return self._B

        # compute argument k
        k2: double = 4.0 * self.a * rho / ((self.a + rho) * (self.a + rho) + zpp * zpp)

        factor1: double = 1.0 / csqrt((self.a+rho)*(self.a+rho) + zpp*zpp)
        factor2: double = (self.a*self.a + rho*rho + zpp*zpp) / ((self.a-rho)*(self.a-rho) + zpp*zpp)
        factor3: double = (self.a*self.a - rho*rho - zpp*zpp) / ((self.a-rho)*(self.a-rho) + zpp*zpp)

        Brho = (2.0e-7 * self.I) * (zpp/rho) * factor1 * (-K(k2) + factor2*E(k2))
        Bzpp = (2.0e-7 * self.I) * factor1 * (K(k2) + factor3 * E(k2))

        # compute loop-oriented azimuthal unit vector in original cartesian basis
        thpphat0: double = (self.n[1] * zp - self.n[2] * yp) / rp
        thpphat1: double = (self.n[2] * xp - self.n[0] * zp) / rp
        thpphat2: double = (self.n[0] * yp - self.n[1] * xp) / rp

        # compute loop-oriented cylindrical radial unit vector in original cartesian basis
        rhohat0: double = thpphat1 * self.n[2] - thpphat2 * self.n[1]
        rhohat1: double = thpphat2 * self.n[0] - thpphat0 * self.n[2]
        rhohat2: double = thpphat0 * self.n[1] - thpphat1 * self.n[0]

        self._B[0] = Brho * rhohat0 + Bzpp * self.n[0]
        self._B[1] = Brho * rhohat1 + Bzpp * self.n[1]
        self._B[2] = Brho * rhohat2 + Bzpp * self.n[2]

        return self._B


class CurrentLoop(_CurrentLoop):
    pass
