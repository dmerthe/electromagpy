# Extension module for magnetostatic fields

import cython
from cython import double
from cython.cimports.libcpp.numbers import pi
from cython.cimports.libcpp.cmath import sqrt as csqrt, comp_ellint_1 as K, comp_ellint_2 as E
from cython.cimports.libcpp.vector import vector
from cython.cimports.electromagpy.fields.field import _Field, dot, cross

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

    def __init__(
            self,
            I: double, radius: double,
            center: vector[double] = None,
            normal: vector[double] = None
    ):

        self.I: double = I
        self.a: double = radius

        if center is not None:
            self.r0: vector[double] = center
        else:
            self.r0: vector[double]
            self.r0.push_back(0.0)
            self.r0.push_back(0.0)
            self.r0.push_back(0.0)

        if normal is not None:
            self.n: vector[double] = normal
        else:
            self.n: vector[double]
            self.n.push_back(0.0)
            self.n.push_back(0.0)
            self.n.push_back(1.0)

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

        if rho == 0.0:

            self._A[0] = 0.0
            self._A[1] = 0.0
            self._A[2] = 0.0

            return self._A

        # compute argument k
        k: double = csqrt(4*self.a * rho / ((self.a + rho)*(self.a + rho) + zpp*zpp))

        # compute azimuthal component in loop-oriented cylindrical coordinates
        Athpp: double = (4.0e-7*self.I/k) * csqrt(self.a/rho) * ((1.0-0.5*k*k)*K(k) - E(k))

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
            # compute on axis B

            Brho = 0.0
            Bzpp = 0.5*mu0*self.a*self.a*self.I / csqrt(self.a*self.a + zpp*zpp)

        else:
            # compute off axis B

            # compute argument k
            k: double = csqrt(4 * self.a * rho / ((self.a + rho) * (self.a + rho) + zpp * zpp))

            factor1: double = 1.0 / csqrt((self.a+rho)*(self.a+rho) + zpp*zpp)
            factor2: double = (self.a*self.a + rho*rho + zpp*zpp) / ((self.a-rho)*(self.a-rho) + zpp*zpp)
            factor3: double = (self.a*self.a - rho*rho - zpp*zpp) / ((self.a-rho)*(self.a-rho) + zpp*zpp)

            Brho = (2.0e-7 * self.I) * (zpp/rho) * factor1 * (-K(k) + factor2*E(k))
            Bzpp = (2.0e-7 * self.I) * factor1 * (K(k) + factor3 * E(k))

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
