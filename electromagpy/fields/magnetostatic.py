# Extension module for magnetostatic fields

import cython
from cython import double
from cython.cimports.libcpp.vector import vector
from cython.cimports.electromagpy.fields.field import _Field, cross

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

    def __init__(self, I: double, radius: double, center: vector[double]):

        self._I = I
        self._radius = radius
        self._center = center

    @cython.ccall
    def A(self, r: vector[double], t: double) -> vector[double]:

        # magnetic potential of current loop

        return self._A

    @cython.ccall
    def B(self, r: vector[double], t: double) -> vector[double]:

        # magnetic field of current loop

        return self._B
