# Electromagnetic fields, either explicitly defined or generated from bodies
from collections.abc import Iterable

import cython
from cython import int, double, void
from cython.cimports.libcpp.vector import vector
import numpy as np

@cython.cfunc
def dot(v1: vector[double], v2: vector[double]) -> double:
    """Compute the dot product of two 3-vectors"""
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

@cython.cfunc
def cross(i: int, v1: vector[double], v2: vector[double]) -> double:
    """Compute the ith element of the cross product of two 3-vectors"""

    j: int = (i + 1) % 3
    k: int = (i + 2) % 3

    return v1[j] * v2[k] - v1[k] * v2[j]


@cython.cclass
class _Field:
    """
    Base class for all electromagnetic fields.

    Establishes the electric potential (V), electric field (E), magnetic
    potential (A) and magnetic field (B).
    """

    def __cinit__(self):

        self._V = 0.0

        # Initialize the vector field values
        self._E.push_back(0.0)
        self._E.push_back(0.0)
        self._E.push_back(0.0)

        self._A.push_back(0.0)
        self._A.push_back(0.0)
        self._A.push_back(0.0)

        self._B.push_back(0.0)
        self._B.push_back(0.0)
        self._B.push_back(0.0)

    @cython.ccall
    def V(self, r: vector[double], t: double) -> double:
        """Update and return self._V"""
        return self._V

    @cython.ccall
    def E(self, r: vector[double], t: double) -> vector[double]:
        """Update and return self._E"""
        return self._E

    @cython.ccall
    def A(self, r: vector[double], t: double) -> vector[double]:
        """Update and return self._A"""
        return self._A

    @cython.ccall
    def B(self, r: vector[double], t: double) -> vector[double]:
        """Update and return self._B"""
        return self._B


class Field(_Field):
    """Python wrapper for _Field C type"""
    pass


@cython.cclass
class _CompositeField(_Field):

    def __init__(self, components: vector[_Field]):

        self._components = components

    @cython.ccall
    def V(self, r: vector[double], t: double) -> double:

        self._V = 0.0

        for component in self._components:
            self._V += component.V(r, t)

        return self._V

    @cython.ccall
    def E(self, r: vector[double], t: double) -> vector[double]:

        self._E[0] = 0.0
        self._E[1] = 0.0
        self._E[2] = 0.0

        for component in self._components:

            _E: vector[double] = component.E(r, t)

            self._E[0] += _E[0]
            self._E[1] += _E[1]
            self._E[1] += _E[1]

        return self._E

    @cython.ccall
    def A(self, r: vector[double], t: double) -> vector[double]:

        self._A[0] = 0.0
        self._A[1] = 0.0
        self._A[2] = 0.0

        for component in self._components:
            _A: vector[double] = component.A(r, t)

            self._A[0] += _A[0]
            self._A[1] += _A[1]
            self._A[1] += _A[1]

        return self._A

    @cython.ccall
    def B(self, r: vector[double], t: double) -> vector[double]:

        self._B[0] = 0.0
        self._B[1] = 0.0
        self._B[2] = 0.0

        for component in self._components:
            _B: vector[double] = component.B(r, t)

            self._B[0] += _B[0]
            self._B[1] += _B[1]
            self._B[1] += _B[1]

        return self._B


class CompositeField(_CompositeField):
    pass
