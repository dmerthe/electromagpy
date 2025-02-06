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

    j: cython.int = (i + 1) % 3
    k: cython.int = (i + 2) % 3

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

        # Initialize the vector values
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
