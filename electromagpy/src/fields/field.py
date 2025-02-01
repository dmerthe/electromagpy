# Electromagnetic fields, either explicitly defined or generated from bodies
import cython
import numpy as np


@cython.cfunc
def dot(v1: cython.double[3], v2: cython.double[3]) -> cython.double:
    """Compute the dot product of two 3-vectors"""
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

@cython.cfunc
def cross(i: cython.int, v1: cython.double[3], v2: cython.double[3]) -> cython.double:
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

    _V: cython.double  # electric potential
    _E: cython.double[3]  # electric field
    _A: cython.double[3]  # magnetic potential
    _B: cython.double[3]  # magnetic field

    # @cython.cfunc
    # def _eval_V(self, r: cython.double[3], t: cython.double) -> cython.void:
    #     """Evaluate the electric potential at the point r and time t"""
    #     pass
    #
    # @cython.cfunc
    # def _eval_E(self, r: cython.double[3], t: cython.double) -> cython.void:
    #     """Evaluate the electric field at the point r and time t"""
    #     pass
    #
    # @cython.cfunc
    # def _eval_A(self, r: cython.double[3], t: cython.double) -> cython.void:
    #     """Evaluate the magnetic potential at the point r and time t"""
    #     pass
    #
    # @cython.cfunc
    # def _eval_B(self, r: cython.double[3], t: cython.double) -> cython.void:
    #     """Evaluate the magnetic field at the point r and time t"""
    #     pass
    #
    # @cython.cfunc
    # def V(self, r: cython.double[3], t: cython.double) -> cython.double:
    #
    #     self._eval_V(r, t)
    #
    #     return self._V
    #
    # @cython.cfunc
    # def E(self, r: cython.double[3], t: cython.double) -> cython.double[:]:
    #
    #     self._eval_E(r, t)
    #
    #     E_view = cython.declare(cython.double[:], self._E)
    #
    #     return E_view
    #
    # @cython.cfunc
    # def A(self, r: cython.double[3], t: cython.double) -> cython.double[:]:
    #
    #     self._eval_A(r, t)
    #
    #     A_view = cython.declare(cython.double[:], self._A)
    #
    #     return A_view
    #
    # @cython.cfunc
    # def B(self, r: cython.double[3], t: cython.double) -> cython.double[:]:
    #
    #     self._eval_B(r, t)
    #
    #     B_view = cython.declare(cython.double[:], self._B)
    #
    #     return B_view


def py_field_factory(cfieldclass):
    """
    Generate the python wrapper for the given C field class
    """

    class PythonField(cfieldclass):

        def __init__(self, *args):
            self._field = cfieldclass(*args)

        # def V(self, r, t):
        #     return self._field.V(r, t)
        #
        # def E(self, r, t):
        #     return np.array([Ei for Ei in self._field.E(r, t)])
        #
        # def A(self, r, t):
        #     return np.array([Ai for Ai in self._field.A(r, t)])
        #
        # def B(self, r, t):
        #     return np.array([Bi for Bi in self._field.B(r, t)])

    return PythonField


Field = py_field_factory(_Field)
