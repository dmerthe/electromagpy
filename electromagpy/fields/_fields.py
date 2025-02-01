# Electromagnetic fields, either explicitly defined or generated from bodies

from numba import jit, float64, void
import numpy as np


class Field:
    """
    Base class for all electromagnetic fields.

    Establishes the electric potential (V), electric field (E), magnetic
    potential (A) and magnetic field (B).
    """

    _V = 0.0  # electric potential
    _E = np.array([0.0, 0.0, 0.0])  # electric field
    _A = np.array([0.0, 0.0, 0.0])  # magnetic potential
    _B = np.array([0.0, 0.0, 0.0])  # magnetic field

    @jit(void(float64[:], float64))
    def _eval_V(self, r, t):
        """Evaluate the electric potential at the point r and time t"""
        pass

    @cython.cfunc
    def _eval_E(self, r: cython.double[3], t: cython.double) -> cython.void:
        """Evaluate the electric field at the point r and time t"""
        pass

    @cython.cfunc
    def _eval_A(self, r: cython.double[3], t: cython.double) -> cython.void:
        """Evaluate the magnetic potential at the point r and time t"""
        pass

    @cython.cfunc
    def _eval_B(self, r: cython.double[3], t: cython.double) -> cython.void:
        """Evaluate the magnetic field at the point r and time t"""
        pass

    @cython.cfunc
    def V(self, r: cython.double[3], t: cython.double) -> cython.double:

        self._eval_V(r, t)

        return self._V

    @cython.cfunc
    def E(self, r: cython.double[3], t: cython.double) -> cython.double[:]:

        self._eval_E(r, t)

        E_view = cython.declare(cython.double[:], self._E)

        return E_view

    @cython.cfunc
    def A(self, r: cython.double[3], t: cython.double) -> cython.double[:]:

        self._eval_A(r, t)

        A_view = cython.declare(cython.double[:], self._A)

        return A_view

    @cython.cfunc
    def B(self, r: cython.double[3], t: cython.double) -> cython.double[:]:

        self._eval_B(r, t)

        B_view = cython.declare(cython.double[:], self._B)

        return B_view
