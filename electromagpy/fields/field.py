# distutils: language=c++
# Electromagnetic fields, either explicitly defined or generated from bodies
from collections.abc import Iterable

import cython
from cython import int, ulong, double, void
from cython.cimports.libcpp.vector import vector
from cython.cimports.electromagpy.particles.particles import _Particle
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
        self._E = [0.0, 0.0, 0.0]
        self._A = [0.0, 0.0, 0.0]
        self._B = [0.0, 0.0, 0.0]
        self.particles = []

    @cython.cfunc
    def eval_V(self, r: vector[double], t: double) -> void:
        """Update self._V"""
        pass

    @cython.cfunc
    def eval_E(self, r: vector[double], t: double) -> void:
        """Update self._E"""
        pass

    @cython.cfunc
    def eval_A(self, r: vector[double], t: double) -> void:
        """Update self._A"""
        pass

    @cython.cfunc
    def eval_B(self, r: vector[double], t: double) -> void:
        """Update self._B"""
        pass

    @cython.ccall
    def V(self, r: vector[double], t: double) -> double:
        """Update and return self._V"""

        self.eval_V(r, t)

        return self._V

    @cython.ccall
    def E(self, r: vector[double], t: double) -> vector[double]:
        """Update and return self._E"""

        self.eval_E(r, t)

        return self._E

    @cython.ccall
    def A(self, r: vector[double], t: double) -> vector[double]:
        """Update and return self._A"""

        self.eval_A(r, t)

        return self._A

    @cython.ccall
    def B(self, r: vector[double], t: double) -> vector[double]:
        """Update and return self._B"""

        self.eval_B(r, t)

        return self._B

    @cython.cfunc
    def eval_F(self, r: vector[double], v: vector[double], t: double, q: double, F: double[3]) -> void:

        self.eval_E(r, t)
        self.eval_B(r, t)

        F[0] = q * (self._E[0] + cross(0, v, self._B))
        F[1] = q * (self._E[1] + cross(1, v, self._B))
        F[2] = q * (self._E[2] + cross(2, v, self._B))

    @cython.ccall
    def push(self, particles: list, t1: double, t2: double, dt: double):

        M: ulong = len(particles)
        N: ulong = cython.cast(ulong, (t2-t1) / dt)

        trajectory = np.empty((M, N, 3), dtype=np.double)
        traj_view = cython.declare(double[:, :, :], trajectory)

        t: double = t1

        r: vector[double] = [0.0, 0.0, 0.0]
        v: vector[double] = [0.0, 0.0, 0.0]

        F = cython.declare(double[3])
        a1 = cython.declare(double[3])
        a2 = cython.declare(double[3])

        for i in range(M):

            q: double = particles[i].q
            m: double = particles[i].m

            r = particles[i].r
            v = particles[i].v

            for k in range(3):
                F[k] = 0.0
                a1[k] = 0.0
                a2[k] = 0.0

            for j in range(N):

                # Update the force
                self.eval_F(r, v, t, q, F)

                # Update the acceleration
                a1[0] = F[0] / m
                a1[1] = F[1] / m
                a1[2] = F[2] / m

                # update the coordinates
                r[0] += v[0] * dt + 0.5 * a1[0] * dt * dt
                r[1] += v[1] * dt + 0.5 * a1[1] * dt * dt
                r[2] += v[2] * dt + 0.5 * a1[2] * dt * dt

                # update the force again, at the new coordinate
                self.eval_F(r, v, t, q, F)

                # update the acceleration again
                a2[0] = F[0] / m
                a2[1] = F[1] / m
                a2[2] = F[2] / m

                # update the velocity
                v[0] += 0.5 * (a1[0] + a2[0]) * dt
                v[1] += 0.5 * (a1[1] + a2[1]) * dt
                v[2] += 0.5 * (a1[2] + a2[2]) * dt

                # update the time
                t += dt

                # update trajectory
                traj_view[i, j, 0] = r[0]
                traj_view[i, j, 1] = r[1]
                traj_view[i, j, 2] = r[2]

            # update particle coordinates and velocity
            particles[i].r = r
            particles[i].v = v

        return np.asarray(traj_view)


class Field(_Field):
    """Python wrapper for _Field C type"""
    pass


@cython.cclass
class _CompositeField(_Field):

    def __init__(self, components: Iterable[Field]):

        self._components = components

    @cython.cfunc
    def eval_V(self, r: vector[double], t: double) -> void:

        self._V = 0.0

        for component in self._components:

            component.eval_V(r, t)

            self._V += component._V

    @cython.cfunc
    def eval_E(self, r: vector[double], t: double) -> void:

        self._E[0] = 0.0
        self._E[1] = 0.0
        self._E[2] = 0.0

        for component in self._components:

            component.eval_E(r, t)

            self._E[0] += component._E[0]
            self._E[1] += component._E[1]
            self._E[1] += component._E[1]

    @cython.cfunc
    def eval_A(self, r: vector[double], t: double) -> void:

        self._A[0] = 0.0
        self._A[1] = 0.0
        self._A[2] = 0.0

        for component in self._components:

            component.eval_A(r, t)

            self._A[0] += component._A[0]
            self._A[1] += component._A[1]
            self._A[1] += component._A[1]

    @cython.cfunc
    def eval_B(self, r: vector[double], t: double) -> void:

        self._B[0] = 0.0
        self._B[1] = 0.0
        self._B[2] = 0.0

        for component in self._components:

            component.eval_B(r, t)

            self._B[0] += component._B[0]
            self._B[1] += component._B[1]
            self._B[1] += component._B[1]


class CompositeField(_CompositeField):
    pass
