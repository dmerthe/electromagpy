from libcpp.vector cimport vector

cdef class _Field:

    cdef double _V  # electric potential
    cdef vector[double] _E  # electric field
    cdef vector[double] _A  # magnetic potential
    cdef vector[double] _B # magnetic field

    cpdef double V(self, vector[double] r, double t)
    cpdef vector[double] E(self, vector[double] r, double t)
    cpdef vector[double] A(self, vector[double] r, double t)
    cpdef vector[double] B(self, vector[double] r, double t)
