from libcpp.vector cimport vector

cdef class _Particle:

    cdef double _q
    cdef double _m
    cdef vector[double] _r
    cdef vector[double] _v
