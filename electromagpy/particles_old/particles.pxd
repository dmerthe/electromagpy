from libcpp.vector cimport vector

cdef class _Particle:

    cdef double _q  # electric charge
    cdef double _m  # mass

    cdef vector[double] _r  # coordinate vector [x, y, z]
    cdef vector[double] _v  # velocity vector [v_x, v_y, v_z]

    cdef double _PE  # potential energy in joules; set by interacting field
    cdef vector[double] _F  # force in Newtons; set by interacting field
