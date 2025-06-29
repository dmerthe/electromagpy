from libcpp.vector cimport vector
from electromagpy.fields.field cimport FieldStruct, Field

cdef class Field:

    cdef double _V = 0.0
    cdef vector[double] _E = [0.0, 0.0, 0.0]
    cdef vector[double] _A = [0.0, 0.0, 0.0]
    cdef vector[double] _B = [0.0, 0.0, 0.0]

    cdef __cinit__(self):
        



    cdef void eval_V(vector[double] r, double t):
        


    cdef void eval_E(vector[double] r, double t)
    cdef void eval_A(vector[double] r, double t)
    cdef void eval_B(vector[double] r, double t)

    cpdef float V(vector[double] r, double t)
    cpdef np.ndarray E(vector[double] r, double t)
    cpdef np.ndarray A(vector[double] r, double t)
    cpdef np.ndarray B(vector[double] r, double t)