from libcpp.vector cimport vector
import numpy as np

ctypedef void (*eval_V_component)(double V, vector[double] r, double t)
ctypedef void (*eval_E_component)(vector[double] E, vector[double] r, double t)
ctypedef void (*eval_A_component)(vector[double] A, vector[double] r, double t)
ctypedef void (*eval_B_component)(vector[double] B, vector[double] r, double t)

ctypedef struct FieldStruct:

    double V
    vector[double] E
    vector[double] A
    vector[double] B

    vector[eval_V_component] eval_V_components
    vector[eval_E_component] eval_E_components
    vector[eval_A_component] eval_A_components
    vector[eval_B_component] eval_B_components

    void (*eval_V)(FieldStruct fieldstruct, vector[double] r, double t)
    void (*eval_E)(FieldStruct fieldstruct, vector[double] r, double t)
    void (*eval_A)(FieldStruct fieldstruct, vector[double] r, double t)
    void (*eval_B)(FieldStruct fieldstruct, vector[double] r, double t)


cdef class Field:

    cdef FieldStruct fieldstruct

    cdef double _V
    cdef vector[double] _E
    cdef vector[double] _A
    cdef vector[double] _B

    cdef void eval_V(vector[double] r, double t)
    cdef void eval_E(vector[double] r, double t)
    cdef void eval_A(vector[double] r, double t)
    cdef void eval_B(vector[double] r, double t)

    cpdef float V(vector[double] r, double t)
    cpdef np.ndarray E(vector[double] r, double t)
    cpdef np.ndarray A(vector[double] r, double t)
    cpdef np.ndarray B(vector[double] r, double t)
