from libcpp.vector cimport vector

cdef double dot(vector[double] v1, vector[double] v2)
cdef double cross(int i, vector[double] v1, vector[double] v2)

cdef class _Field:

    # pre-initialized field buffers
    cdef double _V  # electric potential
    cdef vector[double] _E  # electric field
    cdef vector[double] _A  # magnetic potential
    cdef vector[double] _B # magnetic field

    cdef void eval_V(self, vector[double] r, double t)
    cdef void eval_E(self, vector[double] r, double t)
    cdef void eval_A(self, vector[double] r, double t)
    cdef void eval_B(self, vector[double] r, double t)

    cpdef double V(self, vector[double] r, double t)
    cpdef vector[double] E(self, vector[double] r, double t)
    cpdef vector[double] A(self, vector[double] r, double t)
    cpdef vector[double] B(self, vector[double] r, double t)

    cdef void eval_F(self, vector[double] r, vector[double] v, double t, double q, double[3] F)

    cpdef push(self, list particles, double t1, double t2, double dt)
