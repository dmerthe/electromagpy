
cdef class _Field:

    cdef double _V  # electric potential
    cdef double[3] _E  # electric field
    cdef double[3] _A  # magnetic potential
    cdef double[3] _B # magnetic field

    # cdef void _eval_V(self, double[3] r, double t)
    # cdef void _eval_E(self, double[3] r, double t)
    # cdef void _eval_A(self, double[3] r, double t)
    # cdef void _eval_B(self, double[3] r, double t)
    #
    # cdef double V(self, double[3] r, double t)
    # cdef double[:] E(self, double[3] r, double t)
    # cdef double[:] A(self, double[3] r, double t)
    # cdef double[:] B(self, double[3] r, double t)

cdef type py_field_factory(type cfieldclass)
