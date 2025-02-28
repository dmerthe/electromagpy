# distutils: language=c++
# Electromagnetic fields, either explicitly defined or generated from bodies
from libc.stdio cimport printf
from libcpp.vector cimport vector
from field cimport field as cfield
cdef class Vacuum:

    cdef cfield *field

    def __cinit__(self):

        print('Initializing vacuum field values')
        # seg fault here
        self.field[0].V = 0.0
        self.field[0].E = [0.0, 0.0, 0.0]
        self.field[0].A = [0.0, 0.0, 0.0]
        self.field[0].B = [0.0, 0.0, 0.0]

        print('Initializing vacuum eval_V')

        self.field[0].eval_V = self.eval_V

    @staticmethod
    cdef void *eval_V(cfield *field, vector[double]r, double t) except? NULL:
        print('In eval_V')
        field[0].V = 0.0

    cpdef double V(self, vector[double] r, double t):

        print('Evaluating V')
        self.field[0].eval_V(self.field, r, t)

        print('Returning V')
        return self.field[0].V
