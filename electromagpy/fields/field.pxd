from libcpp.vector cimport vector

ctypedef struct field:

    double V
    vector[double] E
    vector[double] A
    vector[double] B

    void *eval_V(field *field, vector[double] r, double t)
    void *eval_E(field *field, vector[double] r, double t)
    void *eval_A(field *field, vector[double] r, double t)
    void *eval_B(field *field, vector[double] r, double t)
