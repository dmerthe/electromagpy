import numpy as np
from electromagpy.fields.electrostatic import Orbitrap
from electromagpy.fields.magnetostatic import UniformB

# k, Rm, C = Orbitrap.params_from_geom(100.0e3, 1.0e-2, 6.0e-2, 21.5e-2)
# print(k, Rm, C)
# field = Orbitrap(k, Rm, C)

field = UniformB(0.0, 0.0, 1.0)

r = np.array([0.0, 1.0, 0.0])
t = 0.0

print(field.V(r, t))
print(field.E(r, t))
print(field.A(r, t))
print(field.B(r, t))
