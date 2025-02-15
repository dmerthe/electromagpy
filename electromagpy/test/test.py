import numpy as np
from electromagpy.particles import dueteron, triton
from electromagpy.fields.electrostatic import UniformE

D = dueteron
T = triton
print(D.q)
field = UniformE(1.0, 1.0, 0.0)

trajectory = field.push([D, T], 0.0, 1.0e-2, 1.0e-9)

print(trajectory)
