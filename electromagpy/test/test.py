import numpy as np
from scipy.constants import eV
import matplotlib.pyplot as plt
from electromagpy.particles import dueteron, triton
from electromagpy.fields.electrostatic import Orbitrap

keV = 1e3*eV
kV = 1.0e3
cm = 1.0e-2

k, Rm, C = Orbitrap.params_from_geom(100*kV, 1*cm, 6*cm, 21.5*cm)

orbitrap = Orbitrap(k, Rm, C)

print(orbitrap.V([1.0*cm, 0.0, 0.0], 0.0))

D = dueteron
D.r = [5*cm, 0.0, 3*cm]
D.v = [0.0, np.sqrt(2*10*keV/D.m), 0.0]


trajectory = orbitrap.push([D], 0.0, 1.0e-3, 1e-9)


fig, ax = plt.subplots()

ax.plot(trajectory[0, :1000, 0], trajectory[0, :1000, 1])

plt.show()

