from scipy.constants import physical_constants, m_e, m_p, e
from electromagpy.particles._particles import Particle

__all__ = [
    "Particle",

    "e",

    "p", "D", "T",

    "B11",
]

Da = physical_constants['atomic mass constant'][0]  # Dalton, unit of mass

e = Particle(e, m_e)  # electron

p = Particle(e, m_p)  # proton
D = Particle(e, 2.01410177811 * Da)  # deuterium
T = Particle(e, 3.01604928 * Da)  # tritium

B11 = Particle(e, 11.009306 * Da)  # boron-11
