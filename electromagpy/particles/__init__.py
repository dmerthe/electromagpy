from scipy.constants import physical_constants, m_e, m_p, e
from electromagpy._particles._particles import Particle

__all__ = [
    "Particle",

    "e",

    "proton", "dueteron", "triton",

    "boron11",
]

Da = physical_constants['atomic mass constant'][0]  # Dalton, unit of mass

electron = Particle(e, m_e)  # electron

proton = Particle(e, m_p)  # proton
dueteron = Particle(e, 2.01410177811 * Da)  # deuterium
triton = Particle(e, 3.01604928 * Da)  # tritium

boron11 = Particle(e, 11.009306 * Da)  # boron-11
