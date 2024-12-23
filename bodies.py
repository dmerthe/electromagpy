# Physical bodies that generate electromagnetic fields

import cython

class Body:
    """
    Base class for all bodies.

    Establishes the electric potential (V), electric field (E), magnetic
    potential (A) and magnetic field (B).
    """

    _V: cython.float
    _E: cython.float[3]
    _A: cython.float[3]
    _B: cython.float[3]

    _V = 0.0
    _E = 0.0, 0.0, 0.0
    _A = 0.0, 0.0, 0.0
    _B = 0.0, 0.0, 0.0

    def V(self, r: cython.float[3], t: cython.float):
        return self._V

    def E(self, r: cython.float[3], t: cython.float):
        return self._E

    def A(self, r: cython.float[3], t: cython.float):
        return self._A

    def B(self, r: cython.float[3], t: cython.float):
        return self._B


class Orbitrap(Body):
    """
    Electrostatic orbitrap, can be defined by its bias (anode-cathode), rc
    (inner/cathode radius), ra (outer/anode radius) and Ra (radius of
    curvature of the anode at z = 0), or by the traditional orbitrap potential
    parameters, k, Rm and C.
    """

    def __init__(
            self, bias: cython.float,
            rc: cython.float = None, ra=None, Ra=None,
            k=None, Rm=None, C=None
    ):

        self.bias = bias

        if rc is not None and ra is not None and Ra is not None:

            self.rc = rc
            self.ra = ra
            self.Ra = Ra

            # Traditional orbitr
            self.Rm = Rm = np.sqrt(ra ** 2 + 2 * ra * Ra)
            self.k = k = 4 * bias / (2 * Rm ** 2 * np.log(ra / rc) - ra ** 2 + rc ** 2)
            self.C = (k / 2) * (ra ** 2 / 2 + Rm ** 2 * np.log(Rm / ra))
        elif k is not None and Rm is not None and C is not None:
            self.Rm = Rm
            self.k = k
            self.C = C
        else:
            raise ValueError("Either (rc, ra, Ra) or (k, Rm, C) must be defined)")

    def V(self, r: Union[np.ndarray, list, tuple], t: float):

        k = self.k
        Rm = self.Rm
        C = self.C

        rr = np.sqrt(r[0] * r[0] + r[1] * r[1])
        z = r[2]

        return 0.5*k * ((z*z - 0.5 * rr*rr) + Rm*Rm*np.log(rr / Rm)) + C

    def E(self, r: Union[np.ndarray, list, tuple], t: float):

        x, y, z = r
        k, Rm = self.k, self.Rm
        r2 = x*x + y*y

        factor = 0.5 * k * (1.0 - Rm*Rm / r2)

        self._E[0] = x * factor
        self._E[1] = y * factor
        self._E[2] = -k * z

        return self._E

    def calc_axial_freq(self, q: float, m: float):
        """
        Calculate the axial oscillation frequency of a particle
        """

        return np.sqrt(self.k * q / m) / (2 * np.pi)

    def calc_circ_orb(self, q: float, m: float, Lz: float):
        """
        Calculate the equivalent circular orbit at z = 0 for a particle based on azimuthal angular momentum
        """

        k = self.k
        Rm = self.Rm
        w0 = self.calc_axial_freq(q,m)

        Lhat = Lz / (m*Rm*w0**2)

        return Rm * np.sqrt(0.5-0.5*np.sqrt(1-8*Lhat**2))
