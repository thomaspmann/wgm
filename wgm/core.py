"""
Calculate the propagation constant of a tapered fiber and a whispering-gallery-mode.

[1] Phase-matched excitation of whispering-gallery-mode resonances by a fiber taper, Knight, J. C.
[2] Asymptotic expansion of morphological resonance frequencies in Mie scattering, S. Schiller

Note: notation used to denote mode numbers differ between [1] and [2]:

    ==============================
    Mode Number |    [1]  |   [2]
    ==============================
    Radial      |     n    |   l
    Angular     |     l    |   m
    Azimuthal   |     m    |   n
    
This code uses the definitions in [1] for consistency.
"""

import numpy as np
from scipy import special


def beta_fiber(rho, lam0, N):
    """
    Calculate the propagation constant of a tapered fiber Eq.1 in [1].
    Note: rho needs to be between 1.4um and 3um for the approximation to hold.
    :param rho: Radius of the fiber 
    :param lam0: Free space wavelength
    :param N: Refractive index of the fiber
    :return: Propagation constant
    """
    k = 2 * np.pi / lam0
    return np.sqrt((k * N) ** 2 - (2.405 / rho) ** 2)


def beta_wgm(lam0, l, x):
    """
    Calculate the propagation constant of the WGM mode using approximation from [1]
    :param lam0: Free space wavelength
    :param l: Angular mode number 
    :param x: size parameter of nlm mode
    :return: Mode propagation constant, beta
    """
    # Free space propagation constant
    k = 2 * np.pi / lam0
    return k * l / x


class WGM:
    def __init__(self, na, ns, mode):
        """
        Initialise the Whispering Gallery Mode object. Equations from [2]
        :param na: Refractive index of medium surrounding sphere
        :param ns: Refractive index of the sphere
        :param mode: Either 'TE' or 'TM'
        """
        self.na = na
        self.ns = ns
        self.m_ratio = ns / na
        self.mode = mode

    def calc_p(self):
        if self.mode == 'TE':
            return 1
        if self.mode == 'TM':
            return 1 / self.m_ratio ** 2

    @staticmethod
    def calc_zeta(n):
        """Calculate the nth zero of the Airy function."""
        return special.ai_zeros(n)[0][-1]

    def e_prime(self, k, zeta):
        m = self.m_ratio
        if k == 4:
            return (-8 + 12 * m ** 4 + m ** 8) / m ** 8
        elif k == 5:
            return 7000 * m ** (-6) * (-28 - m ** 2 + 56 * m ** 4 - 16 * m ** 2 - 7 * m ** 8 + 2 * m ** 10)
        elif k == 6:
            a = 5 * (-200 - 32 * m ** 2 + 526 * m ** 4 - 226 * m ** 6 - 99 * m ** 8 + 62 * m ** 10 + 4 * m ** 12)
            b = 2 * (-400 + 272 * m ** 2 + 744 * m ** 4 - 424 * m ** 6 - 2 * m ** 10 + m ** 12) * zeta ** 3
            return m ** (-8) * (a + b)
        elif k == 7:
            return -269500 * m ** (-8) * (
            -232 + 160 * m ** 2 + 543 * m ** 4 - 447 * m ** 6 - 186 * m ** 8 + 165 * m ** 10 - 15 * m ** 12 + 4 * m ** 14)
        elif k == 8:
            a = -10 * (
            -459200 + 286000 * m ** 2 + 1360312 * m ** 4 - 1305476 * m ** 6 - 433952 * m ** 8 + 717562 * m ** 10 - 209039 * m ** 12
            - 21542 * m ** 14 + 7060 * m ** 16)
            b = 3 * (
            33600 - 441600 * m ** 2 - 626496 * m ** 4 + 891008 * m ** 6 + 306416 * m ** 8 - 505696 * m ** 10 - 72488 * m ** 12
            - 7664 * m ** 14 + 2395 * m ** 16) * zeta ** 3
            return m ** (-10) * zeta * (a + b)

    def e(self, k, zeta):
        if self.mode == 'TE':
            return 0
        else:
            m = self.m_ratio
            return (m ** 2 - 1) * self.e_prime(k, zeta)

    def d(self, k, zeta):
        p = self.calc_p()
        m = self.m_ratio

        if k == 0:
            return -p
        elif k == 1:
            return 2 ** (1 / 3) * 3 * (m ** 2 - 1) * zeta ** 2 / (20 * m)
        elif k == 2:
            return -2 ** (2 / 3) * m ** 2 * p * (-3 + 2 * p ** 2) * zeta / 6
        elif k == 3:
            num = 350 * m ** 4 * (1 - p) * p * (-1 + p + p ** 2) + (m ** 2 - 1) ** 2 * (10 - zeta ** 3)
            den = 700 * m
            return num / den
        elif k == 4:
            num = -2 ** (1 / 3) * m ** 2 * zeta ** 2 * (4 - m ** 2 + self.e(4, zeta))
            den = 20
            return num / den
        elif k == 5:
            num = zeta * (
            40 * (-1 + 3 * m ** 2 - 3 * m ** 4 + 351 * m ** 6) - 479 * (m ** 2 - 1) ** 3 * zeta ** 3 - self.e(5, zeta))
            den = 2 ** (4 / 3) * 6300 * m
            return num / den
        elif k == 6:
            num = 5 * m ** 2 * (-13 - 16 * m ** 2 + 4 * m ** 4) + 2 * m ** 2 * (
            128 - 4 * m ** 2 + m ** 4) * zeta ** 3 - self.e(6, zeta)
            return num / 1400
        elif k == 7:
            a = 100 * (-551 + 2204 * m ** 2 - 3306 * m ** 4 - 73256 * m ** 6 + 10229 * m ** 8)
            b = - 20231 * (m ** 2 - 1) ** 4 * zeta ** 3
            num = zeta ** 2 * (a + b + self.e(7, zeta))
            den = 2 ** (2 / 3) * 16170000 * m
            return num / den
        elif k == 8:
            a = 10 * (11082 + 44271 * m ** 2 - 288 * m ** 4 + 7060 * m ** 6)
            b = -3 * (52544 + 48432 * m ** 2 - 11496 * m ** 4 + 2395 * m ** 6) * zeta ** 3
            brackets = a + b
            num = m ** 2 * zeta * brackets + self.e(8, zeta)
            den = 2 ** (10 / 3) * 141750
            return num / den
        else:
            raise ValueError('k must be an integer between 0 and 8')

    def calc_size_param(self, n, m):
        """
        :param n: Radial mode number (denoted l in [2])
        :param m: Azimuthal mode number (denoted n in [2])
        :return: The resonance size parameter, Eq.1 in [2]
        """
        v = m + 1 / 2
        m_ratio = self.m_ratio
        zeta = self.calc_zeta(n)

        k_sum = 0
        for k in np.arange(0, 9):
            den = v ** (k / 3) * (m_ratio ** 2 - 1) ** ((k + 1) / 2)
            k_sum += self.d(k, zeta) / den
        return v / m_ratio - (zeta / m_ratio) * (v / 2) ** (1 / 3) + k_sum

    def calc_angular_resonance_frequency(self, x, r):
        """
        :param x: The resonance size parameter, Eq.1 in [2] of the n, m mode
        :param r: Radius of the sphere
        :return: The angular resonance frequency
        """
        from scipy.constants import c
        return x * c / (self.na * r)
