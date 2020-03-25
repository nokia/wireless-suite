"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
import numpy as np
from scipy import constants


class PropModel:
    """
    Propagation Model class
    It can be used to define more complex prop models in the future
    """

    def __init__(self, f_mhz, n=2):
        self.f_mhz = f_mhz
        self.n = n  # Attenuation exponent

    def get_free_space_pl_db(self, d_m, shadowing_db=0):
        noise = np.random.normal(scale=shadowing_db, size=d_m.size)
        return self.n * 10 * np.log10(4 * constants.pi * d_m * self.f_mhz * 1E6 / constants.c) + noise

    def seed(self, seed=0):
        np.random.seed(seed)
