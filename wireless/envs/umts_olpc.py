"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
import math
import random
from gym import spaces, Env
import numpy as np
from numpy import linalg as la
from scipy import constants

from ..utils.misc import calculate_thermal_noise
from ..utils.prop_model import PropModel


class UlOpenLoopPowerControl(Env):
    BTS_POS = [0, 0]  # Base Transceiver Station position
    P0_TX_UE_DBM = +3  # Initial uplink transmit power of User Equipment (UE).
    UE_V = 2  # UE speed in m/s
    DT_MS = 20  # Time equivalence of one step
    SNR_MIN = -20  # Minimum measurable SNR value in dB
    SNR_MAX = 20  # Maximum measurable SNR value in dB

    def __init__(self, x_max_m=10, y_max_m=10, f_carrier_mhz=2655, bw_mhz=10, snr_tgt_db=4, t_max=512, n=3):
        """
         This environment implements a free-space scenario with a BTS at coordinates
         [0, 0] and one UE at a random location. Each step the UE moves
         linearly in a random direction with constant speed 2 m/s .
         The agent interacting with the environment is the BTS.
         On each time step the agent must select one of four possible Power
         Control (PC) commands to increase/decrease the UL transmit power. The
         objective of this power control is to measure an UL SNR as close as
         possible to the SNR target (4 dB by default). The PC commands (i.e.
         action space) are:
            Action 0 --> -1 dB
            Action 1 -->  0 dB
            Action 2 --> +1 dB
            Action 3 --> +3 dB

         As output of each step, the environment returns the following to the
         invoking agent:
            State:  Current UL SNR (single integer value between -20 and +20 with 1 dB step resolution)
            Reward:  0  if |SNR-SNR_target| <= 1 dB
                    -1  otherwise
        """
        self._seed = None
        self.x_max_m = x_max_m  # Width of 2D scenario
        self.y_max_m = y_max_m  # Height of 2D scenario
        self.bts_pos = [0, 0]
        self.f_carrier_mhz = f_carrier_mhz
        self.bw_mhz = bw_mhz
        self.snr_tgt_db = snr_tgt_db
        self.t_max = t_max
        self.propagation_model = PropModel(self.f_carrier_mhz, n=n)
        self.ue_pos = None          # To be initialized in reset
        self.v_x = None             # To be initialized in reset
        self.v_y = None             # To be initialized in reset
        self.p_tx_ue_dbm = None     # To be initialized in reset
        self.step_count = None      # To be initialized in reset
        self.state = None           # To be initialized in reset

        self.observation_space = spaces.Box(np.array([-10, -10]), np.array([+10, +10]), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)

        self.seed()
        self.reset()

    def seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        self.propagation_model.seed(seed=seed)
        self._seed = seed

    def _calculate_ul_snr(self):
        ue_bts_distance_m = la.norm(self.ue_pos - self.bts_pos)
        loss_db = self.propagation_model.get_free_space_pl_db(ue_bts_distance_m)[0]
        p_rx_dbm = self.p_tx_ue_dbm - loss_db
        n_mw = calculate_thermal_noise(self.bw_mhz)
        snr_db = p_rx_dbm - 10 * np.log10(n_mw)
        snr_db = round(snr_db)
        return max(min(snr_db, self.SNR_MAX), self.SNR_MIN)

    def render(self, mode='human'):
        pass

    def reset(self):
        self.ue_pos = np.random.rand(2) * np.array([self.x_max_m, self.y_max_m])

        theta = random.random() * 2 * constants.pi  # Random direction
        self.v_x = math.cos(theta) * self.UE_V
        self.v_y = math.sin(theta) * self.UE_V

        self.p_tx_ue_dbm = self.P0_TX_UE_DBM
        self.step_count = 0
        self.state = self._calculate_ul_snr()
        return self.state

    def _update_tx_pwr(self, action):
        if action == 0:
            self.p_tx_ue_dbm -= 1
        elif action == 2:
            self.p_tx_ue_dbm += 1
        elif action == 3:
            self.p_tx_ue_dbm += 3

    def step(self, action):
        assert self.action_space.contains(action)
        self.ue_pos += np.array([self.v_x, self.v_y]) * self.DT_MS * 1E-3  # Move UE
        self._update_tx_pwr(action)
        self.step_count += 1
        snr = self._calculate_ul_snr()
        self.state = snr  # Update state
        reward = 0 if np.abs(snr - self.snr_tgt_db) <= 1 else -1
        done = True if self.step_count >= self.t_max else False

        return self.state, reward, done, {}
