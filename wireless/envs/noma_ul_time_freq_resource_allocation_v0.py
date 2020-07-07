"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""

from .time_freq_resource_allocation_v0 import *


class NomaULTimeFreqResourceAllocationV0(TimeFreqResourceAllocationV0):

    def __init__(self, n_ues=32, n_prbs=25, n_ues_per_prb=2, buffer_max_size=32, eirp_dbm=13, f_carrier_mhz=2655,
                 max_pkt_size_bits=41250, it=10, t_max=65536):
        super().__init__(n_ues, n_prbs, buffer_max_size, eirp_dbm, f_carrier_mhz, max_pkt_size_bits, it, t_max)

        self.M = n_ues_per_prb # Maximum number of users multiplexed on a PRB 
        self.action_space = spaces.MultiDiscrete([self.K+1]*self.M)

    def reset(self):
        self.rx_pwr_mw = np.zeros(shape=(self.K,))  # Received powers at the current time step

        return super().reset()

    def step(self, action):
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        # Execute action from the last to the first decoded UE
        cumulated_rx_pwr_mw = 0
        # Convert action to a numpy array in case it is a list
        action = np.array(action)
        # Only keep the unique values (UEs) in action
        _, indices = np.unique(action, return_index=True)
        action_tmp = np.ones(self.M,dtype=np.uint32)*self.K
        action_tmp[indices] = action[indices]
        action = action_tmp
        for dim in range(self.M-1, -1, -1):
            ue_action = action[dim]
            # First check if ue_action is not NOOP (no UE selected at this order)
            # NOOP is defined as := self.K, while the UE are 0, ... ,self.K-1
            if ue_action != self.K:
                if np.sum(self.s[ue_action, :]) > 0:  # If packets exist in UE's buffer
                    # Find oldest packet in UE's buffer
                    mask = (self.s[ue_action, :] > 0)
                    subset_idx = np.argmax(self.e[ue_action, mask])
                    l_old = np.arange(self.L)[mask][subset_idx]
        
                    assert self.s[ue_action, l_old] > 0, f"t={self.t}. Oldest packet has size {self.s[ue_action, l_old]} " +\
                                                      f"and age {self.e[ue_action, l_old]}. " +\
                                                      f"User has {np.sum(self.s[ue_action, :])} bits in buffer."  # Sanity check
                    
                    interference_dbm = -105  # Constant interference level throughout the coverage area
                    interference_mw = 10 ** (interference_dbm / 10)
                    sinr = self.rx_pwr_mw[ue_action] / (self.n_mw + interference_mw + cumulated_rx_pwr_mw) # SINR taking into account the intereference from other UEs superposed on the same PRB
                    se = np.log2(1 + sinr / self.SINR_COEFF)  # DL spectral efficiency in bps/Hz
                    se = np.clip(se, 0, 9.6)  # Define an upper bound for the spectral efficiency.
                    tx_data_bits = floor(se * self.bw_mhz / self.Nf * 1E3)  # Bits that can be transmitted
                    # Store the current UE rx_pwr_mw as interference
                    cumulated_rx_pwr_mw += self.rx_pwr_mw[ue_action]
                    while tx_data_bits > 0 and self.s[ue_action, l_old] > 0:  # While there are packets & available capacity
                        if tx_data_bits >= self.s[ue_action, l_old]:  # Full packet transmission
                            tx_data_bits -= self.s[ue_action, l_old]
                            self.s[ue_action, l_old] = 0
                            self.e[ue_action, l_old] = 0
                            l_old = np.argmax(self.e[ue_action, :])  # Find oldest packet in UE's buffer
                        else:  # Partial packet transmission
                            self.s[ue_action, l_old] -= tx_data_bits
                            break

        reward = 0
        self.t += 1  # Update time-step
        self.p = self.t % self.Nf  # Update PRB counter
        if self.p == 0:
            reward = self._calculate_reward()
            self.tti += 1  # Update TTI counter
            self.e[self.s > 0] += 1  # Age buffer packets
            self._generate_traffic()
            self._move_ues()
            self._recalculate_rf()

        self._update_state()
        done = bool(self.t >= self.t_max)
        return np.array(self.state), reward, done, {}

    def _calculate_spectral_efficiency(self, rx_pwr_dbm):
        interference_dbm = -105  # Constant interference level throughout the coverage area

        p_mw = (10 ** (rx_pwr_dbm / 10))  # Rx power in mw
        self.rx_pwr_mw = p_mw
        interference_mw = 10 ** (interference_dbm / 10)

        sinr = p_mw / (self.n_mw + interference_mw)
        se = np.log2(1 + sinr / self.SINR_COEFF)  # DL spectral efficiency in bps/Hz

        self.spectral_efficiency = np.clip(se, 0, 9.6)  # Define an upper bound for the spectral efficiency.