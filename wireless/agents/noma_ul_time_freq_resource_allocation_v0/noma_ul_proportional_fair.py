"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
from wireless.agents.random_agent import RandomAgent
from wireless.agents.time_freq_resource_allocation_v0.proportional_fair import ProportionalFairAgent
import itertools
import heapq 
import numpy as np


class NomaULProportionalFairChannelAwareAgent(RandomAgent):
    CQI2SE = [0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141, 2.4063, 2.7305, 3.3223, 3.9023, 4.5234,
              5.1152, 5.5547, 9.6]

    def __init__(self, action_space, n_ues, n_ues_per_prb, buffer_max_size, n_mw, sinr_coeff):
        RandomAgent.__init__(self, action_space)
        self.t = 0      # Current time step

        self.K = n_ues              # Number of UEs
        self.L = buffer_max_size    # Maximum number of packets per UE buffer
        self.M = n_ues_per_prb      # Maximum number of users multiplexed on a PRB 
        self.n = np.zeros(n_ues)    # Number of past PRB assignments for each UE
        self.n_mw = n_mw            # Thermal noise in mW
        self.sinr_coeff = sinr_coeff  # Rho coefficient to map SINR to spectral efficient.
        
        interference_dbm = -105  # Constant interference level throughout the coverage area
        self._interference_mw = 10 ** (interference_dbm / 10)
        self._cqi2rx_pwr_mw = (np.power(2, self.CQI2SE)-1) * (self.n_mw+self._interference_mw) * self.sinr_coeff
        
        # All possible allocations: all permutations of self.K our of self.M UEs
        self._permutations = list(itertools.permutations(range(self.K), self.M))
        # WSR of each permutation stored as a heapq
        # It is re-computed entirely when p == 0, and updated lazily at each other step
        self._permutations_wsr = None

    def _calculate_wsr(self, perm_idx, rx_pwr_mw, w):
        permutation = self._permutations[perm_idx]
        cumulated_rx_pwr_mw = 0
        wsr = 0
        for pos in range(self.M - 1, -1, -1):
            # ue decoded in pos-th order
            ue = permutation[pos]
            # SINR taking into account the interference from other UEs superposed on the same PRB
            sinr = rx_pwr_mw[ue] / (self.n_mw + self._interference_mw + cumulated_rx_pwr_mw)
            wsr += w[ue] * np.log2(1 + sinr / self.sinr_coeff)  # DL spectral efficiency in bps/Hz
            # Store the current UE rx_pwr_mw as interference
            cumulated_rx_pwr_mw += rx_pwr_mw[ue]
        return wsr

    def act(self, state, reward, done):
        o, cqi, b, buffer_size_per_ue = ProportionalFairAgent.parse_state(state, self.K, self.L)

        p = state[-1]

        w = (1+o)/b * buffer_size_per_ue        # Weight of each UE in the PF scheduler
        rx_pwr_mw = np.zeros(shape=(self.K,))   # Receive power of each UE
        for i in range(16):
            rx_pwr_mw[cqi == i] = self._cqi2rx_pwr_mw[i]

        # Weighted sum-rate maximization considering w and rx_pwr_mw:
        # Find the M UEs out of K that maximize sum w[i]*se[i]      
        if p == 0 or self._permutations_wsr is None:  # Re-compute entirely self._permutations_wsr
            self._permutations_wsr = []
            heapq.heapify(self._permutations_wsr)
            for index in range(len(self._permutations)):
                wsr = self._calculate_wsr(index, rx_pwr_mw, w)
                heapq.heappush(self._permutations_wsr, (-wsr, index))
            
            max_wsr, max_wsr_index = heapq.heappop(self._permutations_wsr)
            heapq.heappush(self._permutations_wsr, (max_wsr, max_wsr_index))
        # When p!=0, perform lazy update since:
        # 1) The weights w has only changed (decreased) for up to self.M UEs
        # 2) In addition, the spectral efficiency have not changed 
        else:   
            while True:
                old_wsr, max_wsr_index = heapq.heappop(self._permutations_wsr)
                old_wsr = -old_wsr
                new_wsr = self._calculate_wsr(max_wsr_index, rx_pwr_mw, w)  # Compute the new WSR
                assert old_wsr >= new_wsr, "The WSR should only decrease in the lazy updates"
                if old_wsr == new_wsr: 
                    # This WSR has not changed -> it is still the highest value
                    # Push it back in the heapq and terminate the while loop
                    heapq.heappush(self._permutations_wsr, (-old_wsr, max_wsr_index))
                    break
                else: 
                    # Otherwise, we update its WSR info and push it in the heapq
                    # The while loop continues
                    heapq.heappush(self._permutations_wsr, (-new_wsr, max_wsr_index))
        
        action = list(self._permutations[max_wsr_index])
        self.n[action] += 1
        self.t += 1
        return action
