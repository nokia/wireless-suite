"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
from wireless.agents.random_agent import RandomAgent
import itertools
import heapq 
import numpy as np


class NomaULProportionalFairChannelAwareAgent(RandomAgent):
    def __init__(self, action_space, n_ues, n_ues_per_prb, buffer_max_size, n_mw, sinr_coeff):
        RandomAgent.__init__(self, action_space)
        self.t = 0      # Current time step

        self.K = n_ues              # Number of UEs
        self.L = buffer_max_size    # Maximum number of packets per UE buffer
        self.M = n_ues_per_prb      # Maximum number of users multiplexed on a PRB 
        self.n = np.zeros(n_ues)    # Number of past PRB assignments for each UE
        self.n_mw = n_mw            # Thermal noise in mW
        self.sinr_coeff = sinr_coeff # Rho coefficient to map SINR to spectral efficient.
        
        self._cqi2se = [0.1523,0.2344,0.3770,0.6016,0.8770,1.1758,1.4766,1.9141,2.4063,2.7305,3.3223,3.9023,4.5234,5.1152,5.5547,9.6]
        
        interference_dbm = -105  # Constant interference level throughout the coverage area
        self._interference_mw = 10 ** (interference_dbm / 10)
        self._cqi2rx_pwr_mw = (np.power(2,self._cqi2se)-1) * (self.n_mw+self._interference_mw) * self.sinr_coeff
        
        # All possible allocations: all permutations of self.K our of self.M UEs
        self._permutations = list(itertools.permutations(range(self.K),self.M))
        # WSR of each permutation stored as a heapq
        # It is re-computed entirely when p == 0, and updated lazily at each other step
        self._permutations_wsr = None

    def act(self, state, reward, done):
        s = np.reshape(state[self.K:self.K*(1 + self.L)], (self.K, self.L))  # Sizes in bits of packets in UEs' buffers
        buffer_size_per_ue = np.sum(s, axis=1)

        e = np.reshape(state[self.K*(1 + self.L):self.K*(1 + 2*self.L)], (self.K, self.L))  # Packet ages in TTIs
        o = np.max(e, axis=1)  # Age of oldest packet for each UE

        qi_ohe = np.reshape(state[self.K + 2 * self.K * self.L:5 * self.K + 2 * self.K * self.L], (self.K, 4))
        qi = np.array([np.where(r == 1)[0][0] for r in qi_ohe])  # Decode One-Hot-Encoded QIs

        p = state[-1]

        # Extract packet delay budget for all UEs
        b = np.zeros(qi.shape)
        b[qi == 3] = 100
        b[qi == 2] = 150
        b[qi == 1] = 30
        b[qi == 0] = 300

        w = (1+o)/b * buffer_size_per_ue        # Weight of each UE in the PF scheduler
        cqi = state[0:self.K]                   # CQI of each UE
        rx_pwr_mw = np.zeros(shape=(self.K,))   # Receive power of each UE
        for i in range(16):
            rx_pwr_mw[cqi == i] = self._cqi2rx_pwr_mw[i]

        # Weighted sum-rate maximization considering w and rx_pwr_mw:
        # Find the M UEs out of K that maximize sum w[i]*se[i]      
        if p == 0 or self._permutations_wsr == None: # Re-compute entirely self._permutations_wsr
            self._permutations_wsr = []
            heapq.heapify(self._permutations_wsr)
            for index in range(len(self._permutations)):
                permutation = self._permutations[index]
                cumulated_rx_pwr_mw = 0
                wsr = 0
                for pos in range(self.M-1, -1, -1):
                    # ue decoded in pos-th order
                    ue = permutation[pos]
                    # SINR taking into account the intereference from other UEs superposed on the same PRB
                    sinr = rx_pwr_mw[ue] / (self.n_mw + self._interference_mw + cumulated_rx_pwr_mw) 
                    wsr += w[ue]*np.log2(1 + sinr / self.sinr_coeff)  # DL spectral efficiency in bps/Hz
                    # Store the current UE rx_pwr_mw as interference
                    cumulated_rx_pwr_mw += rx_pwr_mw[ue]
                heapq.heappush(self._permutations_wsr,(-wsr,index))
            
            max_wsr, max_wsr_index = heapq.heappop(self._permutations_wsr)
            heapq.heappush(self._permutations_wsr,(max_wsr,max_wsr_index))
        # When p!=0, perform lazy update since:
        # 1) The weights w has only changed (decreased) for up to self.M UEs
        # 2) In addition, the spectral efficiency have not changed 
        else:   
            while True:
                old_wsr, max_wsr_index = heapq.heappop(self._permutations_wsr)
                old_wsr = -old_wsr
                # Compute the new WSR
                permutation = self._permutations[max_wsr_index]
                cumulated_rx_pwr_mw = 0
                new_wsr = 0
                for pos in range(self.M-1, -1, -1):
                    # ue decoded in pos-th order
                    ue = permutation[pos]
                    # SINR taking into account the intereference from other UEs superposed on the same PRB
                    sinr = rx_pwr_mw[ue] / (self.n_mw + self._interference_mw + cumulated_rx_pwr_mw) 
                    new_wsr += w[ue]*np.log2(1 + sinr / self.sinr_coeff)  # DL spectral efficiency in bps/Hz
                    # Store the current UE rx_pwr_mw as interference
                    cumulated_rx_pwr_mw += rx_pwr_mw[ue]
                assert old_wsr >= new_wsr, "The WSR should only decrease in the lazy updates"
                if old_wsr == new_wsr: 
                    # This WSR has not changed -> it is still the highest value
                    # Push it back in the heapq and terminate the while loop
                    heapq.heappush(self._permutations_wsr,(-old_wsr,max_wsr_index))
                    break
                else: 
                    # Otherwise, we update its WSR info and push it in the heapq
                    # The while loop continues
                    heapq.heappush(self._permutations_wsr,(-new_wsr,max_wsr_index))
        
        action = list(self._permutations[max_wsr_index])
        self.n[action] += 1
        self.t += 1
        return action