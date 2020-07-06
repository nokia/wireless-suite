"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
from .random_agent import RandomAgent
import numpy as np


class ProportionalFairAgent(RandomAgent):
    def __init__(self, action_space, n_ues, buffer_max_size):
        RandomAgent.__init__(self, action_space)
        self.t = 0      # Current time step

        self.K = n_ues              # Number of UEs
        self.L = buffer_max_size    # Maximum number of packets per UE buffer
        self.n = np.zeros(n_ues)    # Number of past PRB assignments for each UE

    def act(self, state, reward, done):
        s = np.reshape(state[self.K:self.K*(1 + self.L)], (self.K, self.L))  # Sizes in bits of packets in UEs' buffers
        buffer_size_per_ue = np.sum(s, axis=1)

        e = np.reshape(state[self.K*(1 + self.L):self.K*(1 + 2*self.L)], (self.K, self.L))  # Packet ages in TTIs
        o = np.max(e, axis=1)  # Age of oldest packet for each UE

        qi_ohe = np.reshape(state[self.K + 2 * self.K * self.L:5 * self.K + 2 * self.K * self.L], (self.K, 4))
        qi = np.array([np.where(r == 1)[0][0] for r in qi_ohe])  # Decode One-Hot-Encoded QIs

        # Extract packet delay budget for all UEs
        b = np.zeros(qi.shape)
        b[qi == 3] = 100
        b[qi == 2] = 150
        b[qi == 1] = 30
        b[qi == 0] = 300

        priorities = (1+o)/b * buffer_size_per_ue / (1 + self.n)

        action = np.argmax(priorities)
        self.n[action] += 1

        self.t += 1
        return action
    
class ProportionalFairChannelAwareAgent(ProportionalFairAgent):
    def __init__(self, action_space, n_ues, buffer_max_size):
        super().__init__(action_space, n_ues, buffer_max_size)
        
        self._cqi2se = [0.1523,0.2344,0.3770,0.6016,0.8770,1.1758,1.4766,1.9141,2.4063,2.7305,3.3223,3.9023,4.5234,5.1152,5.5547,9.6]
        

    def act(self, state, reward, done):
        s = np.reshape(state[self.K:self.K*(1 + self.L)], (self.K, self.L))  # Sizes in bits of packets in UEs' buffers
        buffer_size_per_ue = np.sum(s, axis=1)

        e = np.reshape(state[self.K*(1 + self.L):self.K*(1 + 2*self.L)], (self.K, self.L))  # Packet ages in TTIs
        o = np.max(e, axis=1)  # Age of oldest packet for each UE

        qi_ohe = np.reshape(state[self.K + 2 * self.K * self.L:5 * self.K + 2 * self.K * self.L], (self.K, 4))
        qi = np.array([np.where(r == 1)[0][0] for r in qi_ohe])  # Decode One-Hot-Encoded QIs

        # Extract packet delay budget for all UEs
        b = np.zeros(qi.shape)
        b[qi == 3] = 100
        b[qi == 2] = 150
        b[qi == 1] = 30
        b[qi == 0] = 300

        cqi = state[0:self.K]
        se = np.zeros(shape=(self.K,))
        for i in range(16):
            se[cqi == i] = self._cqi2se[i]
        priorities = (1+o)/b * buffer_size_per_ue * se

        action = np.argmax(priorities)
        self.n[action] += 1

        self.t += 1
        return action
