"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
from wireless.agents.random_agent import RandomAgent
import numpy as np


class ProportionalFairAgent(RandomAgent):
    def __init__(self, action_space, n_ues, buffer_max_size):
        RandomAgent.__init__(self, action_space)
        self.t = 0      # Current time step

        self.K = n_ues              # Number of UEs
        self.L = buffer_max_size    # Maximum number of packets per UE buffer
        self.n = np.zeros(n_ues)    # Number of past PRB assignments for each UE

    def _calculate_priorities(self, cqi, o, b, buffer_size_per_ue):
        priorities = (1 + o) / b * buffer_size_per_ue / (1 + self.n)
        return priorities

    @staticmethod
    def parse_state(state, num_ues, max_pkts):
        s = np.reshape(state[num_ues:num_ues * (1 + max_pkts)], (num_ues, max_pkts))  # Sizes in bits of packets in UEs' buffers
        buffer_size_per_ue = np.sum(s, axis=1)

        e = np.reshape(state[num_ues * (1 + max_pkts):num_ues * (1 + 2 * max_pkts)], (num_ues, max_pkts))  # Packet ages in TTIs
        o = np.max(e, axis=1)  # Age of oldest packet for each UE

        cqi = state[0:num_ues]

        qi_ohe = np.reshape(state[num_ues + 2 * num_ues * max_pkts:5 * num_ues + 2 * num_ues * max_pkts], (num_ues, 4))
        qi = np.array([np.where(r == 1)[0][0] for r in qi_ohe])  # Decode One-Hot-Encoded QIs

        # Extract packet delay budget for all UEs
        b = np.zeros(qi.shape)
        b[qi == 3] = 100
        b[qi == 2] = 150
        b[qi == 1] = 30
        b[qi == 0] = 300

        return o, cqi, b, buffer_size_per_ue

    def act(self, state, reward, done):
        o, cqi, b, buffer_size_per_ue = self.parse_state(state, self.K, self.L)

        priorities = self._calculate_priorities(cqi, o, b, buffer_size_per_ue)

        action = np.argmax(priorities)
        self.n[action] += 1

        self.t += 1
        return action


class ProportionalFairChannelAwareAgent(ProportionalFairAgent):
    CQI2SE = [0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141, 2.4063, 2.7305, 3.3223, 3.9023, 4.5234,
              5.1152, 5.5547, 9.6]

    def __init__(self, action_space, n_ues, buffer_max_size):
        super().__init__(action_space, n_ues, buffer_max_size)

    def _calculate_priorities(self, cqi, o, b, buffer_size_per_ue):
        se = np.zeros(shape=(self.K,))
        for i in range(16):
            se[cqi == i] = self.CQI2SE[i]
        priorities = (1 + o) / b * buffer_size_per_ue * se
        return priorities


class Knapsackagent(ProportionalFairAgent):
    def __init__(self, action_space, n_ues, buffer_max_size, nprb):
        super().__init__(action_space, n_ues, buffer_max_size)
        self.r = None
        self.Nf = nprb
        self.window = self.Nf * 15

    def _calculate_priorities(self, cqi, o, b, buffer_size_per_ue):
        # Normalized values
        k_cqi = (cqi / 15)
        k_buffer = (buffer_size_per_ue / (self.r + 1))
        k_age = (o / b)
        k_fairness = (1 / (1 + self.n))
        # tanh as ranking function for values
        priorities = 1 * np.tanh(k_cqi) + 1 * np.tanh(k_buffer) + 1 * np.tanh(k_age) + 1 * np.tanh(k_fairness)
        return priorities

    def act(self, state, reward, done):
        # reset the self.r
        if self.t % self.window == 0:
            self.r = np.zeros(shape=(self.K,), dtype=np.float32)

        o, cqi, b, buffer_size_per_ue = self.parse_state(state, self.K, self.L)

        priorities = self._calculate_priorities(cqi, o, b, buffer_size_per_ue)

        self.buffer_size_moving_average(state)

        action = np.argmax(priorities)
        self.n[action] += 1

        self.t += 1
        return action

    def buffer_size_moving_average(self, state):
        s = np.reshape(state[self.K:self.K * (1 + self.L)], (self.K, self.L))  # Size in bits of packets in UEs' buffers
        buffer_size_per_ue = np.sum(s, axis=1)
        # Moving Average of buffer sizes
        if self.t % self.Nf == 0 and self.t != 0:
            self.r = (1 - self.Nf / self.window) * self.r + buffer_size_per_ue * self.Nf / self.window
