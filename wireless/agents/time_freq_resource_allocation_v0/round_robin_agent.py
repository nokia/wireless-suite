"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
from wireless.agents.random_agent import RandomAgent
import numpy as np


class RoundRobinAgent(RandomAgent):
    def __init__(self, action_space, n_ues, buffer_max_size):
        RandomAgent.__init__(self, action_space)
        self.t = 0      # Current time step

        self.K = n_ues              # Number of UEs
        self.L = buffer_max_size    # Maximum number of packets per UE buffer

    def act(self, state, reward, done):
        action = self.t % self.K
        self.t += 1
        return action


class RoundRobinIfTrafficAgent(RoundRobinAgent):
    def __init__(self, action_space, n_ues, buffer_max_size):
        RoundRobinAgent.__init__(self, action_space, n_ues, buffer_max_size)

    def act(self, state, reward, done):
        action0 = self.t % self.K

        s = np.reshape(state[self.K:self.K*(1 + self.L)], (self.K, self.L))
        buffer_size_per_ue = np.sum(s, axis=1)

        action = action0
        while buffer_size_per_ue[action] == 0:
            action = (action + 1) % self.K
            if action == action0:
                break

        self.t += 1
        return action
