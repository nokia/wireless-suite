"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
from collections import defaultdict
import numpy as np


class QLearningAgent:
    def __init__(self, seed=1,
                 learning_rate=1,
                 discount_factor=0.995,
                 exploration_rate=1.0,
                 exploration_decay_rate=0.9999,
                 num_actions=4):
        # Episode 458 is the first episode for epsilon min
        self.learning_rate = learning_rate  # alpha
        self.discount_factor = discount_factor  # gamma
        self.exploration_rate = exploration_rate  # epsilon
        self.exploration_rate_min = 0.010
        self.exploration_decay_rate = exploration_decay_rate  # d
        self.seed = seed
        self.num_actions = num_actions
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))

    def _policy(self, state):
        """
        Returns the probabilities for each action.
        """
        action_probs = np.ones(self.num_actions, dtype=float) * self.exploration_rate / self.num_actions
        best_action = np.argmax(self.q_table[state])
        action_probs[best_action] += (1.0 - self.exploration_rate)
        return action_probs

    def td_update(self, state, action, next_state, reward):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_delta

    def exploration_rate_update(self):
        self.exploration_rate *= self.exploration_decay_rate
        self.exploration_rate = max(self.exploration_rate, self.exploration_rate_min)

    def act(self, state, *_):
        action_probs = self._policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action
