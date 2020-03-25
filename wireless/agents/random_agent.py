"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""


class RandomAgent:
    """
    The world's simplest agent!

    See: https://github.com/openai/gym/blob/master/examples/agents/random_agent.py
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state, reward, done):
        return self.action_space.sample()

    def seed(self, seed=0):
        self.action_space.seed(seed)
