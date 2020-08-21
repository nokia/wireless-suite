"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
import pytest
import random
import gym
import numpy as np


@pytest.fixture
def env():
    env = gym.make('UlOpenLoopPowerControl-v0')  # Init environment
    yield env


class TestTfraV0:
    def test_reproducibility(self, env):
        env.seed(seed=1234)
        random.seed(1234)
        states = []
        rewards = []
        dones = []
        for t in range(64):
            action = random.randint(0, 3)
            state, reward, done, _ = env.step(action)
            states.append(state)
            rewards.append(reward)
            dones.append(done)

        env2 = gym.make('UlOpenLoopPowerControl-v0')  # Init environment
        env2.seed(seed=1234)
        random.seed(1234)
        for t in range(64):
            action = random.randint(0, 3)
            state, reward, done, _ = env2.step(action)
            np.testing.assert_array_equal(state, states[t])
            assert reward == rewards[action]
            assert done == dones[action]

    def test_variability(self, env):
        env.seed(seed=1234)
        random.seed(1234)
        states = []
        rewards = []
        dones = []
        for t in range(64):
            action = random.randint(0, 3)
            state, reward, done, _ = env.step(action)
            states.append(state)
            rewards.append(reward)
            dones.append(done)

        env2 = gym.make('UlOpenLoopPowerControl-v0')  # Init environment
        env2.seed(seed=12345)
        random.seed(12345)
        for t in range(64):
            action = random.randint(0, 3)
            state, reward, done, _ = env.step(action)
            if not np.array_equal(state, states[t]):
                return
            if reward != rewards[t]:
                return
            if done != dones[t]:
                return

        pytest.fail("Different seeds produced the same results.")
