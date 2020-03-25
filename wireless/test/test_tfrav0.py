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
    env = gym.make('TimeFreqResourceAllocation-v0')  # Init environment
    yield env


@pytest.fixture
def env64():
    env = gym.make('TimeFreqResourceAllocation-v0', n_ues=64)  # Init environment
    yield env


class TestTfraV0:
    def test_reproducibility(self, env64):
        env64.seed(seed=1234)
        states = []
        rewards = []
        dones = []
        for action in list(range(64)):
            state, reward, done, _ = env64.step(action)
            states.append(state)
            rewards.append(reward)
            dones.append(done)

        env = gym.make('TimeFreqResourceAllocation-v0', n_ues=64)  # Init environment
        env.seed(seed=1234)
        for action in list(range(64)):
            state, reward, done, _ = env.step(action)
            np.testing.assert_array_equal(state, states[action])
            assert reward == rewards[action]
            assert done == dones[action]

    def test_variability(self, env64):
        env64.seed(seed=1234)
        states = []
        rewards = []
        dones = []
        for action in list(range(64)):
            state, reward, done, _ = env64.step(action)
            states.append(state)
            rewards.append(reward)
            dones.append(done)

        env = gym.make('TimeFreqResourceAllocation-v0', n_ues=64)  # Init environment
        env.seed(seed=12345)
        for action in list(range(64)):
            state, reward, done, _ = env.step(action)
            if not np.array_equal(state, states[action]):
                return
            if reward != rewards[action]:
                return
            if done != dones[action]:
                return

        pytest.fail("Different seeds produced the same results.")

    def test_state_features(self):
        n_ues = 64
        n_steps = 512
        env = gym.make('TimeFreqResourceAllocation-v0', n_ues=n_ues, eirp_dbm=7)  # Low power to have some CQI=0
        env.seed(seed=1234)

        state, _, _, _ = env.step(0)  # Get state to measure its length
        states = np.zeros((n_steps, len(state)), dtype=np.uint32)  # Memory pre-allocation
        for t in range(n_steps):
            action = random.randint(0, n_ues-1)
            state, _, _, _ = env.step(action)
            states[t, :] = state

        # Check CQI range
        assert states[:, :n_ues].min() == 0
        assert states[:, :n_ues].max() == 15
        assert 0 < states[:, :n_ues].mean() < 15
        assert states[:, :n_ues].std() > 1

        # Check size (in bits) of packets in UEs' buffers
        assert states[:, n_ues:n_ues + n_ues * env.L].min() == 0
        assert states[:, n_ues:n_ues + n_ues * env.L].max() >= 41250
        assert states[:, n_ues:n_ues + n_ues * env.L].mean() > 100

        # Check age (in ms) of packets in UEs' buffers
        assert states[:, n_ues + n_ues * env.L:n_ues + 2*n_ues * env.L].min() == 0
        assert states[:, n_ues + n_ues * env.L:n_ues + 2 * n_ues * env.L].max() > 10  # Less n_prbs  yield higher ages

        # TODO: Maybe also check QI

        # Check PRB counter
        assert states[:, -1].min() == 0
        assert states[:, -1].max() == env.Nf-1
