"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
import pytest
import gym
import numpy as np


@pytest.fixture(params=[2, 3])
def env(request):
    env = gym.make('NomaULTimeFreqResourceAllocation-v0', n_ues_per_prb=request.param)  # Init environment
    yield env


@pytest.fixture(params=[2, 3])
def env64(request):
    env = gym.make('NomaULTimeFreqResourceAllocation-v0', n_ues=64, n_ues_per_prb=request.param)  # Init environment
    yield env


class TestNomaULTfraV0:
    def test_reproducibility(self, env64):
        M = env64.M
        np.random.seed(1234)
        actions = np.random.randint(0,64,size=(100,M))
        env64.seed(seed=1234)
        states = []
        rewards = []
        dones = []
        for action in actions:
            state, reward, done, _ = env64.step(action)
            states.append(state)
            rewards.append(reward)
            dones.append(done)

        env = gym.make('NomaULTimeFreqResourceAllocation-v0', n_ues=64, n_ues_per_prb=M)  # Init environment
        env.seed(seed=1234)
        pt = 0
        for action in actions:
            state, reward, done, _ = env.step(action)
            np.testing.assert_array_equal(state, states[pt])
            assert reward == rewards[pt]
            assert done == dones[pt]
            pt += 1

    def test_variability(self, env64):
        M = env64.M
        np.random.seed(1234)
        actions = np.random.randint(0,64,size=(100,M))
        env64.seed(seed=1234)
        states = []
        rewards = []
        dones = []
        for action in actions:
            state, reward, done, _ = env64.step(action)
            states.append(state)
            rewards.append(reward)
            dones.append(done)

        env = gym.make('NomaULTimeFreqResourceAllocation-v0', n_ues=64, n_ues_per_prb=M)  # Init environment
        env.seed(seed=12345)
        pt = 0
        for action in actions:
            state, reward, done, _ = env.step(action)
            if not np.array_equal(state, states[pt]):
                return
            if reward != rewards[pt]:
                return
            if done != dones[pt]:
                return
            pt += 1

        pytest.fail("Different seeds produced the same results.")

    def test_empty_action(self, env, env64):
        env.step([32]*env.M)
        env64.step([64]*env64.M)

    def test_state_features(self):
        n_ues = 64
        n_steps = 512
        M = 2
        env = gym.make('NomaULTimeFreqResourceAllocation-v0', n_ues=n_ues, eirp_dbm=7)  # Low power to have some CQI=0
        env.seed(seed=1234)

        state, _, _, _ = env.step([0]*M)  # Get state to measure its length
        states = np.zeros((n_steps, len(state)), dtype=np.uint32)  # Memory pre-allocation
        for t in range(n_steps):
            action = np.random.randint(0,n_ues,size=M)
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

    def test_consistency_superclass(self, env64):
        M = env64.M
        np.random.seed(1234)
        ofdm_actions = np.random.randint(0,64,size=100)
        env64.seed(seed=1234)
        states = []
        rewards = []
        dones = []
        for action in ofdm_actions:
            noma_action = [action]+[64]*(M-1)
            state, reward, done, _ = env64.step(noma_action)
            states.append(state)
            rewards.append(reward)
            dones.append(done)

        env = gym.make('TimeFreqResourceAllocation-v0', n_ues=64)  # Init TimeFreqResourceAllocation-V0 environment
        env.seed(seed=1234)
        pt = 0
        for action in ofdm_actions:
            state, reward, done, _ = env.step(action)
            np.testing.assert_array_equal(state, states[pt])
            assert reward == rewards[pt]
            assert done == dones[pt]
            pt += 1