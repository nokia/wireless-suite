import gym
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sacred import Experiment

from wireless.agents.q_learning import QLearningAgent


num_episodes = 512
max_steps_per_episode = 512
snr_tgt_db = 4

# Memory allocation
episode_rewards = np.zeros(num_episodes)
epsilon = np.zeros(num_episodes)          # To store exploration level
snr_in_some_episodes = defaultdict(lambda: np.zeros(max_steps_per_episode))  # To store Power Control dynamics
episodes_to_save = np.linspace(0, num_episodes, num=5, dtype=int)


def run_episode(e, env, agent, save_snr=False):
    state = env.reset()

    s = 0  # Step count
    while True:
        # Take a step
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        # Collect progress
        if save_snr:
            snr_in_some_episodes[e][s] = state
        episode_rewards[e] += reward
        agent.td_update(state, action, next_state, reward)
        agent.exploration_rate_update()
        s += 1
        if done:
            break
        state = next_state


def run_n_episodes(num_episodes, env, agent, seed=0, log_progress=True):
    log_period = round(num_episodes / 10)

    for e in range(num_episodes):
        if log_progress and e % log_period == 0:
            print(f"\rEpisode {e}/{num_episodes}.")

        env.seed(seed=seed + e)
        epsilon[e] = agent.exploration_rate
        run_episode(e, env, agent, save_snr=e in episodes_to_save)


# Load agent parameters
with open('../../config/config_agent.json') as f:
    ac = json.load(f)

# Configure experiment
with open('../../config/config_sacred.json') as f:
    sc = json.load(f)  # Sacred Configuration
    ns = sc["sacred"]["n_metrics_points"]  # Number of points per episode to log in Sacred
    ex = Experiment(ac["agent"]["agent_type"], save_git_info=False)
    ex.add_config(sc)
    ex.add_config(ac)
mongo_db_url = f'mongodb://{sc["sacred"]["sacred_user"]}:{sc["sacred"]["sacred_pwd"]}@' + \
               f'{sc["sacred"]["sacred_host"]}:{sc["sacred"]["sacred_port"]}/{sc["sacred"]["sacred_db"]}'
# ex.observers.append(MongoObserver(url=mongo_db_url, db_name=sc["sacred"]["sacred_db"]))  # Uncomment to save to DB

# Load environment parameters
with open('../../config/config_environment.json') as f:
    ec = json.load(f)
    ex.add_config(ec)


@ex.automain
def main(_run):
    env = gym.make('UlOpenLoopPowerControl-v0', f_carrier_mhz=_run.config['env']['f_carrier_mhz'],
                   t_max=max_steps_per_episode)  # Init environment

    agent = QLearningAgent(seed=_run.config['seed'], num_actions=env.action_space.n)

    run_n_episodes(num_episodes, env, agent, _run.config['seed'])

    # Plot results
    plt.figure()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(num_episodes), episode_rewards, 'g-')
    ax2.plot(range(num_episodes), epsilon, 'b-')

    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Episode reward', color='g')
    ax2.set_ylabel('Exploration rate', color='b')
    plt.grid(True)

    plt.figure()
    for e, snr in snr_in_some_episodes.items():
        plt.plot(snr, label=f'Episode {e}')
    plt.grid(True)
    plt.xlabel('Steps')
    plt.ylabel('SNR')
    plt.legend(loc='upper right')

    plt.show()
