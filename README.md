# Wireless Suite

## Overview
Wireless Suite is a collection of problems in wireless telecommunications.

Comparing research results in telecoms remains a challenge due to the lack of standard problem implementations against
which to benchmark.
To solve this, Wireless Suite implements some well-known problems, built as Open-AI Gym compatible classes.
These are intended to establish performance benchmarks, stimulate reproducible research and foster quantitative
comparison of algorithms for telecommunication problems.

## Getting started
The code has been tested to work on Python 3.7 under Windows 10.

1. Get the code:
    ```
    git clone https://github.com/nokia/wireless-suite.git
    ```

2. Use `pip3` to install the package:
   ```
   cd wireless-suite
   pip3 install .
   ```

3. **OPTIONAL**: Modify the script *scripts/launch_agent.py* to execute a problem of your choosing.

4. **OPTIONAL**: Modify the configuration of your problem at *config/config_environment.json*.

5. Simulate an agent-environment interaction:
    ```
   cd wireless/scripts
   python launch_agent.py
   ```

## Provided problems 

### TimeFreqResourceAllocation-v0
This environment simulates a OFDM resource allocation task, where a limited number of frequency resources are to be
allocated to a large number of User Equipments (UEs) over time.
An agent interacting with this environment plays the role of the MAC scheduler. On each time step, the agent must
allocate one frequency resource to one of a large number of UEs. The agent gets rewarded for these resource allocation
decisions. The reward increases with the number of UEs, whose traffic requirements are satisfied.
The traffic requirements for each UE are expressed in terms of their Guaranteed Bit Rate (if any) and their Packet
Delay Budget (PDP).

You are invited to develop a new agent that interacts with this environment and takes effective resource allocation
decisions.
Five sample agents are provided for reference in the *wireless/agents* folder.
The performance obtained by the default agents on the default environment configuration is:
* Random                          -69590
* Round Robin                     -69638
* Round Robin IfTraffic           -3284
* Proportional Fair               -9595
* Proportional Fair Channel Aware -1729

Note that the above average rewards are negative values. The best performing agent is thus the Proportional Fair Channel Aware.

Additional details about this problem are provided in document *wireless/doc/TimeFreqResourceAllocation-v0.pdf*

### NomaULTimeFreqResourceAllocation-v0
This environment is an extension of the above TimeFreqResourceAllocation-v0 environment, with the difference that it
allows multiple UEs to be allocated on a time-frequency resource. It consists on an uplink power-domain NOMA system,
wherein the base station receives superimposed signals from the multiplexed UEs and performs successive interference
cancellation (SIC) to decode them. 

The default environment can be obtained by setting `"env": "NomaULTimeFreqResourceAllocation-v0"` and
`"n_ues_per_prb": 2` in *config/config_environment.json*. 
Two sample agents are provided for reference in the *wireless/agents* folder. 
The performance obtained on the default environment configuration is:
* Random                          -33499
* NOMA UL Proportional Fair Channel Aware -1431

### Evaluation
The simulated environment can be chosen by setting `"env": "TimeFreqResourceAllocation-v0"` or `"env": "NomaULTimeFreqResourceAllocation-v0"` in *config/config_environment.json*. The script *wireless/scripts/launch_agent.py* runs 16 episodes with a maximum of 65536 time steps each, and collects the reward
obtained by the agent on each time step. The result is calculated as the average reward obtained in all time steps on all episodes.

## How to contribute
There are two main ways of contributing to Wireless Suite:

1. **Implementing new problems**: This version of Wireless Suite contains two problems implementation. New
problems can be easily added as simple variations of the existing ones (e.g. by changing their parameters), or by introducing
fully new problem implementations (e.g. Adaptive Modulation and Coding, Open Loop Power Control, Handover optimization,
etc).

2. **Implementing new agents**: Ideally, new agent contributions shall perform better than the default ones.

## References
1. [Open AI Gym Documentation](http://gym.openai.com/docs/)
2. [How to create new environments for Gym](https://github.com/openai/gym/blob/master/docs/creating-environments.md)
3. [Sacred Documentation](https://sacred.readthedocs.io/en/stable/index.html)


## License

This project is licensed under the BSD-3-Clause license - see the [LICENSE](https://github.com/nokia/wireless-suite/blob/master/LICENSE).