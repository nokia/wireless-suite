from gym.envs.registration import register

register(
    id='TimeFreqResourceAllocation-v0',
    entry_point='wireless.envs.time_freq_resource_allocation_v0:TimeFreqResourceAllocationV0',
)

register(
    id='NomaULTimeFreqResourceAllocation-v0',
    entry_point='wireless.envs.noma_ul_time_freq_resource_allocation_v0:NomaULTimeFreqResourceAllocationV0',
)

register(
    id='UlOpenLoopPowerControl-v0',
    entry_point='wireless.envs.umts_olpc:UlOpenLoopPowerControl',
)
