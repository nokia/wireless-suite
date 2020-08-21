"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
from setuptools import setup

setup(name='wireless-suite',
      version='1.1',
      packages=['wireless', 'wireless.agents', 'wireless.agents.time_freq_resource_allocation_v0', 'wireless.agents.noma_ul_time_freq_resource_allocation_v0', 'wireless.envs', 'wireless.utils'],
      license='„2020 Nokia. Licensed under the BSD 3 Clause license. SPDX-License-Identifier: BSD-3-Clause',
      description='Modules for executing wireless communication problems as OpenAI Gym environments.',
      install_requires=['gym', 'matplotlib', 'numpy', 'scipy', 'sacred', 'pytest']
      )
