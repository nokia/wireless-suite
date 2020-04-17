"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
from setuptools import setup

setup(name='wireless-suite',
      version='1.1',
      packages=['wireless', 'wireless', 'wireless.agents', 'wireless.envs', 'wireless.utils'],
      license='„2020 Nokia. Licensed under the BSD 3 Clause license. SPDX-License-Identifier: BSD-3-Clause',
      description='Modules for executing wireless communication problems as OpenAI Gym environments.',
      install_requires=['gym', 'numpy', 'scipy', 'sacred', 'pytest']
      )
