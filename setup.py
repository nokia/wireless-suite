"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
from setuptools import setup

setup(name='wireless_suite',
      version='1.0',
      packages=['wireless_suite'],
      license='„2020 Nokia. Licensed under the BSD 3 Clause license. SPDX-License-Identifier: BSD-3-Clause',
      description='Modules for executing wireless communication problems as OpenAI Gym environments.',
      install_requires=['gym', 'numpy', 'scipy', 'sacred', 'pytest']
      )
