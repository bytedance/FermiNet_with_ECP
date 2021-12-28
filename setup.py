# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# modified from FermiNet:https://github.com/deepmind/ferminet

"""Setup for pip package."""

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'absl-py',
    'attrs',
    'chex',
    'jax==0.2.12',
    'jaxlib==0.1.65+cuda102',
    'kfac_ferminet_alpha @ git+https://github.com/deepmind/deepmind_research#egg=kfac_ferminet_alpha&subdirectory=kfac_ferminet_alpha',  # pylint: disable=line-too-long
    'ml-collections',
    'optax',
    'numpy',
    'pandas',
    'pyscf',
    'pyblock',
    'scipy',
    'tables',
    'ferminet @ git+https://github.com/deepmind/ferminet.git@jax'
]


setup(
    name='ferminet_ecp',
    version='0.1',
    description='A libariry combining ferminet with effective core potential.',
    author='ByteDance',
    author_email='lixiang.62770689@bytedance.com',
    # Contained modules and scripts.
    scripts=['bin/ferminet_ecp'],
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require={'testing': ['flake8', 'pylint', 'pytest', 'pytype']},
    platforms=['any'],
    license='Apache 2.0',
)
