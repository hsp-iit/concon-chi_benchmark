# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

from setuptools import setup, find_packages
from vlpers.third_party.install import install_third_party
from setuptools.command.install import install
from setuptools.command.develop import develop


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        install_third_party()
        
class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        install_third_party()

setup(
    name='vlpers', 
    version='1.0',
    packages=find_packages(),
    cmdclass={'install': PostInstallCommand, 'develop': PostDevelopCommand},
    entry_points={
        'console_scripts': [
            'vlpers_init = vlpers.third_party.install:install_third_party',
            'vlpers_remove = vlpers.third_party.install:remove_third_party',
            'download_conconchi = vlpers.datasets.download_3c:download_dataset',
        ],
    },
)

