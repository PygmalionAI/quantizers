from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess
import os

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        subprocess.check_call("cd third_party/llama.cpp && make", shell=True)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        if not os.path.exists('third_party/llama.cpp'):
            print("Error: llama.cpp submodule not found. Please run 'git submodule update --init --recursive'")
            return
        install.run(self)
        subprocess.check_call("cd third_party/llama.cpp && make", shell=True)

setup(
    name='quantizers',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'quantizers=main:main'
        ]
    },
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)