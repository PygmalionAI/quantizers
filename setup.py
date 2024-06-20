from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess
import os
import multiprocessing

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        num_jobs = multiprocessing.cpu_count()
        subprocess.check_call(f"cd third_party/llama.cpp && make -j{num_jobs}", shell=True)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        if not os.path.exists('third_party/llama.cpp'):
            print("Error: llama.cpp submodule not found. Please run 'git submodule update --init --recursive'")
            return
        install.run(self)
        num_jobs = multiprocessing.cpu_count()
        subprocess.check_call(f"cd third_party/llama.cpp && make{num_jobs}", shell=True)

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