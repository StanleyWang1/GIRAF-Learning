#!/usr/bin/env python3

from setuptools import setup

from catkin_pkg.python_setup import generate_distutils_setup


setup_args = generate_distutils_setup(
    packages=["control", "giraf_optitrak_teleop"],
    package_dir={"": "."},
)

setup(**setup_args)
