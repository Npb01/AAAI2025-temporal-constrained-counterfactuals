from setuptools import setup, find_packages

import codecs
import os

from setuptools import setup, find_packages
here = os.path.abspath(os.path.dirname(__file__))

setup(
    name='declare4py',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['numpy', 'pandas', 'pm4py', 'matplotlib', 'boolean.py', 'clingo', 'ltlf2dfa'],
)
