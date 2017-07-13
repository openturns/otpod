#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from setuptools import setup, find_packages
 

# Get the version from __init__.py
with open('otpod/__init__.py') as fid:
    for line in fid:
        if line.startswith('__version__'):
            version = line.strip().split()[-1][1:-1]
            break
 
setup(
 
    # library name
    name='otpod',
 
    # code version
    version=version,
    
    # list libraries to be imported
    packages=find_packages(),
 
    author="Antoine Dumas",
 
    # Votre email, sachant qu'il sera publique visible, avec tous les risques
    # que ça implique.
    author_email="dumas@phimeca.com",
 
    # Descriptions
    description="Build Probability of Detection curve",
    long_description=open('README.rst').read(),
 
    # List of dependancies
    install_requires= ['statsmodels>=0.6.1',
                       'numpy>=1.10.4',
                       'scikit-learn>=0.17',
                       'matplotlib>=1.5.1',
                       'scipy>=0.17.0',
                       'logging',
                       'decorator>=4.0.9']
 
    # Enable to take into account MANIFEST.in
    # include_package_data=True,
)