#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from setuptools import setup, find_packages
 

import otpod
 

setup(
 
    # library name
    name='otpod',
 
    # code version
    version=otpod.__version__,
    
    # list libraries to be imported
    packages=find_packages(),
 
    author="Antoine Dumas",
 
    # Votre email, sachant qu'il sera publique visible, avec tous les risques
    # que ça implique.
    author_email="dumas@phimeca.com",
 
    # Descriptions
    description="Build Probability of Detection",
    long_description=open('README.rst').read(),
 
    # List of dependancies
    # install_requires= ["openturns==1.6"]
 
    # Enable to take into account MANIFEST.in
    # include_package_data=True,
)