#! /usr/bin/env python

import os
import fnmatch

#############################################################################
########## NOTEBOOK TEST CONFIGURATION ######################################
#############################################################################

current_path = os.path.dirname(os.path.realpath(__file__))
ipynb_path = current_path + '/../doc/source/examples'
ipynbs = []
for root, dirnames, filenames in os.walk(ipynb_path):
    for filename in fnmatch.filter(filenames, '*.ipynb'):
        ipynb = os.path.join(root, filename)
        if not 'ipynb_checkpoints' in ipynb:  # exclude automatic backups
            ipynbs.append(ipynb)

# remove heavy consuming notebook
ipynbs.sort()
ipynbs.pop(0) ## adaptive signal pod
ipynbs.pop(-1) ## quantile regression pod

def pytest_addoption(parser):
    parser.addoption("--notebook", action="store_true",
        help="run all combinations")

def pytest_generate_tests(metafunc):
    if 'notebook' in metafunc.fixturenames:
        if metafunc.config.option.notebook:
            metafunc.parametrize("notebook", ipynbs)
        else:
            metafunc.parametrize("notebook", [])
