#! /usr/bin/env python

import os
import fnmatch

#############################################################################
########## NOTEBOOK TEST CONFIGURATION ######################################
#############################################################################

current_path = os.path.dirname(os.path.realpath(__file__))
ipynb_path = current_path + "/../doc/source/examples"
ipynbs = []
for root, dirnames, filenames in os.walk(ipynb_path):
    for filename in fnmatch.filter(filenames, "*.ipynb"):
        ipynb = os.path.join(root, filename)
        if not "ipynb_checkpoints" in ipynb:  # exclude automatic backups
            ipynbs.append(ipynb)

print(ipynbs)
print(root)
# remove heavy consuming notebook
ipynbs.sort()
ipynbs.remove(os.path.join(root, "AdaptiveSignalPOD.ipynb"))  ## adaptive signal pod
ipynbs.remove(os.path.join(root, "adaptiveHitMissPOD.ipynb"))  ## adaptive hit miss pod
ipynbs.remove(
    os.path.join(root, "quantileRegressionPOD.ipynb")
)  ## quantile regression pod


def pytest_addoption(parser):
    parser.addoption("--notebook", action="store_true", help="run all combinations")


def pytest_generate_tests(metafunc):
    if "notebook" in metafunc.fixturenames:
        if metafunc.config.option.notebook:
            metafunc.parametrize("notebook", ipynbs)
        else:
            metafunc.parametrize("notebook", [])
