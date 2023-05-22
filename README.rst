.. image:: https://github.com/openturns/otpod/actions/workflows/build.yml/badge.svg?branch=master
    :target: https://github.com/openturns/otpod/actions/workflows/build.yml

otpod module
============

otpod is a module for `OpenTURNS <http://www.openturns.org>`_.

This package enables to build Probability of Detection (POD) curves from Non
Destructive Test. The curves are built using parametric models : univariate linear
regression, quantile regression, kriging and polynomial chaos. Analysis can be
run in order to test the linear regression hypothesis.

PoD can be built from a set of data or directly from a given physical model that
simulate the Non Destructive Test. In this case, the design of experiments is
defined iteratively.

Sensitivity analysis can be also be performed. The aggregated Sobol indices are
available as well as the perturbation law indices.

Requirements
============

This module is developped in python using several external modules:

- openturns >= 1.18
- statsmodels >= 0.6
- numpy >= 1.10
- sklearn >= 0.17
- matplotlib >= 1.5
- scipy >= 0.17
- logging >= 0.5
- decorator >= 4.0


Installation
============

In a terminal, type in:

.. code-block:: shell

    $ python setup.py install
    $ python setup.py install --user

Add the user option to install without administrator rights.

Or you can install the module using Anaconda repository.

.. code-block:: shell

    $ conda config --add channels conda-forge
    $ conda install otpod

The documentation with examples is available `here <http://openturns.github.io/otpod/master>`_.

Test are available in the 'test' directory. They can be launched with pytest and
the following command in a terminal in the otpod directory:

.. code-block:: shell
    
    $ pytest test/
