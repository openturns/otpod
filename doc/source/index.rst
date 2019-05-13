.. otpod documentation master file, created by
   sphinx-quickstart on Mon Apr  4 17:22:10 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to otpod's documentation!
=================================

otpod is a module for `OpenTURNS <http://www.openturns.org>`_.

This package enables to build Probability of Detection (POD) curves from Non Destructive Test. The curves are built using parametric models : univariate linear regression, quantile regression, kriging and polynomial chaos. Analysis can be run in order to test the linear regression hypothesis.

PoD can be built from a set of data or directly from a given physical model that simulate the Non Destructive Test. In this case, the design of experiments is defined iteratively.

Sensitivity analysis can be also be performed. The aggregated Sobol indices are available as well as the perturbation law indices.

Contents:
---------
    
.. toctree::
    :maxdepth: 3

    user_manual

.. toctree::
    :maxdepth: 2

    examples/examples.rst

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`

