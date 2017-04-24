otpod module
============

otpod is a module for `OpenTURNS <http://www.openturns.org>`_.

Requirements
============

This module is developped in python using several external modules :

- openturns >= 1.6 (=1.8 to use the SobolIndices class)
- statsmodels >= 0.6
- numpy >= 1.10
- sklearn >= 0.17
- matplotlib >= 1.5
- scipy >= 0.17
- logging >= 0.5
- decorator >= 4.0


Installation
============

In a terminal, type in :

.. code-block:: shell

    $ python setup.py install
    $ python setup.py install --user

Add the user option to install without administrator rights.

Documentation is available `here <http://adumasphi.github.io/otpod/>`_.

Test are available in the 'test' directory. They can be launched with pytest and
the following command in a terminal in the otpod directory:

.. code-block:: shell
    
    $ py.test