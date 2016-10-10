# -*- coding: utf-8 -*-
# -*- Python -*-

"""
    This package enables to build Probability of Detection (POD) curves using
    several models : univariate linear regression, quantile regression, kriging
    and polynomial chaos. Sensitivity analysis can be also be performed.
"""

def _initializing():
    # check version of required modules
    from importlib import import_module
    def check_version(module, version, equal=False):
        moduleImport = import_module(module)
        if equal:
            if moduleImport.__version__.split('.')[:2] != version.split('.')[:2]:
                raise ImportError(module + ' version must be ' + version)
        else:
            if moduleImport.__version__.split('.')[:2] < version.split('.')[:2]:
                raise ImportError(module + ' version must be at least ' + version)

    check_version('openturns', '1.6', True)
    check_version('statsmodels', '0.6.1')
    check_version('numpy', '1.10.4')
    check_version('sklearn', '0.17')
    check_version('matplotlib', '1.5.1')
    check_version('scipy', '0.17.0')
    check_version('logging', '0.5.1.2')
    check_version('decorator', '4.0.9')

    # initialize the logger to display informations and warnings
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

_initializing()

from ._univariate_linear_model_analysis import *
from ._univariate_linear_model_pod import *
from ._quantile_regression_pod import *
from ._polynomial_chaos_pod import *
from ._kriging_pod import *
from ._adaptive_signal_pod import *
from ._adaptive_hitmiss_pod import *
from ._pod_summary import *
from ._math_tools import *

__version__ = "0.1"

__all__ = (_univariate_linear_model_analysis.__all__ +
           _univariate_linear_model_pod.__all__ + 
           _quantile_regression_pod.__all__ +
           _polynomial_chaos_pod.__all__ +
           _kriging_pod.__all__ +
           _adaptive_signal_pod.__all__ +
           _adaptive_hitmiss_pod.__all__ +
           _pod_summary.__all__ +
           _math_tools.__all__)

