# -*- coding: utf-8 -*-
# -*- Python -*-

"""
    This package enables to build Probability of Detection (POD) curves using
    several models : univariate linear regression, quantile regression, kriging
    and polynomial chaos. Sensitivity analysis can be also be performed.
"""

from ._univariate_linear_model_analysis import *
from ._univariate_linear_model_pod import *
from ._quantile_regression_pod import *
from ._polynomial_chaos_pod import *
from ._math_tools import *

__version__ = "0.0"

__all__ = (_univariate_linear_model_analysis.__all__ +
           _univariate_linear_model_pod.__all__ + 
           _quantile_regression_pod.__all__ +
           _polynomial_chaos_pod.__all__ +
           _math_tools.__all__)


# check version of required modules
from importlib import import_module
def check_version(module, version, equal=False):
    moduleImport = import_module(module)
    if equal:
        if moduleImport.__version__ != version:
            raise ImportError(module + ' version must be ' + version)
    else:
        if moduleImport.__version__ < version:
            raise ImportError(module + ' version must be at least ' + version)


check_version('openturns', '1.6', True)
check_version('statsmodels', '0.6.1')
check_version('numpy', '1.10.4')
check_version('matplotlib', '1.5.1')
check_version('scipy', '0.17.0')
check_version('logging', '0.5.1.2')
check_version('decorator', '4.0.9')


# initialize the logger to display informations and warnings
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)