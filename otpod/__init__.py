# -*- coding: utf-8 -*-
# -*- Python -*-

"""
    This package enables to build Probability of Detection (POD) curves using
    several models : univariate linear regression, quantile regression, kriging
    and polynomial chaos. Sensitivity analysis can be also be performed.
"""

def _initializing():
    # check openturns version
    from distutils.version import LooseVersion
    import openturns as ot
    if LooseVersion(ot.__version__) < "1.12":
        raise ImportError(ot.__name__ + ' version must be at least 1.12.')

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
from ._sobol_indices import *
from ._pli_pod import *
from ._pli import *
from ._math_tools import *

__version__ = "0.5"

__all__ = (_univariate_linear_model_analysis.__all__ +
           _univariate_linear_model_pod.__all__ + 
           _quantile_regression_pod.__all__ +
           _polynomial_chaos_pod.__all__ +
           _kriging_pod.__all__ +
           _adaptive_signal_pod.__all__ +
           _adaptive_hitmiss_pod.__all__ +
           _pod_summary.__all__ +
           _sobol_indices.__all__ +
           _math_tools.__all__)

