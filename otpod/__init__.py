# -*- coding: utf-8 -*-
# -*- Python -*-

"""
    This package enables to build Probability of Detection (POD) curves using
    several models : univariate linear regression, quantile regression, kriging
    and polynomial chaos. Sensitivity analysis can be also be performed.
"""

from ._univariate_linear_model_analysis import *
from ._univariate_linear_model_pod import *

__version__ = "0.0"

__all__ = (_univariate_linear_model_analysis.__all__ +
           _univariate_linear_model_pod.__all__)
