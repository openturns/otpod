# -*- Python -*-

"""
    This package enables to build Probability of Detection (POD) curves using
    several models : univariate linear regression, quantile regression, kriging
    and polynomial chaos. Sensitivity analysis can be also be performed.
"""


def _initializing():
    # check openturns version
    import openturns as ot

    try:
        from pkg_resources import parse_version
    except ImportError:
        from packaging.version import Version as parse_version
    if parse_version(ot.__version__) < parse_version("1.18"):
        raise ImportError("openturns version must be >=1.18.")

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

__version__ = "0.6.11"

__all__ = (
    _univariate_linear_model_analysis.__all__
    + _univariate_linear_model_pod.__all__
    + _quantile_regression_pod.__all__
    + _polynomial_chaos_pod.__all__
    + _kriging_pod.__all__
    + _adaptive_signal_pod.__all__
    + _adaptive_hitmiss_pod.__all__
    + _pod_summary.__all__
    + _sobol_indices.__all__
    + _math_tools.__all__
)
