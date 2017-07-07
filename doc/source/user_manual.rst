========================
Documentation of the API
========================

This is the user manual for the Python bindings to the otpod library.

.. currentmodule:: otpod

Data analysis
==============

.. autosummary::
    :toctree: _generated/
    :template: class.rst_t
    :nosignatures:

    UnivariateLinearModelAnalysis

POD computation methods
=======================

.. autosummary::
    :toctree: _generated/
    :template: class.rst_t
    :nosignatures:

    UnivariateLinearModelPOD
    QuantileRegressionPOD
    PolynomialChaosPOD
    KrigingPOD

Adaptive algorithms
===================

.. autosummary::
    :toctree: _generated/
    :template: class.rst_t
    :nosignatures:

    AdaptiveSignalPOD
    AdaptiveHitMissPOD

Sensitivity analysis
====================

.. autosummary::
    :toctree: _generated/
    :template: class.rst_t
    :nosignatures:

    SobolIndices
    PLI
    PLIMean
    PLIVariance

Tools
=====

.. autosummary:: 
    :toctree: _generated/
    :template: class.rst_t
    :nosignatures:

    PODSummary
    DataHandling