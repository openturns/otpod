# -*- coding: utf-8 -*-
# -*- Python -*-

__all__ = ['UnivariateLinearModelAnalysis']

import openturns as ot
from ._math_tools import computeBoxCox, computeZeroMeanTest, computeBreuschPaganTest, computeHarrisonMcCabeTest, computeDurbinWatsonTest
from statsmodels.regression.linear_model import OLS
import numpy as np

class UnivariateLinearModelAnalysis():

    """
    Linear regression analysis with residuals hypothesis tests.
    
    **Available constructors**

    UnivariateLinearModelAnalysis(*inputSample, outputSample*)

    UnivariateLinearModelAnalysis(*inputSample, outputSample, noiseThres,
    saturationThres, resDistFact, boxCox*)

    Parameters
    ----------
    inputSample : 2-d sequence of float
        Vector of the defect sizes, of dimension 1.
    outputSample : 2-d sequence of float
        Vector of the signals, of dimension 1.
    noiseThres : float
        Value for low censored data. Default is None.
    saturationThres : float
        Value for high censored data. Default is None
    resDistFact : :py:class:`openturns.DistributionFactory`
        Distribution hypothesis followed by the residuals. Default is 
        :py:class:`openturns.NormalFactory`.
    boxCox : bool or float
        Enable or not the Box Cox transformation. If boxCox is a float, the Box
        Cox transformation is enabled with the given value. Default is False.
    """

    def __init__(self, inputSample, outputSample, noiseThres=None,
                 saturationThres=None, resDistFact=ot.NormalFactory(),
                 boxCox=False):

        self._inputSample = ot.NumericalSample(inputSample)
        self._outputSample = ot.NumericalSample(outputSample)
        self._noiseThres = noiseThres
        self._saturationThres = saturationThres
        self._resDistFact = resDistFact

        # if Box Cox is a float the transformation is enabled with the given value
        if type(boxCox) is float:
            self._lambdaBoxCox = boxCox
            self._boxCox = True
        else:
            self._lambdaBoxCox = None
            self._boxCox = boxCox

        self._size = inputSample.getSize()
        self._dim = inputSample.getDimension()

        # Assertions on parameters
        assert (self._size >=3), "Not enough observations."
        assert (self._size == outputSample.getSize()), \
                "InputSample and outputSample must have the same size."
        assert (self._dim == 1), "InputSample must be of dimension 1."
        assert (outputSample.getDimension() == 1), "OutputSample must be of dimension 1."


    def run(self):
        """
        Run the analysis :

        - Compute the Box Cox parameter if boxCox is True,
        - Compute the transformed signals if boxCox is enabled,
        - Build the univariate linear regression model on the data,
        - Compute the residuals,
        - Run all hypothesis tests.

        """
        
        ###################### Box Cox transform ###############################
        # Compute Box Cox if enabled
        if self._boxCox:
            if self._lambdaBoxCox is None:
                # optimization required
                self._lambdaBoxCox, self._signals = computeBoxCox(self._inputSample,
                                                            self._outputSample)
            else:
                # no optimization for lambdaBoxCox
                self._lambdaBoxCox, self._signals = computeBoxCox(self._inputSample,
                                                            self._outputSample,
                                                            lambdaBoxCox=self._lambdaBoxCox)
        else:
            self._signals = self._outputSample

        ######################### Linear Regression model ######################
        # Create the X matrix : [1, inputSample]
        X = ot.NumericalSample(self._size, [1, 0])
        X[:, 1] = self._inputSample
        algoLinear = OLS(np.array(self._signals), np.array(X)).fit()
        self._intercept = algoLinear.params[0]
        self._slope = algoLinear.params[1]

        # create the linear model as NumericalMathFunction
        def model(x):
            return [self._intercept + self._slope * x[0]]
        LinearRegModel = ot.PythonFunction(1, 1, model)

        
        ############################ Residuals #################################
        # get Residuals
        self._residuals = ot.NumericalSample(np.vstack(algoLinear.resid))
        self._residuals.setDescription(['Residuals'])

        # compute residual distribution
        self._resDist = self._resDistFact.build(self._residuals)

        # get confidence interval at level 95%
        # square array
        self._confInt = algoLinear.conf_int(0.05)

        ########################## Compute test ################################
        # compute R2
        val = ot.MetaModelValidation(self._inputSample, self._signals, LinearRegModel)
        self._R2 = val.computePredictivityFactor()

        # compute Anderson Darling test (normality test)
        testAnderDar = ot.NormalityTest.AndersonDarlingNormal(self._residuals)
        self._pValAnderDar = testAnderDar.getPValue()

        # compute Cramer Von Mises test (normality test)
        testCramVM = ot.NormalityTest.CramerVonMisesNormal(self._residuals)
        self._pValCramVM = testCramVM.getPValue()

        # compute zero residual mean test
        self._pValZeroMean = computeZeroMeanTest(self._residuals)

        # compute Kolomogorov test (fitting test)
        testKol = ot.FittingTest.Kolmogorov(self._residuals, self._resDist, 0.95,
                                            self._resDist.getParametersNumber())
        self._pValKol = testKol.getPValue()

        # compute Breusch Pagan test (homoskedasticity : constant variance)
        self._pValBreuPag = computeBreuschPaganTest(self._inputSample,
                                                    self._residuals)

        # compute Breusch Pagan test (homoskedasticity : constant variance)
        self._pValHarMcCabe = computeHarrisonMcCabeTest(self._residuals)

        # compute Durbin Watson test (autocorrelation == 0)
        self._pValDurWat = computeDurbinWatsonTest(self._inputSample,
                                                   self._residuals)

