# -*- coding: utf-8 -*-
# -*- Python -*-

__all__ = ['DataHandling']

import openturns as ot
import math as m
import numpy as np
from scipy.optimize import fmin


######### ReducedLogLikelihood #########
# This class defines the log likelihood of the residuals versus lambda when
# sigma and beta have been replaced by their expression in function of lambda
# and the data.
# LL(Y_i, a_i; \lambda, \sigma, \beta_0, \beta_1) =
# -N/2\log(2\pi)-N\log(\sigma)-1/(2\sigma^2)\sum_{i=1}^N(T_{\lambda}(Y_i)-(\beta_0+\beta_1 a_i))^2
#
# Parameters
# ----------
# sigma2 : float
#     unbiased estimate of the residual variance.
# beta : sequence of float of dim = 2
#     Least squares estimate of the linear regression model.
class ReducedLogLikelihood(ot.OpenTURNSPythonFunction):

    def __init__(self, a_i, Y_i):
        super(ReducedLogLikelihood, self).__init__(1, 1)
        self.setInputDescription(['lambda'])
        self.setOutputDescription(['LogLikelihood'])
        self.a_i_ = a_i
        self.Y_i_ = Y_i
        self.N_ = a_i.getSize()
        self.sumLogY_i = ot.SymbolicFunction(["y"], ["log(y)"])(Y_i).computeMean()[0] * self.N_
        
    def _exec(self, Lambda):
        Y_lambda = ot.BoxCoxTransform(Lambda)(self.Y_i_)
        algo = ot.LinearLeastSquares(self.a_i_, Y_lambda)
        algo.run()
        beta0 = algo.getConstant()[0]
        beta1 = algo.getLinear()[0, 0]
        sigma2 = (self.N_ - 1.0) / (self.N_ - 2.0) * (Y_lambda - (self.a_i_ * [beta1] + [beta0])).computeVariance()[0]
        return [-0.5 * self.N_ * m.log(sigma2) + (Lambda[0] - 1) * self.sumLogY_i]

######### LinearBoxCoxFactory #########
# This class build the Box Cox optimal transformation corresponding with
# Gaussian residuals once removes the linear trend.
# Given data (a_i, Y_i), find the quadruplet (\lambda, \sigma, \beta_0, \beta_1
# that maximize the log-likelihood function defined by ReducedLogLikelihood.
class LinearBoxCoxFactory:

    def __init__(self, lambdaMin = -3, lambdaMax = 3):
        self.lambdaMin_ = lambdaMin
        self.lambdaMax_ = lambdaMax
        ot.Log.Show(ot.Log.NONE)
        
    def build(self, dataX, dataY):
        logLikelihood = ot.Function(ReducedLogLikelihood(dataX, dataY))
        xlb = np.linspace(self.lambdaMin_,self.lambdaMax_,num=500)
        lambdax = [logLikelihood([x])[0] for x in xlb]
        algo = ot.TNC(logLikelihood)
        algo.setStartingPoint([xlb[np.array(lambdax).argmax()]])
        algo.setBoundConstraints(ot.Interval(self.lambdaMin_, self.lambdaMax_))
        algo.setOptimizationProblem(ot.BoundConstrainedAlgorithmImplementationResult.MAXIMIZATION)
        algo.run()
        optimalLambda = algo.getResult().getOptimizer()[0]

        # graph
        optimalLogLikelihood = algo.getResult().getOptimalValue()
        graph = logLikelihood.draw(0.01 * optimalLambda, 10.0 * optimalLambda)
        c = ot.Cloud([[optimalLambda, optimalLogLikelihood]])
        c.setColor("red")
        c.setPointStyle("circle")
        graph.add(c)
        return ot.BoxCoxTransform([optimalLambda]), graph


######### computeBoxCox #########
# This function applies the Box Cox transformation on the data.
def computeBoxCox(factors, valuesInit, shift):
    # if no affine trend is considered
    graph = ot.Graph()
    myBoxCoxFactory = ot.BoxCoxFactory()
    myModelTransform = myBoxCoxFactory.build(valuesInit, [shift], graph)
    lambdaBoxCox = myModelTransform.getLambda()[0]

    # if an affine trend is considered (more computing time required)
    # works only in 1D
    # myBoxCoxFactory = LinearBoxCoxFactory()
    # myModelTransform, graph = myBoxCoxFactory.build(factors, valuesInit)
    # lambdaBoxCox = myModelTransform.getLambda()[0]
    return lambdaBoxCox, graph



######### computeR2 #########
# This function the R2 of the linear regression model
def computeR2(signals, residuals):
    R2 = 1 - residuals.computeVariance()[0] / \
         (signals - signals.computeMean()).computeVariance()[0]
    return R2


######### computeZeroMeanTest #########
# This function tests if the residuals have a zero mean
def computeZeroMeanTest(residuals):
    # Student test with hypothesis zero mean
    mRes = residuals.computeMean()[0]
    varRes = residuals.computeVariance()[0]
    stderr = np.sqrt(varRes / residuals.getSize())
    statistic = mRes / stderr
    # two sided test
    return 2 * ot.DistFunc.pStudent(residuals.getSize()-1, -np.abs(statistic))

######### computeBreuschPaganTest #########
# This function tests if the residuals are homoskedastics
def computeBreuschPaganTest(x, residuals):
    nx = x.getSize()
    df = 1 # linear regression with 2 parameters -> degree of freedom = 2 - 1
    residuals = np.array(residuals)
    sigma2 = np.sum(residuals**2) / nx
    # Studentized Breusch Pagan
    w = residuals**2 - sigma2
    linResidual = ot.LinearLeastSquares(x, w)
    linResidual.run()
    linModel = linResidual.getMetaModel()
    wpred = np.array(linModel(x))
    # statistic Breusch Pagan
    bp = nx * np.sum(wpred**2) / np.sum(w**2)
    # return complementary cdf of central ChiSquare
    return 1 - ot.DistFunc.pNonCentralChiSquare(df, 0, bp)

######### computeHarrisonMcCabeTest #########
# This function tests if the residuals are homoskedastics
def computeHarrisonMcCabeTest(residuals, breakRatio=0.5, simulationSize=1000):
    # Parameters:
    # breakPoint : ratio of point to define the break, must be 0 < breakRatio< 1
    nx = residuals.getSize()
    residuals = np.array(residuals)
    breakpoint = int(np.floor(breakRatio * nx))
    # statistic Harrison McCabe
    hmc = np.sum(residuals[:breakpoint]**2) /np.sum(residuals**2)

    # pvalue computed by simulation
    stat = np.zeros(simulationSize)
    normalDist = ot.Normal()
    for i in range(simulationSize):
        xSampleNor = normalDist.getSample(nx)
        try:
            stddev = xSampleNor.computeStandardDeviation()[0]
        except:
            stddev = xSampleNor.computeStandardDeviation()[0, 0] # ot <1.17
        xstand = np.array((xSampleNor - xSampleNor.computeMean()[0]) / stddev)
        stat[i] = np.sum(xstand[:breakpoint]**2) / np.sum(xstand**2)

    return np.mean(stat <= hmc)

######### computeDurbinWatsonTest #########
# This function tests if the residuals have non autocorrelation
def computeDurbinWatsonTest(x, residuals, hypothesis="Equal"):
    # Parameters:
    # hypothesis : string
    #    "Equal" : hypothesis is autocorrelation is 0
    #    "Less" : hypothesis is autocorrelation is less than 0
    #    "Greater" : hypothesis is autocorrelation is greater than 0
    nx = x.getSize()
    dim = x.getDimension()
    residuals = np.array(residuals)
    # statistic Durbin Watson
    dw = np.sum(np.diff(np.hstack(residuals))**2)/np.sum(residuals**2)

    # Normal approxiimation of DW to compute the pvalue
    X = ot.Matrix(nx, dim+1)
    X[:, 0] = np.ones((nx, 1))
    X[:, 1] = x
    B = ot.Matrix(nx, dim+1)
    B[0, 1] = x[0][0] - x[1][0]
    B[nx-1, 1] = x[nx-1][0] - x[nx-2][0]
    for i in range(nx - 2):
        B[i+1, 1] = -x[i][0] + 2 * x[i+1][0] - x[i+2][0]

    XtX = X.computeGram()
    XBQt = ot.SquareMatrix(XtX.solveLinearSystem(B.transpose() * X))
    P = 2 * (nx - 1) - XBQt.computeTrace()
    XBTrace = ot.SquareMatrix(XtX.solveLinearSystem(B.computeGram(), False)).computeTrace()
    Q = 2 * (3 * nx - 4) - 2 * XBTrace + ot.SquareMatrix(XBQt * XBQt).computeTrace()
    dmean = P / (nx - (dim + 1))
    dvar = 2.0 / ((nx - (dim + 1)) * (nx - (dim + 1) + 2)) * (Q - P * dmean)

    # compute the pvalue with respect to hypothesis
    # Default pvalue is for hypothesis == "Equal"
    # complementary CDF of standard normal distribution
    pValue = 2 * ot.DistFunc.pNormal(np.abs(dw - dmean) / np.sqrt(dvar), True)
    if hypothesis == "Less":
        pValue = 1 - pValue / 2
    elif hypothesis == "Greater":
        pValue = pValue / 2

    return pValue

######### filterCensoredData #########
# This function filters the input sample and signals in case where low and/or high
# threshold are given.
class DataHandling(object):
    """
    Static methods for data handling.
    """
    @staticmethod
    def filterCensoredData(inputSample, signals, noiseThres, saturationThres):
        """
        Sort inputSample and signals with respect to the censore thresholds.

        Parameters
        ----------
        inputSample : 2-d sequence of float
            Vector of the input sample.
        signals : 2-d sequence of float
            Vector of the signals, of dimension 1.
        noiseThres : float
            Value for low censored data. Default is None.
        saturationThres : float
            Value for high censored data. Default is None

        Returns
        -------
        inputSampleUnc : 2-d sequence of float
            Vector of the input sample in the uncensored area.
        inputSampleNoise : 2-d sequence of float
            Vector of the input sample in the noisy area.
        inputSampleSat : 2-d sequence of float
            Vector of the input sample in the saturation area.
        signalsUnc : 2-d sequence of float
            Vector of the signals in the uncensored area.

        Notes
        -----
        The data are sorted in three different vectors whether they belong to
        the noisy area, the uncensored area or the saturation area.
        """
        # check if one sided censoring
        if noiseThres is None:
            noiseThres = -ot.sys.float_info.max
        if saturationThres is None:
            saturationThres = ot.sys.float_info.max

        # transform in numpy.array
        inputSample = np.array(inputSample)
        signals = np.array(signals)
        # inputSample in the uncensored area
        inputSampleUnc = inputSample[np.hstack(np.logical_and(signals > noiseThres, 
                                            signals < saturationThres))]
        # inputSample in the noisy area
        inputSampleNoise = inputSample[np.hstack(signals <= noiseThres)]
        # inputSample in the saturation area
        inputSampleSat = inputSample[np.hstack(signals >= saturationThres)]
        # signals in the uncensored area
        signalsUnc = signals[np.hstack(np.logical_and(signals > noiseThres,
                                            signals < saturationThres))]

        # transform in Sample
        inputSampleUnc = ot.Sample(inputSampleUnc)
        inputSampleNoise = ot.Sample(inputSampleNoise)
        inputSampleSat = ot.Sample(inputSampleSat)
        signalsUnc = ot.Sample(signalsUnc)

        return inputSampleUnc, inputSampleNoise, inputSampleSat, signalsUnc



######### computeLinearParametersCensored #########
# This function compute the linear regression parameters with censored data
# using the MLE function (from Berens 1988 article)
def MLE(X, defects, defectsNoise, defectsSat, signals, noiseThres,
        saturationThres):
    '''
    Compute - log likelihood on censored data.
    Parameters:
    -----------
    X : Contains [intercept, slope, sigma_residuals]
    defects : vector of the defects in the uncensored area
    defectsNoise : vector of the defects in the noisy area (low censored data)
    defectsSat : vector of the defects in the saturation area
    signals : vector of the signals in the uncensored area
    noiseThres : noise threshold
    saturationThres : saturation threshold
    '''
    b0 = X[0]
    b1 = X[1]
    s = X[2]

    # uncensored area
    MLE = len(defects) * np.log(s * np.sqrt(2. * np.pi))
    MLE += 1.0 / (2 * s**2) * np.sum((signals - (b0 + b1 * defects))**2)

    # noisy area
    Znoise = (noiseThres - (b0 + b1 * defectsNoise)) / s
    cdfZnoise = np.array([ot.DistFunc.pNormal(Znoise[i][0]) for i in range(len(Znoise))])
    MLE += - np.sum(np.log(cdfZnoise))

    # saturation area
    Zsat = (saturationThres - (b0 + b1 * defectsSat)) / s
    cdfZsat = np.array([ot.DistFunc.pNormal(Zsat[i][0]) for i in range(len(Zsat))])
    MLE += - np.sum(np.log(1. - cdfZsat))

    if np.isnan(MLE):
        MLE = np.inf
    return MLE

def computeLinearParametersCensored(initialStartMLE, defects, defectsNoise,
                            defectsSat, signals, noiseThres, saturationThres):
    """
    Compute the linear regression parameters using the MLE function taking
    into account the censored data.
    """

    defects = np.array(defects)
    defectsNoise = np.array(defectsNoise)
    defectsSat = np.array(defectsSat)
    signals = np.array(signals)

    func = lambda x: MLE(x, defects, defectsNoise, defectsSat,
                              signals, noiseThres, saturationThres)

    testMLE = func(initialStartMLE)
    if testMLE == np.inf:
        iteration = 0
        while testMLE == np.inf and iteration < 100:
            iteration += 1
            initialStartMLE = np.random.randn(3)
            testMLE = func(initialStartMLE)
        if iteration == 100:
            raise Exception('Maximum Likelihood optimization for censored '+\
                            'data : cannot find initial starting point.')
    res = fmin(func, initialStartMLE, disp=0, full_output=1, maxiter=500)
    if res[4] == 2:
        ot.Log.Show(ot.Log.WARN)
        ot.Log.Warn('Maximum Likelihood optimization for censored data : '+\
                    'maximum number of iterations reached.')
    return res[0]
