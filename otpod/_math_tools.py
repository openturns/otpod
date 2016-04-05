# -*- coding: utf-8 -*-
# -*- Python -*-

__all__ = []

import openturns as ot
import math as m
from numpy import array, linspace, where



#
# LL(Y_i, a_i; \lambda, \sigma, \beta_0, \beta_1) =
# -N/2\log(2\pi)-N\log(\sigma)-1/(2\sigma^2)\sum_{i=1}^N(T_{\lambda}(Y_i)-(\beta_0+\beta_1 a_i))^2

class ReducedLogLikelihood(ot.OpenTURNSPythonFunction):

    """
    ReducedLogLikelihood
    This class defines the log likelihood of the residuals versus lambda when
    sigma and beta have been replaced by their expression in function of lambda
    and the data.

    Parameters:
    -----------
    sigma2 : float
        unbiased estimate of the residual variance.
    beta : sequence of float of dim = 2
        Least squares estimate of the linear regression model.
"""

    def __init__(self, a_i, Y_i):
        super(ReducedLogLikelihood, self).__init__(1, 1)
        self.setInputDescription(['lambda'])
        self.setOutputDescription(['LogLikelihood'])
        self.a_i_ = a_i
        self.Y_i_ = Y_i
        self.N_ = a_i.getSize()
        self.sumLogY_i = ot.NumericalMathFunction("y", "log(y)")(Y_i).computeMean()[0] * self.N_
        
    def _exec(self, Lambda):
        Y_lambda = ot.BoxCoxTransform(Lambda)(self.Y_i_)
        algo = ot.LinearLeastSquares(self.a_i_, Y_lambda)
        algo.run()
        beta0 = algo.getConstant()[0]
        beta1 = algo.getLinear()[0, 0]
        sigma2 = (self.N_ - 1.0) / (self.N_ - 2.0) * (Y_lambda - (self.a_i_ * [beta1] + [beta0])).computeVariance()[0]
        return [-0.5 * self.N_ * m.log(sigma2) + (Lambda[0] - 1) * self.sumLogY_i]

######### LinearBoxCoxFactory #########
# Cette classe construit la transformation de Box-Cox optimale correspondant
# a des residus Gaussiens une fois supprimee une tendance affine.
# Given data (a_i, Y_i), find the quadruplet (\lambda, \sigma, \beta_0, \beta_1
# that maximize the log-likelihood function defined by ReducedLogLikelihood
class LinearBoxCoxFactory:

    def __init__(self, lambdaMin = -3, lambdaMax = 3):
        self.lambdaMin_ = lambdaMin
        self.lambdaMax_ = lambdaMax
        ot.Log.Show(Log.NONE)
        
    def build(self, dataX, dataY):
        logLikelihood = ot.NumericalMathFunction(ReducedLogLikelihood(dataX, dataY))
        xlb = linspace(self.lambdaMin_,self.lambdaMax_,num=500)
        lambdax = [logLikelihood([x])[0] for x in xlb]
        algo = ot.TNC(logLikelihood)
        algo.setStartingPoint([xlb[array(lambdax).argmax()]])
        algo.setBoundConstraints(ot.Interval(self.lambdaMin_, self.lambdaMax_))
        algo.setOptimizationProblem(ot.BoundConstrainedAlgorithmImplementationResult.MAXIMIZATION)
        algo.run()
        optimalLambda = algo.getResult().getOptimizer()[0]
        # Pour verifier qu'on a bien l'optimum
        # optimalLogLikelihood = algo.getResult().getOptimalValue()
        # graph = logLikelihood.draw(0.1 * optimalLambda, 10.0 * optimalLambda)
        # c = Cloud([[optimalLambda, optimalLogLikelihood]])
        # c.setColor("red")
        # c.setPointStyle("circle")
        # graph.add(c)
        return ot.BoxCoxTransform([optimalLambda])


######### computeBoxCox ######### 
# cette fonction fait la transformation de Box Cox sur le sdonnnes
# le lambda est estime par max de vraisemblance sur les donnees initiales

def computeBoxCox(myData, seuilData):
    valuesInit = myData[:,1]
    # Si on ne considere pas de tendance
    # myBoxCoxFactory = BoxCoxFactory()
    # myModelTransform = myBoxCoxFactory.build(valuesInit)
    # Si on considere une tendance affine
    factors = myData[:,0]
    myBoxCoxFactory = LinearBoxCoxFactory()
    myModelTransform = myBoxCoxFactory.build(factors, valuesInit)
    lambdaBoxCox = myModelTransform.getLambda()[0]
    seuilBC = myModelTransform([seuilData])[0]
    valuesBC = myModelTransform(valuesInit)
    valuesBC.setDescription(['BoxCox('+ valuesInit.getDescription()[0] + ')'])
    return lambdaBoxCox, valuesBC, seuilBC