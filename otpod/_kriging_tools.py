# -*- coding: utf-8 -*-
# -*- Python -*-

import openturns as ot
import numpy as np


__all__ = []


def estimKrigingTheta(algoKriging, lowerBound, upperBound, size):
    """
    Estimate the kriging theta values with an initial random search using
    a Sobol sequence of size samples.
    """
    # get input parameters of the kriging algorithm
    X = algoKriging.getInputSample()
    Y = algoKriging.getOutputSample()
    
    algoKriging.run()
    resultKriging = algoKriging.getResult()
    covarianceModel = resultKriging.getCovarianceModel()
    basis = resultKriging.getBasisCollection()
    llf = algoKriging.getLogLikelihoodFunction()

    # create uniform distribution of the parameters bounds
    dim = len(lowerBound)
    distBoundCol = []
    for i in range(dim):
        distBoundCol += [ot.Uniform(lowerBound[i], upperBound[i])]
    distBound = ot.ComposedDistribution(distBoundCol)    

    # Generate starting points with a low discrepancy sequence
    thetaStart = ot.LowDiscrepancyExperiment(ot.SobolSequence(), distBound,
                                                            size).generate()
    # Get the best theta from the maximum llf value
    llfValue = llf(thetaStart)
    indexMax = np.argmax(llfValue)
    bestTheta = thetaStart[indexMax]

    # update theta after random search
    covarianceModel.setScale(bestTheta)

    # set TNC optim
    optimizer = ot.TNC()
    searchInterval = ot.Interval(lowerBound, upperBound)
    optimizer.setBoundConstraints(searchInterval)
    # Now the KrigingAlgorithm is used to optimize the likelihood using a
    # good starting point
    algoKriging = ot.KrigingAlgorithm(X, Y, basis, covarianceModel, True)
    algoKriging.setOptimizer(optimizer)
    algoKriging.run() 
    return algoKriging


def computeLOO(inputSample, outputSample, krigingResult):
    """
    Compute the Leave One out prediction analytically.
    """
    inputSample = np.array(inputSample)
    outputSample = np.array(outputSample)

    # get covariance model
    cov = krigingResult.getCovarianceModel()
    # get input transformation
    t = krigingResult.getTransformation()
    # check if the transformation was enabled or not and if so transform
    # the input sample
    if t.getInputDimension() == inputSample.shape[1]:
        normalized_inputSample = np.array(t(inputSample))
    else:
        normalized_inputSample = inputSample

    # correlation matrix and Cholesky decomposition
    Rtrianglow = np.array(cov.discretize(normalized_inputSample))
    R = Rtrianglow + Rtrianglow.T - np.eye(Rtrianglow.shape[0])
    # get sigma2 (covariance model scale parameters)
    sigma2 = krigingResult.getSigma2()
    # get coefficient and compute trend
    basis = krigingResult.getBasisCollection()[0]
    F1 = krigingResult.getTrendCoefficients()[0]
    size = inputSample.shape[0]
    p = F1.getDimension()
    F = np.ones((size, p))
    for i in range(p):
        F[:, i] = np.hstack(basis.build(i)(normalized_inputSample))
    # Calcul de y_loo
    K = sigma2 * R
    Z = np.zeros((p, p))
    S = np.vstack([np.hstack([K, F]), np.hstack([F.T, Z])])
    S_inv = np.linalg.inv(S)
    B = S_inv[:size:, :size:]
    B_but_its_diag = B * (np.ones(B.shape) - np.eye(size))
    B_diag = np.atleast_2d(np.diag(B)).T
    y_loo = (- np.dot(B_but_its_diag / B_diag, outputSample)).ravel()
    return y_loo

def computeQ2(inputSample, outputSample, krigingResult):
    """
    Compute the Q2 using the analytical loo prediction.
    """
    y_loo = computeLOO(inputSample, outputSample, krigingResult)
    # Calcul du Q2
    delta = (np.hstack(outputSample) - y_loo)
    return 1 - np.mean(delta**2)/np.var(outputSample)

def computePODSamplePerDefect(defect, detection, krigingResult, distribution,
                              simulationSize, samplingSize):
    """
    Compute the POD sample for a defect size.
    """

    dim = distribution.getDimension()
    # create a distibution with a dirac distribution for the defect size
    diracDist = [ot.Dirac(defect)]
    diracDist += [distribution.getMarginal(i+1) for i in xrange(dim-1)]
    distribution = ot.ComposedDistribution(diracDist)

    # create a sample for the Monte Carlo simulation and confidence interval
    MC_sample = distribution.getSample(samplingSize)
    # Kriging_RV = ot.KrigingRandomVector(krigingResult, MC_sample)
    # Y_sample = Kriging_RV.getSample(simulationSize)
    Y_sample = randomVectorSampling(krigingResult, MC_sample, simulationSize,
                                    samplingSize)

    # compute the POD for all simulation size
    POD_MCPG_a = [float(np.where(Y_sample[i] >  \
                  detection)[0].shape[0])/Y_sample.shape[1] for i \
                  in xrange(simulationSize)]
    # compute the variance of the MC simulation using TCL
    VAR_TCL = np.array(POD_MCPG_a)*(1-np.array(POD_MCPG_a)) / Y_sample.shape[1]
    # Create distribution of the POD estimator for all simulation 
    POD_PG_dist = []
    for i in xrange(simulationSize):
        if VAR_TCL[i] > 0:
            POD_PG_dist += [ot.Normal(POD_MCPG_a[i],np.sqrt(VAR_TCL[i]))]
        else:
            if POD_MCPG_a[i] < 1:
                POD_PG_dist += [ot.Dirac([0.])]
            else:
                POD_PG_dist += [ot.Dirac([1.])]
    POD_PG_alea = ot.ComposedDistribution(POD_PG_dist)
    # get a sample of these distributions
    POD_PG_sample = POD_PG_alea.getSample(samplingSize)
    POD_PG_sample_array = np.array(POD_PG_sample)
    POD_PG_sample_array.resize((simulationSize * samplingSize,1))
    POD_PG_sample = ot.NumericalSample(POD_PG_sample_array)

    return POD_PG_sample

def randomVectorSampling(resultKriging, sample, simulationSize, samplingSize):
    """
    Kriging Random vector perso
    """
    
    # only compute the variance
    variance = np.hstack([resultKriging.getConditionalCovariance(
                        sample[i])[0,0] for i in xrange(samplingSize)])
    pred = resultKriging.getConditionalMean(sample)

    normalSample = ot.Normal().getSample(simulationSize)
    # with numpy broadcasting
    randomVector = np.array(normalSample)* np.sqrt(variance) + np.array(pred)
    return randomVector
