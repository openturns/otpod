# -*- coding: utf-8 -*-
# -*- Python -*-

__all__ = ['AdaptiveSignalPOD']

import openturns as ot
import numpy as np
from ._pod import POD
from scipy.interpolate import interp1d
from _decorator import DocInherit, keepingArgs
from _progress_bar import updateProgress
from _kriging_tools import estimKrigingTheta, computeLOO, computeQ2, computePODSamplePerDefect
import logging

class AdaptiveSignalPOD(POD):
    """
    Adaptive algorithm for signal data type.

    **Available constructor:**

    AdaptiveSignalPOD(*initialDOE, physicalModel, detection, noiseThres,
    saturationThres, boxCox*)

    Parameters
    ----------
    inputSample : 2-d sequence of float
        Vector of the input values. The first column must correspond with the
        defect sizes.
    outputSample : 2-d sequence of float
        Vector of the signals, of dimension 1.
    detection : float
        Detection value of the signal.
    noiseThres : float
        Value for low censored data. Default is None.
    saturationThres : float
        Value for high censored data. Default is None
    boxCox : bool or float
        Enable or not the Box Cox transformation. If boxCox is a float, the Box
        Cox transformation is enabled with the given value. Default is False.

    Warnings
    --------
    The first column of the input sample must corresponds with the defects sample.

    Notes
    -----
    This class aims at building the POD based on a kriging model. No assumptions
    are required for the residuals with this method. The POD are computed by
    simulating conditional prediction. For each, a Monte Carlo simulation is
    performed. The accuracy of the Monte Carlo simulation is taken into account
    using the TCL.
    
    The return POD model corresponds with an interpolate function built
    with the POD values computed for the given defect sizes. The default values
    are 20 defect sizes between the minimum and maximum value of the defect sample.
    The defect sizes can be changed using the method *setDefectSizes*.

    The default kriging model is built with a linear basis only for the defect
    size and constant otherwise. The covariance model is an anisotropic squared
    exponential model. Parameters are estimated using the TNC algorithm, the
    initial starting point of the TNC is found thanks to a quasi random search 
    of the best loglikelihood value among 1000 computations.

    For advanced use, all parameters can be defined thanks to dedicated set 
    methods. Moreover, if the user has already built a kriging result, 
    it can be given as parameter using the method *setKrigingResult*,
    then the POD is computed based on this kriging result.

    A progress bar is shown if the verbosity is enabled. It can be disabled using
    the method *setVerbose*.
    """

    def __init__(self, inputDOE, outputDOE, physicalModel, detection, noiseThres=None,
                 saturationThres=None, boxCox=False):

        # initialize the POD class
        super(AdaptiveSignalPOD, self).__init__(inputDOE, outputDOE,
                                 detection, noiseThres, saturationThres, boxCox)
        # inherited attributes
        # self._simulationSize
        # self._detection
        # self._inputSample
        # self._outputSample
        # self._noiseThres
        # self._saturationThres        
        # self._lambdaBoxCox
        # self._boxCox
        # self._size
        # self._dim
        # self._censored

        assert (self._dim > 1), "Dimension of inputSample must be greater than 1."

        self._normalDist = ot.Normal()
        self._samplingSize = 1000 # Number of MC simulations to compute POD
        self._simulationSize = 100
        self._enrichDOESize = 100
        self._nMaxIteration = 20
        self._verbose = True

        if self._censored:
            logging.info('Censored data are not taken into account : the ' + \
                         'kriging model is only built on filtered data.')

        # Run the preliminary run of the POD class
        result = self._run(self._inputSample, self._outputSample, self._detection,
                           self._noiseThres, self._saturationThres, self._boxCox,
                           self._censored)

        # get some results
        self._input = result['inputSample']
        self._signals = result['signals']
        self._detectionBoxCox = result['detectionBoxCox']
        self._boxCoxTransform = result['boxCoxTransform']

        # define the defect sizes for the interpolation function if not defined
        if self._defectSizes is None:
            self._defectNumber = 10
            self._defectSizes = np.linspace(self._input[:,0].getMin()[0], 
                                      self._input[:,0].getMax()[0],
                                      self._defectNumber)

    def run(self):
        """
        Build the POD models.

        Notes
        -----
        This method build the kriging model. First the censored data
        are filtered if needed. The Box Cox transformation is performed if it is
        enabled. Then it builds the POD models : conditional samples are 
        simulated for each defect size, then the distributions of the probability
        estimator (for MC simulation) are built. Eventually, a sample of this
        distribution is used to compute the mean POD and the POD at the confidence
        level.
        """


        inputMin = self._input.getMin()
        inputMax = self._input.getMax()
        marginals = [ot.Uniform(inputMin[i], inputMax[i]) for i in range(self._dim)]
        distribution = ot.ComposedDistribution(marginals)

        samplingSize = 1000 # Number of MC simulations to compute POD
        simulationSize = 100
        enrichDOESize = 100
        doeCandidate = distribution.getSample(enrichDOESize)

        iteration = 0
        while iteration < self._nMaxIteration:
            iteration += 1
            if self._verbose:
                print 'Iteration : {}/{}'.format(iteration, self._nMaxIteration)
            # First build of the kriging
            algoKriging = self._buildKrigingAlgo(self._input, self._signals)
            algoKriging.run()

            Q2 = computeQ2(self._input, self._signals, algoKriging.getResult())
            print Q2

            # for the first iteration the optimization is always performed
            if Q2 < 0.95 or iteration == 1:
                if self._verbose:
                    'Optimization of the covariance model parameters...'
                covDim = algoKriging.getResult().getCovarianceModel().getScale().getDimension()
                lowerBound = [0.001] * covDim
                upperBound = [50] * covDim               
                algoKriging = estimKrigingTheta(algoKriging,
                                                      lowerBound, upperBound,
                                                      self._initialStartSize)
                algoKriging.run()

            krigingResult = algoKriging.getResult()
            self._covarianceModel = krigingResult.getCovarianceModel()
            self._basis = krigingResult.getBasisCollection()
            metamodel = krigingResult.getMetaModel()

            Q2 = computeQ2(self._input, self._signals, algoKriging.getResult())
            if self._verbose:
                print 'Q2 : {0.4f}'.format(Q2)

            # compute POD (ptrue = pn-1) for bias reducing in the criterion
            # Monte Carlo for all defect sizes in a vectorized way
            samplePred = distribution.getSample(samplingSize)[:,1:]
            fullSamplePred = ot.NumericalSample(samplingSize * self._defectNumber, self._dim)
            for i, defect in enumerate(self._defectSizes):
                fullSamplePred[samplingSize*i:samplingSize*(i+1), :] = \
                                        self._mergeDefectInX(defect, samplePred)
            predictionSample = metamodel(fullSamplePred)
            predictionSample = np.reshape(predictionSample, (samplingSize,
                                                    self._defectNumber), 'F')
            currentPOD = np.mean(predictionSample > self._detectionBoxCox, axis=0)

            # compute criterion
            # Compute criterion for all candidate in the candidate doe
            criterion = []
            for icand, candidate in enumerate(doeCandidate):

                # add the current candidate to the kriging doe
                inputAugmented = self._input[:]
                inputAugmented.add(candidate)
                signalsAugmented = self._signals[:]
                # predict the signal value of the candidate using the current
                # kriging model
                signalsAugmented.add(metamodel(candidate))
                # create a temporary kriging model with the new doe and without
                # updating the covariance model parameters
                algoKrigingTemp = ot.KrigingAlgorithm(inputAugmented, signalsAugmented,
                                                      self._basis,
                                                      self._covarianceModel,
                                                      True)
                algoKrigingTemp.run()
                krigingResultTemp = algoKrigingTemp.getResult()

                # compute the criterion for all defect size
                crit = []
                for idef, defect in enumerate(self._defectSizes):
                    podSample = computePODSamplePerDefect(defect, self._detectionBoxCox,
                        krigingResultTemp,distribution, simulationSize, samplingSize)

                    meanPOD = np.mean(podSample)
                    varPOD = np.var(podSample, ddof=1)
                    crit.append(varPOD + (meanPOD - currentPOD[idef])**2)
                # compute the criterion
                criterion.append(np.sqrt(np.mean(crit)))
                
                if self._verbose:
                    updateProgress(icand, int(doeCandidate.getSize()), 'Computing criterion')

            # look for the best candidate
            indexOpt = np.argmax(criterion)
            candidateOpt = doeCandidate[indexOpt]
            # add new point to DOE
            self._input.add(candidateOpt)
            if self._boxCox:
                self._signals.add(self._boxCoxTransform(self._physicalModel(candidateOpt)))
            else:
                self._signals.add(self._physicalModel(candidateOpt))
            # remove added candidate
            doeCandidate.erase(indexOpt)
            if self._verbose:
                    print 'Criterion value : {:0.4f}'.format(criterion[indexOpt])
                    print 'Added point : {}'.format(candidateOpt)
                    print ''


    def _buildKrigingAlgo(self, inputSample, outputSample):
        """
        Build the functional chaos algorithm without running it.
        """
        if self._basis is None:
            # create linear basis only for the defect parameter (1st parameter),
            # constant otherwise
            input = ['x'+str(i) for i in range(self._dim)]
            functions = []
            # constant
            functions.append(ot.NumericalMathFunction(input, ['y'], ['1']))
            # linear for the first parameter only
            functions.append(ot.NumericalMathFunction(input, ['y'], [input[0]]))
            self._basis = ot.Basis(functions)

        if self._covarianceModel is None:
            # anisotropic squared exponential covariance model
            covColl = ot.CovarianceModelCollection(self._dim)
            for i in xrange(self._dim):
                covColl[i]  = ot.SquaredExponential(1, 1.)
            self._covarianceModel = ot.ProductCovarianceModel(covColl)

        algoKriging = ot.KrigingAlgorithm(inputSample, outputSample, self._basis,
                                                     self._covarianceModel, True)
        algoKriging.run()
        return algoKriging

    def _mergeDefectInX(self, defect, X):
        """
        defect : scalar of the defect value
        X : sample without the defect column
        """
        size = X.getSize()
        dim = X.getDimension() + 1
        samplePred = ot.NumericalSample(size, dim)
        samplePred[:, 0] = ot.NumericalSample(size, [defect])
        samplePred[:, 1:] = X
        return samplePred