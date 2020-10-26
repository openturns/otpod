# -*- coding: utf-8 -*-
# -*- Python -*-

__all__ = ['AdaptiveSignalPOD']

import os
import openturns as ot
import numpy as np
from ._pod import POD
from scipy.interpolate import interp1d
from ._progress_bar import updateProgress
from ._kriging_tools import KrigingBase
import logging
import matplotlib.pyplot as plt

class AdaptiveSignalPOD(POD, KrigingBase):
    """
    Adaptive algorithm for signal data type.

    **Available constructor:**

    AdaptiveSignalPOD(*inputDOE, outputDOE,  physicalModel, nMorePoints, detection, noiseThres,
    saturationThres, boxCox*)

    Parameters
    ----------
    inputDOE : 2-d sequence of float
        Vector of the input values. The first column must correspond with the
        defect sizes.
    outputDOE : 2-d sequence of float
        Vector of the signals, of dimension 1.
    physicalModel : :py:class:`openturns.Function`
        True model used to compute the real signal value to be added to the DOE.
    nMorePoints : positive int
        The number of points to add to the DOE, computed by the *physicalModel*.
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
    The first column of the input sample must corresponds with the defect sizes.

    Notes
    -----
    This class aims at building the POD based on a kriging model where the design
    of experiments is iteratively enriched. The initial design of experiments is
    given as input parameters. The enrichment criterion is based on the integrated
    mean squared of the POD. The criterion is computed on several candidate
    points and the one that minimizes the criterion is added to the current
    design of experiments. The sample of candidate points is created using 
    a low discrepancy sequence (Sobol') if the input distribution has an
    independant copula, otherwise a Monte Carlo experiment is used. This is a 
    time consuming technique because it requires to compute the mean and variance
    of the POD for all candidate points. The stopping criterion is only based 
    on the number of points that must be added to the design of experiments.

    No assumptions are required for the residuals with this method. The POD are
    computed by simulating conditional predictions. For each, a Monte Carlo
    simulation is performed. The accuracy of the Monte Carlo simulation is taken
    into account using the TCL.
    
    The return POD model corresponds with an interpolate function built
    with the POD values computed for the given defect sizes. The default values
    are 20 defect sizes between the minimum and maximum value of the defect sample.
    The defect sizes can be changed using the method *setDefectSizes*. It is
    adviced to run a preliminary POD study in order to know the interesting range
    of defect sizes. This enables reducing the computing time.

    The default kriging model is built with a linear basis only for the defect
    size and constant otherwise. The covariance model is an anisotropic squared
    exponential model. Parameters are estimated using the Mutli start TNC
    algorithm with 100 computations.

    In the algorithm, when a point is added to the design of experiments, the kriging
    model is not always optimized. The covariance model scale coefficients are
    optimized only if the Q2 value is lower than 0.95.

    For advanced use, all parameters can be defined thanks to dedicated set 
    methods.

    A progress bar is shown if the verbosity is enabled. It can be disabled using
    the method *setVerbose*.
    """

    def __init__(self, inputDOE, outputDOE, physicalModel, nMorePoints,
                 detection, noiseThres=None, saturationThres=None, boxCox=False):

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

        self._physicalModel = physicalModel
        self._basis = None
        self._covarianceModel = None
        self._distribution = None
        self._initialStartSize = 100
        self._samplingSize = 5000 # Number of MC simulations to compute POD
        self._candidateSize = 1000
        self._nIteration = nMorePoints
        self._verbose = True
        self._graph = False # flag to print or not the POD curves at each iteration
        self._probabilityLevel = None # default graph option
        self._confidenceLevel = None # default graph option
        self._graphDirectory = None # graph directory for saving
        
        self._normalDist = ot.Normal()

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
        self._shift = result['shift']

        # define the defect sizes for the interpolation function if not defined
        self._defectNumber = 10
        self._defectSizes = np.linspace(self._input[:,0].getMin()[0], 
                                        self._input[:,0].getMax()[0],
                                        self._defectNumber)

    def run(self):
        """
        Launch the algorithm and build the POD models.

        Notes
        -----
        This method launches the iterative algorithm. First the censored data
        are filtered if needed. The Box Cox transformation is performed if it is
        enabled. Then the enrichment of the design of experiments is performed.
        Once the algorithm stops, it builds the POD models : conditional samples are 
        simulated for each defect size, then the distributions of the probability
        estimator (for MC simulation) are built. Eventually, a sample of this
        distribution is used to compute the mean POD and the POD at the confidence
        level.
        """

        # Create an initial uniform distribution if not given
        if self._distribution is None:
            inputMin = self._input.getMin()
            inputMin[0] = np.min(self._defectSizes)
            inputMax = self._input.getMax()
            inputMax[0] = np.max(self._defectSizes)
            marginals = [ot.Uniform(inputMin[i], inputMax[i]) for i in range(self._dim)]
            self._distribution = ot.ComposedDistribution(marginals)

        # Create the design of experiments of the candidate points where the
        # criterion is computed
        if self._distribution.hasIndependentCopula():
            # without copula use low discrepancy experiment as first doe
            doeCandidate = ot.LowDiscrepancyExperiment(ot.SobolSequence(), 
                            self._distribution, self._candidateSize).generate()
        else:
            # else simple Monte Carlo distribution
            doeCandidate = self._distribution.getSample(self._candidateSize)

        # build initial kriging model
        # build the kriging model without optimization
        algoKriging, transformation = self._buildKrigingAlgo(self._input, self._signals)
        if self._verbose:
            print('Building the kriging model')
            print('Optimization of the covariance model parameters...')

        llDim = algoKriging.getReducedLogLikelihoodFunction().getInputDimension()
        lowerBound = [0.001] * llDim
        upperBound = [50] * llDim
        algoKriging = self._estimKrigingTheta(algoKriging,
                                              lowerBound, upperBound,
                                              self._initialStartSize)
        algoKriging.run()

        # Get kriging results
        self._krigingResult = algoKriging.getResult() 
        self._covarianceModel = self._krigingResult.getCovarianceModel()
        self._basis = self._krigingResult.getBasisCollection()
        metamodel = ot.ComposedFunction(self._krigingResult.getMetaModel(), transformation)

        self._Q2 = self._computeQ2(self._input, self._signals, self._krigingResult, transformation)
        if self._verbose:
            print('Kriging validation Q2 (>0.9): {:0.4f}\n'.format(self._Q2))

        plt.ion()
        # Start the improvment loop
        iteration = 0
        while iteration < self._nIteration:
            iteration += 1
            if self._verbose:
                print('Iteration : {}/{}'.format(iteration, self._nIteration))

            # compute POD (ptrue = pn-1) for bias reducing in the criterion
            # Monte Carlo for all defect sizes in a vectorized way.
            # get Sample for all parameters except the defect size
            samplePred = self._distribution.getSample(self._samplingSize)[:,1:]
            fullSamplePred = ot.Sample(self._samplingSize * self._defectNumber,
                                                self._dim)
            # Add the defect sizes as first value 
            for i, defect in enumerate(self._defectSizes):
                fullSamplePred[self._samplingSize*i:self._samplingSize*(i+1), :] = \
                                        self._mergeDefectInX(defect, samplePred)
            meanPredictionSample = metamodel(fullSamplePred)
            meanPredictionSample = np.reshape(meanPredictionSample, (self._samplingSize,
                                                    self._defectNumber), 'F')
            # compute the POD for all defect sizes
            currentPOD = np.mean(meanPredictionSample > self._detectionBoxCox, axis=0)

            # Compute criterion for all candidate in the candidate doe
            criterion = 1000000000
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

                # normalization
                mean = inputAugmented.computeMean()
                stddev = inputAugmented.computeStandardDeviationPerComponent()
                linear = ot.SquareMatrix(self._dim)
                for j in range(self._dim):
                    linear[j, j] = 1.0 / stddev[j] if abs(stddev[j]) > 1e-12 else 1.0
                zero = [0.0] * self._dim
                transformation = ot.LinearFunction(mean, zero, linear)

                algoKrigingTemp = ot.KrigingAlgorithm(transformation(inputAugmented), signalsAugmented,
                                                      self._covarianceModel,
                                                      self._basis)
                optimizer = algoKrigingTemp.getOptimizationAlgorithm()
                optimizer.setMaximumIterationNumber(0)
                algoKrigingTemp.setOptimizationAlgorithm(optimizer)
                algoKrigingTemp.run()
                krigingResultTemp = algoKrigingTemp.getResult()

                # compute the criterion for all defect size
                crit = []
                # save results, used to compute the PODModel et PODCLModel
                PODPerDefect = ot.Sample(self._simulationSize *
                                         self._samplingSize, self._defectNumber)
                for idef, defect in enumerate(self._defectSizes):
                    podSample = self._computePODSamplePerDefect(defect,
                        self._detectionBoxCox, krigingResultTemp, transformation,
                        self._distribution, self._simulationSize, self._samplingSize)
                    PODPerDefect[:, idef] = podSample

                    meanPOD = podSample.computeMean()[0]
                    varPOD = podSample.computeVariance()[0]
                    crit.append(varPOD + (meanPOD - currentPOD[idef])**2)
                # compute the criterion aggregated for all defect sizes
                newCriterion = np.sqrt(np.mean(crit))

                # check if the result is better or not
                if newCriterion < criterion:
                    self._PODPerDefect = PODPerDefect
                    criterion = newCriterion
                    indexOpt = icand
                
                if self._verbose:
                    updateProgress(icand, int(doeCandidate.getSize()), 'Computing criterion')

            # get the best candidate
            candidateOpt = doeCandidate[indexOpt]
            # add new point to DOE
            self._input.add(candidateOpt)
            # add the signal computed by the physical model
            if self._boxCox:
                self._signals.add(self._boxCoxTransform(self._physicalModel(candidateOpt) + [self._shift]))
            else:
                self._signals.add(self._physicalModel(candidateOpt))
            # remove added candidate from the doeCandidate
            doeCandidate.erase(indexOpt)
            if self._verbose:
                print('Criterion value : {:0.4f}'.format(criterion))
                print('Added point : {}'.format(candidateOpt))
                print('Update the kriging model')

            # update the kriging model without optimization
            algoKriging, transformation = self._buildKrigingAlgo(self._input, self._signals)
            algoKriging.setOptimizeParameters(False)
            algoKriging.run()
            self._Q2 = self._computeQ2(self._input, self._signals, algoKriging.getResult(), transformation)

            # Check the quality of the kriging model if it needs optimization
            if self._Q2 < 0.95:
                if self._verbose:
                    print('Optimization of the covariance model parameters...')

                algoKriging.setOptimizeParameters(True)
                algoKriging = self._estimKrigingTheta(algoKriging,
                                                      lowerBound, upperBound,
                                                      self._initialStartSize)
                algoKriging.run()

            # Get kriging results
            self._krigingResult = algoKriging.getResult()
            self._covarianceModel = self._krigingResult.getCovarianceModel()
            self._basis = self._krigingResult.getBasisCollection()

            self._Q2 = self._computeQ2(self._input, self._signals, self._krigingResult, transformation)
            if self._verbose:
                print('Kriging validation Q2 (>0.9): {:0.4f}'.format(self._Q2))

            if self._graph:
                # create the interpolate function of the POD model
                meanPOD = self._PODPerDefect.computeMean()
                interpModel = interp1d(self._defectSizes, np.array(meanPOD), kind='linear')
                self._PODmodel = ot.PythonFunction(1, 1, interpModel)
                # The POD at confidence level is built in getPODCLModel() directly
                fig, ax = self.drawPOD(self._probabilityLevel, self._confidenceLevel)
                plt.draw()
                plt.pause(0.001)
                plt.show()
                if self._graphDirectory is not None:
                    if not os.path.exists(self._graphDirectory):
                        os.makedirs(self._graphDirectory)
                    fig.savefig(os.path.join(self._graphDirectory, 'AdaptiveSignalPOD_')+str(iteration),
                                bbox_inches='tight', transparent=True)

        # Compute the final POD with the last updated kriging model
        if self._verbose:
                print('\nStart computing the POD with the last updated kriging model')
        # compute the sample containing the POD values for all defect 
        self._PODPerDefect = ot.Sample(self._simulationSize *
                                         self._samplingSize, self._defectNumber)
        for i, defect in enumerate(self._defectSizes):
            self._PODPerDefect[:, i] = self._computePODSamplePerDefect(defect,
                self._detectionBoxCox, self._krigingResult, transformation,
                self._distribution, self._simulationSize, self._samplingSize)
            if self._verbose:
                updateProgress(i, self._defectNumber, 'Computing POD per defect')

        # compute the mean POD 
        meanPOD = self._PODPerDefect.computeMean()
        # create the interpolate function of the POD model
        interpModel = interp1d(self._defectSizes, np.array(meanPOD), kind='linear')
        self._PODmodel = ot.PythonFunction(1, 1, interpModel)

        # The POD at confidence level is built in getPODCLModel() directly

        # remove the interactive plotting
        plt.ioff()

    def getOutputDOE(self):
        """
        Accessor to the final output values of the DOE.
        """
        if self._boxCox:
            invBoxCox = self._boxCoxTransform.getInverse()
            return invBoxCox(self._signals) - self._shift
        else:
            return self._signals

    def getInputDOE(self):
        """
        Accessor to the final input values of the DOE.
        """
        return self._input 

    def getCandidateSize(self):
        """
        Accessor to the number of candidate points.

        Returns
        -------
        size : int
            The number of candidate points on which the criterion is computed.
        """
        return self._candidateSize

    def setCandidateSize(self, size):
        """
        Accessor to the number of candidate points.

        Parameters
        ----------
        size : int
            The number of candidate points on which the criterion is computed
        """
        self._candidateSize = size

    def getGraphActive(self):
        """
        Accessor to the graph verbosity.

        Returns
        -------
        graphVerbose : bool
            Enable or disable the display of the POD graph at each iteration. Default
            is False. 
        """
        return self._graph

    def setGraphActive(self, graphVerbose, probabilityLevel=None, confidenceLevel=None,
                       directory=None):
        """
        Accessor to the graph verbosity.

        Parameters
        ----------
        graphVerbose : bool
            Enable or disable the display of the POD graph at each iteration.
        probabilityLevel : float
            The probability level for which the defect size is computed. Default
            is None.
        confidenceLevel : float
            The confidence level associated to the given probability level the
            defect size is computed. Default is None.
        directory : string
            Directory where to save the graphs as png files.
        """
        if type(graphVerbose) is not bool:
            raise TypeError("The parameter 'graphVerbose' is not a bool.")
        elif type(directory) is not str and directory is not None:
            raise TypeError("The parameter 'directory' is not a string.")
        else:
            self._graph = graphVerbose
            self._probabilityLevel = probabilityLevel
            self._confidenceLevel = confidenceLevel
            self._graphDirectory = directory

    def _mergeDefectInX(self, defect, X):
        """
        defect : scalar of the defect value
        X : sample without the defect column
        """
        size = X.getSize()
        dim = X.getDimension() + 1
        samplePred = ot.Sample(size, dim)
        samplePred[:, 0] = ot.Sample(size, [defect])
        samplePred[:, 1:] = X
        return samplePred
