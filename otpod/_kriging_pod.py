# -*- coding: utf-8 -*-
# -*- Python -*-

__all__ = ['KrigingPOD']

import openturns as ot
import numpy as np
from ._pod import POD
from scipy.interpolate import interp1d
from _decorator import DocInherit, keepingArgs
from _progress_bar import updateProgress
from _kriging_tools import estimKrigingTheta, computeLOO, computeQ2, computePODSamplePerDefect
import logging

class KrigingPOD(POD):
    """
    Kriging based POD.

    **Available constructor:**

    KrigingPOD(*inputSample, outputSample, detection, noiseThres,
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

    def __init__(self, inputSample=None, outputSample=None, detection=None, noiseThres=None,
                 saturationThres=None, boxCox=False):

        # initialize the POD class
        super(KrigingPOD, self).__init__(inputSample, outputSample,
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

        self._userKriging = False
        self._krigingResult = None
        self._distribution = None
        self._basis = None
        self._covarianceModel = None
        self._samplingSize = 5000
        self._defectSizes = None
        self._initialStartSize = 1000
        self._verbose = True

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

        # define the defect sizes for the interpolation function if not defined
        if self._defectSizes is None:
            self._defectNumber = 20
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

        # run the chaos algorithm and get result if not given
        if not self._userKriging:
            if self._verbose:
                print('Start optimizing covariance model parameters...')
            # build the kriging algorithm without optimizer
            self._algoKriging = self._buildKrigingAlgo(self._input, self._signals)
            # optimize the covariance model parameters and return the kriging
            # algorithm with the run launched
            covDim = self._algoKriging.getResult().getCovarianceModel().getScale().getDimension()
            lowerBound = [0.001] * covDim
            upperBound = [50] * covDim               
            self._algoKriging = estimKrigingTheta(self._algoKriging,
                                                  lowerBound, upperBound,
                                                  self._initialStartSize)
            if self._verbose:
                print('Kriging optimizer completed')
            self._krigingResult = self._algoKriging.getResult()

        # compute the Q2
        self._Q2 = computeQ2(self._input, self._signals, self._krigingResult)

        if self._distribution is None:
            inputMin = self._input.getMin()
            inputMax = self._input.getMax()
            marginals = [ot.Uniform(inputMin[i], inputMax[i]) for i in range(self._dim)]
            self._distribution = ot.ComposedDistribution(marginals)

        # compute the sample containing the POD values for all defect 
        self._PODPerDefect = ot.NumericalSample(self._simulationSize *
                                         self._samplingSize, self._defectNumber)
        for i, defect in enumerate(self._defectSizes):
            self._PODPerDefect[:, i] = computePODSamplePerDefect(defect,
                self._detectionBoxCox, self._krigingResult, self._distribution,
                self._simulationSize, self._samplingSize)
            if self._verbose:
                updateProgress(i, self._defectNumber, 'Computing POD per defect')

        # compute the mean POD 
        meanPOD = self._PODPerDefect.computeMean()
        # create the interpolate function of the POD model
        interpModel = interp1d(self._defectSizes, np.array(meanPOD), kind='linear')
        self._PODmodel = ot.PythonFunction(1, 1, interpModel)

        # The POD at confidence level is built in getPODCLModel() directly


    def getPODModel(self):
        """
        Accessor to the POD model.

        Returns
        -------
        PODModel : :py:class:`openturns.NumericalMathFunction`
            The function which computes the probability of detection for a given
            defect value.
        """
        return self._PODmodel

    def getPODCLModel(self, confidenceLevel=0.95):
        """
        Accessor to the POD model at a given confidence level.

        Parameters
        ----------
        confidenceLevel : float
            The confidence level the POD must be computed. Default is 0.95

        Returns
        -------
        PODModelCl : :py:class:`openturns.NumericalMathFunction`
            The function which computes the probability of detection for a given
            defect value at the confidence level given as parameter.
        """
        # Compute the quantile at the given confidence level for each
        # defect quantile and build the interpolate function.
        PODQuantile = self._PODPerDefect.computeQuantilePerComponent(
                                                            1. - confidenceLevel)
        interpModel = interp1d(self._defectSizes, PODQuantile, kind='linear')
        PODmodelCl = ot.PythonFunction(1, 1, interpModel)

        return PODmodelCl

    @DocInherit # decorator to inherit the docstring from POD class
    @keepingArgs # decorator to keep the real signature
    def computeDetectionSize(self, probabilityLevel, confidenceLevel=None):
        return self._computeDetectionSize(self.getPODModel(),
                                          self.getPODCLModel(confidenceLevel),
                                          probabilityLevel,
                                          confidenceLevel,
                                          np.min(self._defectSizes),
                                          np.max(self._defectSizes))

    @DocInherit # decorator to inherit the docstring from POD class
    @keepingArgs # decorator to keep the real signature
    def drawPOD(self, probabilityLevel=None, confidenceLevel=None, defectMin=None,
                defectMax=None, nbPt=100, name=None):

        if defectMin is None:
            defectMin = np.min(self._defectSizes)
        else:
            if defectMin < np.min(self._defectSizes):
                raise ValueError('DefectMin must be greater than the minimum ' + \
                                 'of the given defect sizes.')
            if defectMin > np.max(self._defectSizes):
                raise ValueError('DefectMin must be lower than the maximum ' + \
                                 'of the given defect sizes.')
        if defectMax is None:
            defectMax = np.max(self._defectSizes)
        else:
            if defectMax > np.max(self._defectSizes):
                raise ValueError('DefectMax must be lower than the maximum ' + \
                                 'of the given defect sizes.')
            if defectMax < np.min(self._defectSizes):
                raise ValueError('DefectMax must be greater than the maximum ' + \
                                 'of the given defect sizes.')

        if confidenceLevel is None:
            fig, ax = self._drawPOD(self.getPODModel(), None,
                                probabilityLevel, confidenceLevel, defectMin,
                                defectMax, nbPt, name)
        elif confidenceLevel is not None:
            fig, ax = self._drawPOD(self.getPODModel(), self.getPODCLModel(confidenceLevel),
                    probabilityLevel, confidenceLevel, defectMin,
                    defectMax, nbPt, name)

        ax.set_title('POD - Kriging model')
        if name is not None:
            fig.savefig(name, bbox_inches='tight', transparent=True)

        return fig, ax

    @DocInherit # decorator to inherit the docstring from POD class
    @keepingArgs # decorator to keep the real signature
    def drawValidationGraph(self, name=None):

        y_loo = computeLOO(self._input, self._signals, self._krigingResult)
        fig, ax = self._drawValidationGraph(self._signals, y_loo)
        ax.set_title("Validation of the Kriging model")
        ax.set_ylabel('Predicted leave one out signals')

        if name is not None:
            fig.savefig(name, bbox_inches='tight', transparent=True)
        return fig, ax

    def getQ2(self):
        """
        Accessor to the Q2 value. 

        Returns
        -------
        Q2 : float
            The Q2 value computed analytically using Dubrule (1983) technique.
        """
        return self._Q2

    def getSamplingSize(self):
        """
        Accessor to the Monte Carlo sampling size.

        Returns
        -------
        size : int
            The size of the Monte Carlo simulation used to compute the POD for
            each defect size.
        """
        return self._samplingSize

    def setSamplingSize(self, size):
        """
        Accessor to the Monte Carlo sampling size.

        Parameters
        ----------
        size : int
            The size of the Monte Carlo simulation used to compute the POD for
            each defect size.
        """
        self._samplingSize = size

    def getInitialStartSize(self):
        """
        Accessor to the initial random search size.

        Returns
        -------
        size : int
            The size of the initial random search to find the best loglikelihood
            value to start the TNC algorithm to optimize the covariance model
            parameters. Default is 1000.
        """
        return self._initialStartSize

    def setInitialStartSize(self, size):
        """
        Accessor to the initial random search size.

        Parameters
        ----------
        size : int
            The size of the initial random search to find the best loglikelihood
            value to start the TNC algorithm to optimize the covariance model
            parameters.
        """
        self._initialStartSize = size

    def getDefectSizes(self):
        """
        Accessor to the defect size where POD is computed.

        Returns
        -------
        defectSize : sequence of float
            The defect sizes where the Monte Carlo simulation is performed to
            compute the POD.
        """
        return self._defectSizes

    def setDefectSizes(self, size):
        """
        Accessor to the defect size where POD is computed.

        Parameters
        ----------
        defectSize : sequence of float
            The defect sizes where the Monte Carlo simulation is performed to
            compute the POD.
        """
        size = np.hstack(np.array(size))
        size.sort()
        self._defectSizes = size.copy()
        minMin = self._input[:, 0].getMin()[0]
        maxMax = self._input[:, 0].getMax()[0]
        if size.max() > maxMax or size.min() < minMin:
            raise ValueError('Defect sizes must range between ' + \
                             '{:0.4f} '.format(np.ceil(minMin*10000)/10000) + \
                             'and {:0.4f}.'.format(np.floor(maxMax*10000)/10000))
        self._defectNumber = self._defectSizes.shape[0]

    def setDistribution(self, distribution):
        """
        Accessor to the parameters distribution. 

        Parameters
        ----------
        distribution : :py:class:`openturns.ComposedDistribution`
            The input parameters distribution used for the Monte Carlo simulation.
        """
        try:
            ot.ComposedDistribution(distribution)
        except NotImplementedError:
            raise Exception('The given parameter is not a ComposedDistribution.')
        self._distribution = distribution

    def getDistribution(self):
        """
        Accessor to the parameters distribution. 

        Returns
        -------
        distribution : :py:class:`openturns.ComposedDistribution`
            The input parameters distribution used for the Monte Carlo simulation.
            Default is a Uniform distribution for all parameters.
        """
        if self._distribution is None:
            print 'The run method must be launched first.'
        else:
            return self._distribution

    def setBasis(self, basis):
        """
        Accessor to the kriging basis. 

        Parameters
        ----------
        basis : :py:class:`openturns.Basis`
            The basis used as trend in the kriging model.
        """
        try:
            ot.Basis(basis)
        except NotImplementedError:
            raise Exception('The given parameter is not a Basis.')
        self._basis = basis

    def getBasis(self):
        """
        Accessor to the kriging basis. 

        Returns
        -------
        basis : :py:class:`openturns.Basis`
            The basis used as trend in the kriging model. Default is a linear
            basis for the defect and constant for the other parameters.
        """
        if self._basis is None:
            print 'The run method must be launched first.'
        else:
            return self._basis

    def setCovarianceModel(self, covarianceModel):
        """
        Accessor to the kriging covariance model. 

        Parameters
        ----------
        covarianceModel : :py:class:`openturns.CovarianceModel`
            The covariance model in the kriging model.
        """
        try:
            ot.CovarianceModel(covarianceModel)
        except NotImplementedError:
            raise Exception('The given parameter is not a CovarianceModel.')
        self._covarianceModel = covarianceModel

    def getCovarianceModel(self):
        """
        Accessor to the kriging covariance model. 

        Returns
        -------
        covarianceModel : :py:class:`openturns.CovarianceModel`
            The covariance model in the kriging model. Default is an anisotropic
            squared exponential covariance model.
        """
        if self._covarianceModel is None:
            print 'The run method must be launched first.'
        else:
            return self._covarianceModel

    def setKrigingResult(self, result):
        """
        Accessor to the kriging result.

        Parameters
        ----------
        result : :py:class:`openturns.KrigingResult`
            The kriging result.
        """
        try:
            ot.KrigingResult(result)
        except NotImplementedError:
            raise Exception('The given parameter is not an KrigingResult.')
        self._krigingResult = result
        self._userKriging = True

    def getKrigingResult(self):
        """
        Accessor to the kriging result.

        Returns
        -------
        result : :py:class:`openturns.KrigingResult`
            The kriging result.
        """
        if self._krigingResult is None:
            print 'The run method must be launched first.'
        else:
            return self._krigingResult

    def getVerbose(self):
        """
        Accessor to the verbosity.

        Returns
        -------
        verbose : bool
            Enable or disable the verbosity. Default is True. 
        """
        return self._verbose

    def setVerbose(self, verbose):
        """
        Accessor to the verbosity.

        Parameters
        ----------
        verbose : bool
            Enable or disable the verbosity.
        """
        if type(verbose) is not bool:
            raise TypeError('The parameter is not a bool.')
        else:
            self._verbose = verbose


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
