# -*- coding: utf-8 -*-
# -*- Python -*-

__all__ = ['KrigingPOD']

import openturns as ot
import numpy as np
from ._pod import POD
from scipy.interpolate import interp1d
from ._progress_bar import updateProgress
from ._kriging_tools import KrigingBase
import logging

class KrigingPOD(POD, KrigingBase):
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
    exponential model. Parameters are estimated using a Multi Start TNC
    algorithm, the 100 initial starting points are defined according to a Sobol
    sequence .

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
        self._initialStartSize = 100
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

        # define the default defect sizes for the interpolation function if not defined
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
            algoKriging = self._buildKrigingAlgo(self._input, self._signals)
            # optimize the covariance model parameters and return the kriging
            # algorithm with the run launched
            llDim = algoKriging.getReducedLogLikelihoodFunction().getInputDimension()
            lowerBound = [0.001] * llDim
            upperBound = [50] * llDim
            algoKriging = self._estimKrigingTheta(algoKriging,
                                                  lowerBound, upperBound,
                                                  self._initialStartSize)
            algoKriging.run()
            if self._verbose:
                print('Kriging optimizer completed')
            self._krigingResult = algoKriging.getResult()

        # compute the Q2
        self._Q2 = self._computeQ2(self._input, self._signals, self._krigingResult)
        if self._verbose:
            print('kriging validation Q2 (>0.9): {:0.4f}'.format(self._Q2))

        # set default uniform distribution with min and max of the given defect sizes 
        if self._distribution is None:
            inputMin = self._input.getMin()
            inputMin[0] = np.min(self._defectSizes)
            inputMax = self._input.getMax()
            inputMax[0] = np.max(self._defectSizes)
            marginals = [ot.Uniform(inputMin[i], inputMax[i]) for i in range(self._dim)]
            self._distribution = ot.ComposedDistribution(marginals)

        # compute the sample containing the POD values for all defect 
        self._PODPerDefect = ot.Sample(self._simulationSize *
                                         self._samplingSize, self._defectNumber)
        for i, defect in enumerate(self._defectSizes):
            self._PODPerDefect[:, i] = self._computePODSamplePerDefect(defect,
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
