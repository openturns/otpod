# -*- coding: utf-8 -*-
# -*- Python -*-

__all__ = ['QuantileRegressionPOD']

import openturns as ot
import numpy as np
from ._pod import POD
from statsmodels.regression.quantile_regression import QuantReg
from scipy.interpolate import interp1d
from ._decorator import DocInherit, keepingArgs
from ._progress_bar import updateProgress
import matplotlib.pyplot as plt
import logging


class QuantileRegressionPOD(POD):
    """
    Quantile regression based POD.

    **Available constructor:**

    QuantileRegressionPOD(*inputSample, outputSample, detection, noiseThres,
    saturationThres, boxCox*)

    Parameters
    ----------
    inputSample : 2-d sequence of float
        Vector of the defect sizes, of dimension 1.
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

    Notes
    -----
    This class aims at building the POD based on a quantile regression
    model. The return POD model corresponds with an interpolate function built
    with the defect values computed for the given quantile as parameters. The
    default is 21 quantile values from 0.05 to 0.98. They can be user-defined
    using the method *setQuantile*.

    The confidence level is computed by bootstrap. The POD model at the given
    confidence level is also an interpolate function based on the defect quantile
    value computed at the given confidence level.

    The computeDetectionSize method calls the real quantile regression
    at the given probability level.

    A progress bar is shown if the verbosity is enabled. It can be disabled using
    the method *setVerbose*.
    """

    def __init__(self, inputSample=None, outputSample=None, detection=None, noiseThres=None,
                 saturationThres=None, boxCox=False):

        self._quantile = np.linspace(0.05, 0.98, 21)
        self._verbose = True

        # initialize the POD class
        super(QuantileRegressionPOD, self).__init__(inputSample, outputSample,
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
        
        # assertion input dimension is 1
        assert (self._dim == 1), "Dimension of inputSample must be 1."

        if self._censored:
            logging.info('Censored data are not taken into account : the quantile ' + \
                         'regression model is only performed on filtered data.')


    def run(self):
        """
        Build the POD models.

        Notes
        -----
        This method build the quantile regression model. First the censored data
        are filtered if needed. The Box Cox transformation is performed if it is
        enabled. Then it builds the POD model for given data and computes using
        bootstrap all the defects quantile needed to build the POD model at the
        confidence level.
        """

        # Run the preliminary run of the POD class
        result = self._run(self._inputSample, self._outputSample, self._detection,
                           self._noiseThres, self._saturationThres, self._boxCox,
                           self._censored)

        # get some results
        self._defects = result['inputSample']
        self._signals = result['signals']
        self._detectionBoxCox = result['detectionBoxCox']

        defectsSize = self._defects.getSize()

        # create the quantile regression object
        X = ot.NumericalSample(defectsSize, [1, 0])
        X[:, 1] = self._defects
        self._algoQuantReg = QuantReg(np.array(self._signals), np.array(X))

        # Compute the defect quantile
        defectMax = self._defects.getMax()[0]
        defectList = []
        for probLevel in self._quantile:
            # fit the quantile regression and return the NMF
            model = self._buildModel(1. - probLevel)
            # Solve the model == detectionBoxCox with defects 
            # boundaries = [0, defectMax]
            defectList.append(ot.Brent().solve(model, self._detectionBoxCox,
                                               0, defectMax))
        # create support of the interpolating function including
        # point (0, 0) and point (defectMax, max(quantile))
        xvalue = np.hstack([0, defectList, defectMax])
        yvalue = np.hstack([0., self._quantile, self._quantile.max()])
        interpModel = interp1d(xvalue, yvalue, kind='linear')
        self._PODmodel = ot.PythonFunction(1, 1, interpModel)


        ############ Confidence interval with bootstrap ########################
        # Compute a NsimulationSize defect sizes for all quantiles
        data = ot.NumericalSample(self._size, 2)
        data[:, 0] = self._inputSample
        data[:, 1] = self._outputSample
        # bootstrap of the data
        bootstrapExp = ot.BootstrapExperiment(data)
        # create a numerical sample which contains for all simulations the 
        # defect quantile value. The goal is to compute the QuantilePerComponent
        # of the simulation for each defect quantile (columns)
        self._defectsPerQuantile = ot.NumericalSample(self._simulationSize, self._quantile.size)
        for i in range(self._simulationSize):
            # generate a sample with replacement within data of the same size
            bootstrapData = bootstrapExp.generate()
            # run the preliminary analysis : censore checking and box cox
            result = self._run(bootstrapData[:,0], bootstrapData[:,1], self._detection,
                               self._noiseThres, self._saturationThres,
                               self._boxCox, self._censored)

            # get some results
            defects = result['inputSample']
            signals = result['signals']
            detectionBoxCox = result['detectionBoxCox']
            defectsSize = defects.getSize()

            # new quantile regression algorithm
            X = ot.NumericalSample(defectsSize, [1, 0])
            X[:, 1] = defects
            algoQuantReg = QuantReg(np.array(signals), np.array(X))

            # compute the quantile defects
            defectMax = defects.getMax()[0]
            defectList = []
            for probLevel in self._quantile:
                fit = algoQuantReg.fit(1. - probLevel, max_iter=300, p_tol=1e-2)
                def model(x):
                    X = ot.NumericalPoint([1, x[0]])
                    return ot.NumericalPoint(fit.predict(X))
                model = ot.PythonFunction(1, 1, model)
                # Solve the model == detectionBoxCox with defects 
                # boundaries = [-infinity, defectMax] : it allows negative defects
                # when for small prob level, there is no intersection with
                # the detection threshold
                defectList.append(ot.Brent().solve(model, detectionBoxCox,
                                                   -ot.SpecFunc.MaxNumericalScalar,
                                                   defectMax))
            # add the quantile in the numerical sample as the ith simulation
            self._defectsPerQuantile[i, :] = defectList
            if self._verbose:
                updateProgress(i, self._simulationSize, 'Computing defect quantile')

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
        defectsQuantile = self._defectsPerQuantile.computeQuantilePerComponent(
                                                                confidenceLevel)

        xvalue = np.hstack([0, np.array(defectsQuantile), self._defects.getMax()[0]])
        yvalue = np.hstack([0., self._quantile, self._quantile.max()])
        interpModel = interp1d(xvalue, yvalue, kind='linear')
        PODmodelCl = ot.PythonFunction(1, 1, interpModel)

        return PODmodelCl

    def getR2(self, quantile):
        """
        Accessor to the pseudo R2 value.
        
        Parameters
        ----------
        quantile : float
            The quantile value for which the regression is performed.       

        Returns
        -------
        R2 : float
            The pseudo R2 value.
        """
        return self._algoQuantReg.fit(quantile).prsquared

    def getQuantile(self):
        """
        Accessor to the quantile list for the regression.
        """
        return self._quantile

    def setQuantile(self, quantile):
        """
        Accessor to the quantile list for the regression.

        Parameters
        ----------
        quantile : sequence of float
            The quantile value for which the regression is performed and the 
            corresponding defect size is computed.
        """
        quantile = np.hstack(np.array(quantile))
        quantile.sort()
        if quantile.max() >= 1 or quantile.min() <= 0:
            raise ValueError('Quantile values must range between ]0, 1[.')
        self._quantile = quantile

    @DocInherit # decorator to inherit the docstring from POD class
    @keepingArgs # decorator to keep the real signature
    def computeDetectionSize(self, probabilityLevel, confidenceLevel=None):
        defectMin = self._defects.getMin()[0]
        defectMax = self._defects.getMax()[0]
        # compute 'a90'
        model = self._buildModel(1. - probabilityLevel)
        detectionSize = ot.NumericalPointWithDescription(1, ot.Brent().solve(
                                    model, self._detectionBoxCox, 0, defectMax))
        description = ['a'+str(int(probabilityLevel*100))]

        # compute 'a90_95'
        if confidenceLevel is not None:
            detectionSize.add(ot.Brent().solve(self.getPODCLModel(confidenceLevel),
                                               probabilityLevel,
                                               defectMin, defectMax))
            description.append('a'+str(int(probabilityLevel*100))+'/'\
                                                +str(int(confidenceLevel*100)))
        # add description to the NumericalPoint
        detectionSize.setDescription(description)
        return detectionSize

    @DocInherit # decorator to inherit the docstring from POD class
    @keepingArgs # decorator to keep the real signature
    def drawPOD(self, probabilityLevel=None, confidenceLevel=None, defectMin=None,
                defectMax=None, nbPt=100, name=None):

        if defectMin is None:
            defectMin = np.min(self._defects)
        else:
            if defectMin < np.min(self._defects):
                raise ValueError('DefectMin must be greater than the minimum ' + \
                                 'of the given defect sizes.')
            if defectMin > np.max(self._defects):
                raise ValueError('DefectMin must be lower than the maximum ' + \
                                 'of the given defect sizes.')
        if defectMax is None:
            defectMax = np.max(self._defects)
        else:
            if defectMax > np.max(self._defects):
                raise ValueError('DefectMax must be lower than the maximum ' + \
                                 'of the given defect sizes.')
            if defectMax < np.min(self._defects):
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

        ax.set_title('POD - Quantile regression model')
        if name is not None:
            fig.savefig(name, bbox_inches='tight', transparent=True)

        return fig, ax

    def drawLinearModel(self, probabilityLevel, name=None):
        """
        Draw the quantile regression prediction versus the true data.

        Parameters
        ----------
        probabilityLevel : float
            The probability level for which the quantile regression is performed
        name : string
            name of the figure to be saved with *transparent* option sets to True
            and *bbox_inches='tight'*. It can be only the file name or the 
            full path name. Default is None.

        Returns
        -------
        fig : `matplotlib.figure <http://matplotlib.org/api/figure_api.html>`_
            Matplotlib figure object.
        ax : `matplotlib.axes <http://matplotlib.org/api/axes_api.html>`_
            Matplotlib axes object.
        """

        model = self._algoQuantReg.fit(1. - probabilityLevel)

        defects = self._defects
        signals = self._signals
        fittedSignals = model.fittedvalues

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(defects, signals, 'b.', label='Data', ms=9)
        ax.plot(defects, fittedSignals, 'r-', label='Linear regression model')
        ax.set_xlabel('Defects')
        ax.set_ylabel('Signals')
        ax.set_title('Quantile regression model at level (1 - ' + \
                                        str(probabilityLevel) + ')')
        ax.grid()
        ax.legend(loc='upper left')

        if name is not None:
            fig.savefig(name, bbox_inches='tight', transparent=True)

        return fig, ax

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

    def _buildModel(self, probabilityLevel):
        """
        Build the NumericalMathFunction at the given probabilityLevel. It is
        used in the run and in computeDetectionSize in order to do not use the
        interpolate function.
        """
        fit = self._algoQuantReg.fit(probabilityLevel, max_iter=300, p_tol=1e-2)
        def model(x):
            X = ot.NumericalPoint([1, x[0]])
            return ot.NumericalPoint(fit.predict(X))
        return ot.PythonFunction(1, 1, model)