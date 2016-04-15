# -*- coding: utf-8 -*-
# -*- Python -*-

__all__ = []

import openturns as ot
import numpy as np
from ._math_tools import computeBoxCox, censureFilter


class _Results():
    """
    This class contains the result of the run. Instances are created
    for uncensored data or if needed for censored data.
    """
    def __init__(self):
        pass


class POD(object):
    """
    Base class to compute the POD with the children class.
    """

    def __init__(self, inputSample, outputSample, detection, noiseThres,
                 saturationThres, boxCox):

        self._simulationSize = 1000


        # inherited attributes
        self._inputSample = ot.NumericalSample(np.vstack(inputSample))
        self._outputSample = ot.NumericalSample(np.vstack(outputSample))
        self._detection = detection
        self._noiseThres = noiseThres
        self._saturationThres = saturationThres

        # if Box Cox is a float the transformation is enabled with the given value
        if type(boxCox) is float:
            self._lambdaBoxCox = boxCox
            self._boxCox = True
        else:
            self._lambdaBoxCox = None
            self._boxCox = boxCox

        self._size = self._inputSample.getSize()
        self._dim = self._inputSample.getDimension()

        #################### check attributes for censoring ####################
        # Add flag to tell if censored data must taken into account or not.
        if self._noiseThres is not None or self._saturationThres is not None:
            # flag to tell censoring is enabled
            self._censored = True
            # Results instances are created for both cases.
            self._resultsCens = _Results()
            self._resultsUnc = _Results()
        else:
            self._censored = False
            # Results instance is created only for uncensored case.
            self._resultsUnc = _Results()

        # Assertions on parameters
        assert (self._size >=3), "Not enough observations."
        assert (self._size == self._outputSample.getSize()), \
                "InputSample and outputSample must have the same size."
        assert (self._outputSample.getDimension() == 1), "OutputSample must be of dimension 1."

    def _run(self):
        """
        Run common preliminary analysis to all methods to build POD. 
        """
        #################### Filter censored data ##############################
        if self._censored:
            # check if one sided censoring
            if self._noiseThres is None:
                noiseThres = -ot.sys.float_info.max
            if self._saturationThres is None:
                saturationThres = ot.sys.float_info.max
            # Filter censored data
            defects, defectsNoise, defectsSat, signals = \
                censureFilter(self._inputSample, self._outputSample,
                              self._noiseThres, self._saturationThres)
        else:
            defects, signals = self._inputSample, self._outputSample
            defectsNoise = None
            defectsSat = None

        defectsSize = defects.getSize()

        ###################### Box Cox transformation ##########################
        # Compute Box Cox if enabled
        if self._boxCox:
            # optimization required, get optimal lambda without graph
            self._lambdaBoxCox, graph = computeBoxCox(defects, signals)

            # Transformation of data
            boxCoxTransform = ot.BoxCoxTransform([self._lambdaBoxCox])
            signals = boxCoxTransform(signals)
            if self._censored:
                if self._noiseThres is not None:
                    noiseThres = boxCoxTransform([self._noiseThres])[0]
                if self._saturationThres is not None:
                    saturationThres = boxCoxTransform([self._saturationThres])[0]
            detection = boxCoxTransform([self._detection])[0]
        else:
            signals = signals
            noiseThres = self._noiseThres
            saturationThres = self._saturationThres
            detection = self._detection

        return defects, defectsNoise, defectsSat, signals, detection, \
               noiseThres, saturationThres

    def getSimulationSize(self):
        """
        Accessor to the simulation size.
        """
        return self._simulationSize

    def setSimulationSize(self, size):
        """
        Accessor to the simulation size

        Parameters
        ----------
        size : int
            The size of the simulation used to compute the confidence interval.
        """
        self._simulationSize = size

    def _computeDetectionSize(self, model, probabilityLevel, defectMin, defectMax):
        return ot.Brent().solve(model, probabilityLevel, defectMin, defectMax)
