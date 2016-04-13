# -*- coding: utf-8 -*-
# -*- Python -*-

__all__ = []

import openturns as ot
import numpy as np

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

        # Assertions on parameters
        assert (self._size >=3), "Not enough observations."
        assert (self._size == self._outputSample.getSize()), \
                "InputSample and outputSample must have the same size."
        assert (self._outputSample.getDimension() == 1), "OutputSample must be of dimension 1."


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
