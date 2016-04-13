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

        self._simulationSize = 100


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

    # def _runComputeBoxCox(self):
    #     # Compute Box Cox if enabled
    #     if self._boxCox:
    #         if self._lambdaBoxCox is None:
    #             # optimization required, get optimal lambda and graph
    #             self._lambdaBoxCox, self._graphBoxCox = computeBoxCox(defects, signals)

    #         # Transformation of data
    #         boxCoxTransform = ot.BoxCoxTransform([self._lambdaBoxCox])
    #         signals = boxCoxTransform(signals)
    #         if self._noiseThres is not None:
    #             noiseThres = boxCoxTransform([self._noiseThres])[0]
    #         if self._saturationThres is not None:
    #             saturationThres = boxCoxTransform([self._saturationThres])[0]
    #     else:
    #         noiseThres = self._noiseThres
    #         saturationThres = self._saturationThres