# -*- coding: utf-8 -*-
# -*- Python -*-

__all__ = ['UnivariateLinearRegressionPOD']


class UnivariateLinearRegressionPOD():

    """
    doc 

    """

    def __init__(self, inputSample, outputSample, detection, noiseThres=None,
                 saturationThres=None, resDistFact=None,
                 boxCox=False):

        self.inputSample = inputSample
        self.outputSample = outputSample
        self.detection = detection
        self.noiseThres = noiseThres
        self.saturationThres = saturationThres
        self.resDistFact = resDistFact
        self.boxCox = boxCox

    def run(self):
        """
        Bla bla bla

        Parameters
        ----------
        sdfs : float
            dfsdf
        oko : bool
            ture
        """
        pass
