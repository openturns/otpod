# -*- coding: utf-8 -*-
# -*- Python -*-

__all__ = ['UnivariateLinearModelAnalysis']

import openturns as ot

class UnivariateLinearModelAnalysis():

    """
    Build a linear regression model and perform hypothesis test on the residuals.
    
    **Available constructors**

    UnivariateLinearModelAnalysis(*inputSample, outputSample*)

    UnivariateLinearModelAnalysis(*inputSample, outputSample, noiseThres,
    saturationThres, resDistFact, boxCox*)

    Parameters
    ----------
    inputSample : 2-d sequence of float
        Vector of the defects size, of dimension 1.
    outputSample : 2-d sequence of float
        Vector of the signals, of dimension 1.
    noiseThres : float
        Low censored value
    """

    def __init__(self, inputSample, outputSample, noiseThres=None,
                 saturationThres=None, resDistFact=ot.NormalFactory(),
                 boxCox=False):

        self.inputSample = inputSample
        self.outputSample = outputSample
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
