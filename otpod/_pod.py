# -*- coding: utf-8 -*-
# -*- Python -*-

__all__ = []

import openturns as ot
from openturns.viewer import View
import matplotlib.pyplot as plt
import numpy as np
from ._math_tools import computeBoxCox, DataHandling

class POD(object):
    """
    Base class to compute the POD with the subclass.
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
        else:
            self._censored = False

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
            # Filter censored data
            defects, defectsNoise, defectsSat, signals = \
                DataHandling.filterCensoredData(self._inputSample, self._outputSample,
                              self._noiseThres, self._saturationThres)
        else:
            defects, signals = self._inputSample, self._outputSample
            defectsNoise = None
            defectsSat = None

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

################################################################################
################ Common methods called inside subclass #########################
################################################################################

    def _computeDetectionSize(self, model, modelCl, probabilityLevel, confidenceLevel=None):
        """
        Compute the detection size for a given probability level.

        Parameters
        ----------
        probabilityLevel : float
            The probability level for which the defect size is computed.
        confidenceLevel : float
            The confidence level associated to the given probability level the
            defect size is computed. Default is None.

        Returns
        -------
        result : collection of :py:class:`openturns.NumericalPointWithDescription`
            A list of NumericalPointWithDescription containing the detection size
            computing for each case.
        """

        defectMin = self._inputSample.getMin()[0]
        defectMax = self._inputSample.getMax()[0]

        # compute 'a90'
        model = self.getPODModel()
        detectionSize = ot.NumericalPointWithDescription(1, ot.Brent().solve(model,
                                        probabilityLevel, defectMin, defectMax))
        description = ['a'+str(int(probabilityLevel*100))]

        # compute 'a90_95'
        if confidenceLevel is not None:
            model = self.getPODCLModel(confidenceLevel=confidenceLevel)
            detectionSize.add(ot.Brent().solve(model, probabilityLevel,
                                               defectMin, defectMax))
            description.append('a'+str(int(probabilityLevel*100))+'/'\
                                                +str(int(confidenceLevel*100)))
        # add description to the NumericalPoint
        detectionSize.setDescription(description)
        return detectionSize

    def _drawPOD(self, PODmodel, PODmodelCl, probabilityLevel=None,
                 confidenceLevel=None, defectMin=None, defectMax=None,
                 nbPt=100, name=None):
        """
        Draw the POD curve.

        Parameters
        ----------
        probabilityLevel : float
            The probability level for which the defect size is computed. Default
            is None.
        confidenceLevel : float
            The confidence level associated to the given probability level the
            defect size is computed. Default is None.
        defectMin, defectMax : float
            Define the interval where the curve is plotted. Default : min and
            max values of the inputSample.
        nbPt : int
            The number of points to draw the curves. Default is 100.
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

        if defectMin is None:
            defectMin = self._inputSample.getMin()[0]
        if defectMax is None:
            defectMax = self._inputSample.getMax()[0]

        fig, ax = plt.subplots(figsize=(8, 6))
        # POD model graph
        View(PODmodel.draw(defectMin, defectMax, nbPt), axes=[ax],
            plot_kwargs={'color':'red', 'label':'POD'})

        if confidenceLevel is not None:
            # POD at confidence level graph
            View(PODmodelCl.draw(defectMin, defectMax, nbPt), axes=[ax],
                plot_kwargs={'color':'blue', 'label':'POD at confidence level '+\
                                                      str(confidenceLevel)})
        if probabilityLevel is not None:
            # horizontal line at the given probability level
            ax.hlines(probabilityLevel, defectMin, defectMax, 'black', 'solid',
                      'Probability level '+str(probabilityLevel))

            # compute detection size at the given probability level
            detectionSize = self.computeDetectionSize(probabilityLevel, confidenceLevel)
            ax.vlines(detectionSize[0], 0., probabilityLevel, 'red', 'dashed',
                      'a'+str(int(probabilityLevel*100))+' : '+str(round(detectionSize[0], 3)))
            if confidenceLevel is not None:
                ax.vlines(detectionSize[1], 0., probabilityLevel, 'blue', 'dashed',
                      'a'+str(int(probabilityLevel*100))+'/'+str(int(confidenceLevel*100))+\
                      ' : '+str(round(detectionSize[1], 3)))

        ax.legend(loc='lower right')
        ax.set_xlabel('Defects')
        ax.set_ylabel('POD')
        return fig, ax