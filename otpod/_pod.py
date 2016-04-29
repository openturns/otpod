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
        assert (self._outputSample.getDimension() == 1), "Dimension outputSample must be 1."


    def getSimulationSize(self):
        """
        Accessor to the simulation size.

        Returns
        ----------
        size : int
            The size of the simulation used to compute the confidence interval.
        """
        return self._simulationSize

    def setSimulationSize(self, size):
        """
        Accessor to the simulation size.

        Parameters
        ----------
        size : int
            The size of the simulation used to compute the confidence interval.
        """
        self._simulationSize = size

################################################################################
################ Common methods called inside subclass #########################
################################################################################

    def _computeDetectionSize(self, model, modelCl, probabilityLevel,
                              confidenceLevel=None, defectMin=None, defectMax=None):
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
            A NumericalPointWithDescription containing the detection size
            computed at the given probability level and confidence level if provided.
        """

        if defectMin is None:
            defectMin = self._inputSample.getMin()[0]
        if defectMax is None:
            defectMax = self._inputSample.getMax()[0]

        # compute 'a90'
        if not (model([defectMin])[0] <= probabilityLevel <= model([defectMax])[0]):
            raise Exception('The POD model does not contain, for the given ' + \
                             'defect interval, the wanted probability level.')
        detectionSize = ot.NumericalPointWithDescription(1, ot.Brent().solve(model,
                                        probabilityLevel, defectMin, defectMax))
        description = ['a'+str(int(probabilityLevel*100))]

        # compute 'a90_95'
        if confidenceLevel is not None:
            if not (modelCl([defectMin])[0] <= probabilityLevel <= modelCl([defectMax])[0]):
                raise Exception('The POD model at the confidence level does not '+\
                                'contain, for the given defect interval, the '+\
                                'wanted probability level.')
            detectionSize.add(ot.Brent().solve(modelCl, probabilityLevel,
                                               defectMin, defectMax))
            description.append('a'+str(int(probabilityLevel*100))+'/'\
                                                +str(int(confidenceLevel*100)))
        # add description to the NumericalPoint
        detectionSize.setDescription(description)
        return detectionSize

    def _drawPOD(self, PODmodel, PODmodelCl=None, probabilityLevel=None,
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
            max values of the input sample.
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


    def _drawValidationGraph(self, target, prediction):
        """
        Draw the validation graph of the metamodel.

        Parameters
        ----------
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
        target = np.hstack(target)
        prediction = np.hstack(prediction)
        fig, ax = plt.subplots(figsize=(8, 8))
        # compute boundaries of the graph assuming target and prediction > 0 
        _min = 0.99*np.concatenate([target, prediction]).min()
        _max = 1.01*np.concatenate([target, prediction]).max()
        ax.plot([_min, _max], [_min, _max], 'r-', lw=0.5)
        ax.plot(target, prediction, 'b.', ms=9)
        ax.grid()
        ax.set_xlim(_min, _max)
        ax.set_ylim(_min, _max)
        ax.set_xlabel('Signals')
        ax.set_aspect(1.)
        return fig, ax

    def _run(self, inputSample, outputSample, detection, noiseThres,
             saturationThres, boxCox, censored):
        """
        Run common preliminary analysis to all methods to build POD. 
        """
         #################### Filter censored data ##############################
        if censored:
            # Filter censored data
            inputSample, inputSampleNoise, inputSampleSat, signals = \
                DataHandling.filterCensoredData(inputSample, outputSample,
                              noiseThres, saturationThres)
        else:
            inputSample, signals = inputSample, outputSample
            inputSampleNoise, inputSampleSat = None, None

        ###################### Box Cox transformation ##########################
        # Compute Box Cox if enabled
        if boxCox:
            # optimization required, get optimal lambda without graph
            self._lambdaBoxCox, self._graphBoxCox = computeBoxCox(inputSample, signals)

            # Transformation of data
            boxCoxTransform = ot.BoxCoxTransform([self._lambdaBoxCox])
            signals = boxCoxTransform(signals)
            if censored:
                if noiseThres is not None:
                    noiseThres = boxCoxTransform([noiseThres])[0]
                if saturationThres is not None:
                    saturationThres = boxCoxTransform([saturationThres])[0]
            detectionBoxCox = boxCoxTransform([detection])[0]
        else:
            detectionBoxCox = detection
            self._lambdaBoxCox = None

        return {'inputSample':inputSample, 'signals':signals,
                'detectionBoxCox':detectionBoxCox}

    def drawBoxCoxLikelihood(self, name=None):
        """
        Draw the loglikelihood versus the Box Cox parameter.

        Parameters
        ----------
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

        Notes
        -----
        This method is available only when the parameter *boxCox* is set to True.
        """

        # Check is the censored model exists when asking for it 
        if not self._boxCox:
            raise Exception('The Box Cox transformation is not enabled.')

        fig, ax = plt.subplots(figsize=(8, 6))
        # get the graph from the method 'computeBoxCox'
        View(self._graphBoxCox, axes=[ax])
        ax.set_xlabel('Box Cox parameter')
        ax.set_ylabel('LogLikelihood')
        ax.set_title('Loglikelihood versus Box Cox parameter')

        if name is not None:
            fig.savefig(name, bbox_inches='tight', transparent=True)

        return fig, ax

    def getBoxCoxParameter(self):
        """
        Accessor to the Box Cox parameter. 

        Returns
        -------
        lambdaBoxCox : float
            The Box Cox parameter used to transform the data. If the transformation
            is not enabled None is returned. 
        """
        return self._lambdaBoxCox