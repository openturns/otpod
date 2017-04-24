# -*- coding: utf-8 -*-
# -*- Python -*-

__all__ = ['SobolIndices']

import openturns as ot
from openturns.viewer import View
import matplotlib.pyplot as plt
import numpy as np
import logging

class SobolIndices():
    """
    Sensitivity analysis based on Sobol' indices.

    **Available constructor:**

    SobolIndices(*krigingPOD, N*)

    Parameters
    ----------
    krigingPOD : :class:`KrigingPOD` or :class:`AdaptiveSignalPOD`
        The kriging POD object where the run method has been performed.
    N : int
        Size of samples to generate

    returns
    -------
    sa : :class:`openturns.SobolIndicesAlgorithm`
        The openturns object that perform the sensitivity algorithm.

    Notes
    -----
    This class uses the :class:`openturns.SobolIndicesAlgorithm` class
    of OpenTURNS. The sensitivity analysis can be performed only with a POD
    built with a Kriging metamodel.

    The sensitivity analysis allows to computed aggregated Sobol indices for
    the given range of defect sizes. The default defect sizes correspond with
    those defined in the *krigingPod* object. It can be changed using
    :func:`setDefectSizes`.

    The four methods developed in OpenTURNS are availables and can be chosen
    thanks to :func:`setSensitivityMethod`. The default
    method is "Saltelli".

    The result of the sensitivity analysis is available using
    :func:`getSensitivityResult`. It returns the openturns sensitivity object
    from which the sensitivity values are given using proper methods.
    """

    def __init__(self, krigingPOD, N):
        try:
            self._krigingResult = krigingPOD.getKrigingResult()
        except AttributeError:
            raise Exception("Sobol indices can only be computed based on a " + \
                            "POD built with kriging.")

        # dimension is minus 1 to remove the defect parameter
        self._dim = self._krigingResult.getMetaModel().getInputDimension() - 1
        assert (self._dim >=2), "The number of parameters must be greater or " + \
                "equal than 2 to be able to perform the sensitivity analysis."
        self._defectSizes = krigingPOD.getDefectSizes()
        self._defectNumber = self._defectSizes.shape[0]
        self._detectionBoxCox = krigingPOD._detectionBoxCox

        # the distribution of the parameters without the one of the defects.
        tmpDistribution = krigingPOD.getDistribution()
        self._distribution = ot.ComposedDistribution([tmpDistribution.getMarginal(i) for i in range(1, self._dim+1)])

        # number of samples
        self._N = N

        # build the NumericalMathFunction which computed the POD for a given
        # realization and for all defect sizes.
        self._PODaggr = ot.NumericalMathFunction(PODaggr(self._krigingResult,
                            self._dim, self._defectSizes, self._detectionBoxCox))

        # initialize method parameter and sa attribute
        self._method = "Saltelli"
        self._sa = None

    def run(self):
        """
        Compute the Sobol indices with the chosen algorithm. 
        """

        if self._method == "Saltelli":
            self._sa = ot.SaltelliSensitivityAlgorithm(self._distribution, self._N, self._PODaggr, False)
        elif self._method == "Martinez":
            self._sa = ot.MartinezSensitivityAlgorithm(self._distribution, self._N, self._PODaggr, False)
        elif self._method == "Jansen":
            self._sa = ot.JansenSensitivityAlgorithm(self._distribution, self._N, self._PODaggr, False)
        elif self._method == "MauntzKucherenko":
            self._sa = ot.MauntzKucherenkoSensitivityAlgorithm(self._distribution, self._N, self._PODaggr, False)

    def getSensitivityResult(self):
        """
        Accessor to the OpenTURNS sensitivity object.

        Returns
        -------
        sa : :class:`~openturns.SobolIndicesAlgorithm`
        """
        if self._sa is None:
            raise Exception("The run method must launched first.")
        else:
            return self._sa

    def drawIndices(self, label):
        """
        Plot the aggregated Sobol indices.

        Parameters
        ----------
        label : sequence of float
            The name of the input parameters

        Returns
        -------
        fig : `matplotlib.figure <http://matplotlib.org/api/figure_api.html>`_
            Matplotlib figure object.
        ax : `matplotlib.axes <http://matplotlib.org/api/axes_api.html>`_
            Matplotlib axes object.
        """
        if len(label) != self._dim:
            raise AttributeError("The label dimension must be {}.").format(self._dim)

        graph = self._sa.draw()
        fig, ax = plt.subplots()
        View(graph, axes=[ax])
        ax.set_xticks(np.array(range(self._dim))+1)
        ax.set_xlim(0.5, self._dim+0.5)
        ax.set_xticklabels(label)
        fig.show()
        return fig, ax

    def getSensitivityMethod(self):
        """
        Accessor to the sensitivity method.
        
        Returns
        -------
        method : str
            The sensitivity method.
        """
        return self._method


    def setSensitivityMethod(self, method):
        """
        Accessor to the sensitivity method.
        
        Parameters
        ----------
        method : str
            The sensitivity method: either "Saltelli", "Martinez", "Jansen" or
            "MauntzKucherenko". Default is "Saltelli".
        """
        if method in ["Saltelli", "Jansen", "Martinez", "MauntzKucherenko"]:
            self._method = method
        else:
            raise AttributeError('The sensitivity method is not known, it ' + \
                'must be "Saltelli", "Martinez", "Jansen" or "MauntzKucherenko".')

    def setDefectSizes(self, size):
        """
        Accessor to the defect size where the POD is computed.

        Parameters
        ----------
        defectSize : sequence of float
            The defect sizes where the Monte Carlo simulation is performed to
            compute the POD.
        """
        size = np.hstack(np.array(size))
        size.sort()
        self._defectSizes = size.copy()
        minMin = self._krigingResult.getInputSample()[:, 0].getMin()[0]
        maxMax = self._krigingResult.getInputSample()[:, 0].getMax()[0]
        if size.max() > maxMax or size.min() < minMin:
            raise ValueError('Defect sizes must range between ' + \
                             '{:0.4f} '.format(np.ceil(minMin*10000)/10000) + \
                             'and {:0.4f}.'.format(np.floor(maxMax*10000)/10000))
        self._defectNumber = self._defectSizes.shape[0]

    def getDefectSizes(self):
        """
        Accessor to the defect size where the POD is computed.

        Returns
        -------
        defectSize : sequence of float
            The defect sizes where the Monte Carlo simulation is performed to
            compute the POD.
        """
        return self._defectSizes


class PODaggr(ot.OpenTURNSPythonFunction):
    """
    Aggregate function that compute the POD for a given points for all
    defect sizes given as parameter.

    Parameters
    ----------
    krigingResult : :class:`~openturns.KrigingResult`
        The kriging result object obtained after building the POD.
    dim : integer
        The number of input parameters of the function without the defect.
    defectSizes : sequence of float
        The defect size values for which the POD is computed.
    detection : float
        Detection value of the signal after box cox if it was enabled : must
        be "detectionBoxCox" from the POD object.
    """
    def __init__(self, krigingResult, dim, defectSizes, detection):
        super(PODaggr, self).__init__(dim, defectSizes.shape[0])
        self.krigingResult = krigingResult
        self.defectNumber = len(defectSizes)
        self.defectSizes = defectSizes
        self.detection = detection
    
    def _exec(self, X):
        # create sample with combining all defect size with the given X
        x = np.array(X, ndmin=2)
        x = x.repeat(self.defectNumber, axis=0)
        xWitha = np.concatenate((np.vstack(self.defectSizes), x), axis=1)

        # compute the kriging mean and variance
        mean = np.array(self.krigingResult.getConditionalMean(xWitha))

        # Two solutions to compute the variance, the second seems a bit faster
        #var = np.diag(self.krigingResult.getConditionalCovariance(xWitha))
        var = np.array([self.krigingResult.getConditionalCovariance(p)[0, 0] for p in xWitha])

        # check if the variance is positive of not, accept negative values
        # if they are > -1e-2, else raise an error. 
        if (var < 0).all():
            logging.warn("Warning : some variance values are negatives, " + \
                         "the kriging model may not be accurate enough.")

            if (var[var<0] < 1e-2).all():
                raise ValueError("Variance values are lower than 1e-2. Please " +\
                    "check the validity of the kriging model.")
            else:
                var = np.abs(var)

        # compute the quantile
        quantile = np.vstack((self.detection - mean) / np.sqrt(var))
        prob = 1. - np.array([ot.DistFunc.pNormal(q[0]) for q in quantile])
        return prob