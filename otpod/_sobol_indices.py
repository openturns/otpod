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

    SobolIndices(*POD, N*)

    Parameters
    ----------
    POD : :class:`KrigingPOD`, :class:`AdaptiveSignalPOD` or :class:`PolynomialChaosPOD`
        The POD object where the run method has been performed.
    N : int
        Size of samples to generate

    returns
    -------
    sa : :py:class:`openturns.SobolIndicesAlgorithm`
        The openturns object that perform the sensitivity algorithm.

    Notes
    -----
    This class uses the :class:`openturns.SobolIndicesAlgorithm` class
    of OpenTURNS. The sensitivity analysis can be performed only with a POD
    built with a Kriging metamodel or a polynomial chaos where the input
    dimension is greater than 3 (counting the defect).

    When using Kriging, the POD at a given point is computed using the kriging 
    mean and variance. For polynomial chaos, random coefficients are generated, 
    the signal is computed for all coefficients and the POD is eventually
    estimated. The default simulation size is set to 1000. This value can be
    changed using :func:`setSimulationSize`.

    The sensitivity analysis allows to computed aggregated Sobol indices for
    the given range of defect sizes. The default defect sizes correspond with
    those defined in the *POD* object. It can be changed using
    :func:`setDefectSizes`.

    The four methods developed in OpenTURNS are availables and can be chosen
    thanks to :func:`setSensitivityMethod`. The default
    method is "Saltelli".

    The result of the sensitivity analysis is available using
    :func:`getSensitivityResult`. It returns the openturns sensitivity object
    from which the sensitivity values are given using proper methods.
    """

    def __init__(self, POD, N):

        className = type(POD).__name__
        if className == "PolynomialChaosPOD":
            self._podResult = POD.getPolynomialChaosResult()
            self._podType = "chaos"
        elif className in ["KrigingPOD", "AdaptiveSignalPOD"]:
            self._podResult = POD.getKrigingResult()
            self._podType = "kriging"
        else:
            raise Exception("Sobol indices can only be computed based on a " + \
                            "POD built with Kriging or polynomial chaos.")

        # dimension is minus 1 to remove the defect parameter
        self._dim = self._podResult.getMetaModel().getInputDimension() - 1
        assert (self._dim >=2), "The number of parameters must be greater or " + \
                "equal than 2 to be able to perform the sensitivity analysis."
        self._POD = POD
        self._defectSizes = POD.getDefectSizes()
        self._defectNumber = self._defectSizes.shape[0]
        self._detectionBoxCox = POD._detectionBoxCox
        self._simulationSize = 1000

        # the distribution of the parameters without the one of the defects.
        tmpDistribution = POD.getDistribution()
        self._distribution = ot.ComposedDistribution([tmpDistribution.getMarginal(i) for i in range(1, self._dim+1)])

        # number of samples
        self._N = N

        # initialize method parameter and sa attribute
        self._method = "Saltelli"
        self._sa = None

    def run(self):
        """
        Compute the Sobol indices with the chosen algorithm. 
        """

        # create the Function which computes the POD for a given
        # realization and for all defect sizes.
        if self._podType == "kriging":
            self._PODaggr = ot.Function(PODaggrKriging(self._POD,
                            self._dim, self._defectSizes, self._detectionBoxCox))
        elif self._podType == "chaos":
            self._PODaggr = ot.Function(PODaggrChaos(self._POD,
                            self._dim, self._defectSizes, self._detectionBoxCox,
                            self._simulationSize))

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
        sa : :py:class:`openturns.SobolIndicesAlgorithm`
        """
        if self._sa is None:
            raise Exception("The run method must launched first.")
        else:
            return self._sa

    def drawAggregatedIndices(self, label=None, name=None):
        """
        Plot the aggregated Sobol indices.

        Parameters
        ----------
        label : sequence of float
            The name of the input parameters
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
        if label is None:
            label = ot.Description.BuildDefault(self._dim, "X")
        else:
            if len(label) != self._dim:
                raise AttributeError("The label dimension must be {}.").format(self._dim)

        graph = self._sa.draw()
        fig, ax = plt.subplots(figsize=(8, 6))
        View(graph, axes=[ax])
        ax.set_xticks(np.array(range(self._dim))+1)
        ax.set_xlim(0.5, self._dim+0.5)
        ax.set_xticklabels(label)
        ax.set_title(graph.getTitle())
        if name is not None:
            fig.savefig(name, bbox_inches='tight', transparent=True)

        return fig, ax

    def drawFirstOrderIndices(self, label=None, name=None):
        """
        Plot the first Sobol indices for all defect values.

        Parameters
        ----------
        label : sequence of float
            The name of the input parameters
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
        fig, ax = self._drawIndices('first', label, name)
        return fig, ax

    def drawTotalOrderIndices(self, label=None, name=None):
        """
        Plot the total Sobol indices for all defect values.

        Parameters
        ----------
        label : sequence of float
            The name of the input parameters
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
        fig, ax = self._drawIndices('total', label, name)
        return fig, ax

    def _drawIndices(self, order, label, name=None):
        """
        Based method to plot the Sobol indices.

        Parameters
        ----------
        order : string
            Either first or total
        label : sequence of float
            The name of the input parameters
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
        if label is None:
            label = ot.Description.BuildDefault(self._dim, "X")
        else:
            if len(label) != self._dim:
                raise AttributeError("The label dimension must be {}.").format(self._dim)

        # get the indices values that can be computed
        xplot = ot.Sample(0, 1)
        yplot = ot.Sample(0, self._dim)
        for i, defect in enumerate(self._defectSizes):
            try:
                if order == 'first':
                    yplot.add(self._sa.getFirstOrderIndices(i))
                elif order == 'total':
                    yplot.add(self._sa.getTotalOrderIndices(i))
                xplot.add([defect])
            except:
                pass

        # define different colors for each parameter
        colors = ot.Drawable.BuildDefaultPalette(self._dim)

        fig, ax = plt.subplots(figsize=(8, 2*self._dim))
        for output in range(self._dim):
            ax.plot(xplot, yplot[:, output], color=colors[output], marker='o',
                                             ls='', label=label[output])
        ax.set_xlabel('Defects')
        ax.set_ylabel('Sensitivity indices')
        if order == 'first':
            ax.set_title('First order sensitivity indices - {} Algorithm'.format(self._method))
        elif order == 'total':
            ax.set_title('Total order sensitivity indices - {} Algorithm'.format(self._method))
        ax.set_ylim(0, 1.2)
        ax.grid()
        ax.legend(loc='lower left')
        if name is not None:
            fig.savefig(name, bbox_inches='tight', transparent=True)
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
        minMin = self._POD._input[:, 0].getMin()[0]
        maxMax = self._POD._input[:, 0].getMax()[0]
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

    def setDistribution(self, distribution):
        """
        Accessor to the parameters distribution. 

        Parameters
        ----------
        distribution : :py:class:`openturns.ComposedDistribution`
            The input parameters distribution used for the Monte Carlo simulation.
        """
        try:
            ot.ComposedDistribution(distribution)
        except NotImplementedError:
            raise Exception('The given parameter is not a ComposedDistribution.')

        if distribution.getDimension() != self._dim:
            raise AttributeError("The dimension of the distribution must be {}.".format(self._dim))
        self._distribution = distribution

    def getDistribution(self):
        """
        Accessor to the parameters distribution. 

        Returns
        -------
        distribution : :py:class:`openturns.ComposedDistribution`
            The input parameters distribution used for the Monte Carlo simulation.
            Default is a Uniform distribution for all parameters.
        """
        return self._distribution

    def getSimulationSize(self):
        """
        Accessor to the simulation size when using polynomial chaos.

        Returns
        ----------
        size : int
            The size of the simulation used to compute POD at a given point.
        """
        return self._simulationSize

    def setSimulationSize(self, size):
        """
        Accessor to the simulation size when using polynomial chaos.

        Parameters
        ----------
        size : int
            The size of the simulation used to compute at a given point. Default
            is 1000.
        """
        self._simulationSize = size


class PODaggrKriging(ot.OpenTURNSPythonFunction):
    """
    Aggregate function that compute the POD for a given points for all
    defect sizes given as parameter.

    Parameters
    ----------
    krigingPOD : :class:`KrigingPOD` or :class:`AdaptiveSignalPOD` 
        The kriging POD object obtained after building the POD.
    dim : integer
        The number of input parameters of the function without the defect.
    defectSizes : sequence of float
        The defect size values for which the POD is computed.
    detection : float
        Detection value of the signal after box cox if it was enabled : must
        be "detectionBoxCox" from the POD object.
    """
    def __init__(self, krigingPOD, dim, defectSizes, detection):

        super(PODaggrKriging, self).__init__(dim, defectSizes.shape[0])
        self.krigingResult = krigingPOD.getKrigingResult()
        self.defectNumber = len(defectSizes)
        self.defectSizes = defectSizes
        self.detection = detection
    
    def _exec(self, X):
        # create sample combining all defect size with the given X
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
            logging.warning("Warning : some variance values are negatives, " + \
                         "the kriging model may not be accurate enough.")

            if (var[var<0] < 1e-2).all():
                raise ValueError("Variance values are lower than -1e-2. Please " +\
                    "check the validity of the kriging model.")
            else:
                var = np.abs(var)

        # compute the quantile
        quantile = np.vstack((self.detection - mean) / np.sqrt(var))
        prob = 1. - np.array([ot.DistFunc.pNormal(q[0]) for q in quantile])
        return prob


class PODaggrChaos(ot.OpenTURNSPythonFunction):
    """
    Aggregate function based on the polynomial chaos that compute the POD for a
    given points for all defect sizes given as parameter.

    Parameters
    ----------
    chaosPOD : :class:`PolynomialChaosPOD`
        The chaos POD object.
    dim : int
        The number of input parameters of the function without the defect.
    defectSizes : sequence of float
        The defect size values for which the POD is computed.
    detection : float
        Detection value of the signal after box cox if it was enabled : must
        be "detectionBoxCox" from the POD object.
    simulationSize : int
        The size of the simulation used to compute the POD at a given point.
    """
    def __init__(self, chaosPOD, dim, defectSizes, detection, simulationSize):
        super(PODaggrChaos, self).__init__(dim, defectSizes.shape[0])
        self.chaosPOD = chaosPOD
        self.dim = dim
        self.defectSizes = defectSizes
        self.defectNumber = len(defectSizes)
        self.simulationSize = simulationSize
        self.detection = detection

        # get the sample of coefficient using the coef distribution 
        # used to compute the POD for a given point
        sampleCoefs = chaosPOD.getCoefficientDistribution().getSample(simulationSize)

        # get some result from the polynomial chaos to build a vectoriel
        # chaos function that return the signal values for all chaos with
        # different coefficients for one specific point
        chaosResult = chaosPOD.getPolynomialChaosResult()
        reducedBasis = chaosResult.getReducedBasis()
        transformation = chaosResult.getTransformation()
        chaosFunctionCol = []
        for i, coefs in enumerate(sampleCoefs):
            standardChaosFunction = ot.LinearCombinationFunction(reducedBasis, coefs)
            chaosFunctionCol.append(ot.ComposedFunction(standardChaosFunction, transformation))
        self.chaosFunction = ot.AggregatedFunction(chaosFunctionCol)

    def _exec(self, X):
        # create sample combining all defect size with the given X
        x = np.array(X, ndmin=2)
        x = x.repeat(self.defectNumber, axis=0)
        xWitha = np.concatenate((np.vstack(self.chaosPOD._defectSizes), x), axis=1)
        # add randomness from the residual, identical for all defect size
        residualsSample = np.hstack(self.chaosPOD._normalDist.getSample(self.simulationSize) * self.chaosPOD._stderr)
        # compute the signal for all chaos
        Y = self.chaosFunction(xWitha)
        # compute the POD for all defect size
        return np.mean((np.array(Y) + residualsSample) > self.detection, axis=1)

    # vectorial way to compute the POD
    def _exec_sample(self, X):
        samplingSize = ot.Sample(X).getSize()

        # create sample containing all input combined with all defect sizes
        fullX = ot.Sample(samplingSize * self.defectNumber,self.dim+1)
        for i, x in enumerate(X):
            x = np.array(x, ndmin=2)
            x = x.repeat(self.defectNumber, axis=0)
            xWitha = np.concatenate((np.vstack(self.defectSizes), x), axis=1)
            fullX[self.defectNumber*i:self.defectNumber*(i+1), :] = xWitha

        # add randomness from the residual, identical for all defect size
        residualsSample = ot.Normal(samplingSize).getSample(self.simulationSize) * self.chaosPOD._stderr
        fullRes = ot.Sample(self.simulationSize, samplingSize * self.defectNumber)
        for i in range(samplingSize):
            fullRes[:, self.defectNumber*i:self.defectNumber*(i+1)] = np.repeat(residualsSample[:, i], self.defectNumber, axis=1)
        fullRes = np.transpose(fullRes)

        # compute the signal
        Y = np.array(self.chaosFunction(fullX))

        # compute the POD
        prob = np.mean((Y + fullRes) > self.detection, axis=1)
        prob = prob.reshape(samplingSize, self.defectNumber)
        return prob

            
