# -*- Python -*-

import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
from ._pli import PLIMeanBase, PLIVarianceBase
import logging

__all__ = ['PLIMean', 'PLIVariance']


class PLIBase():
    """
    PLI base class.

    Notes
    -----
    PLI specific base class for the POD. Compute the indices for each defect
    size.
    """
    def __init__(self, POD, delta):

        className = type(POD).__name__
        if className == "PolynomialChaosPOD":
            self._podResult = POD.getPolynomialChaosResult()
            self._metamodel = self._podResult.getMetaModel()
            self._podType = "chaos"
        elif className in ["KrigingPOD", "AdaptiveSignalPOD"]:
            self._podResult = POD.getKrigingResult()
            self._metamodel = ot.ComposedFunction(self._podResult.getMetaModel(), POD._transformation)
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
        self._samplingSize = 10000

        # the distribution of the parameters without the one of the defects.
        tmpDistribution = POD.getDistribution()
        self._distribution = ot.ComposedDistribution([tmpDistribution.getMarginal(i)
                                                for i in range(1, self._dim+1)])

        self._delta = np.vstack(np.array(delta))

        self._gaussKronrod = ot.GaussKronrod(50, 1e-5,
                        ot.GaussKronrodRule(ot.GaussKronrodRule.G7K15))

        # initialize result matrix
        self._initializeResultMatrix()

    def _initializeResultMatrix(self):
        # initialize 3d matrix to save results for display
        marginals = np.arange(0, self._dim, 1)
        deltaMesh, defectMesh, marginalMesh = np.meshgrid(marginals, self._delta,
                                                          self._defectSizes)
        # matrix order [delta, marginal, defect]
        self._indices = np.zeros((self._delta.shape[0], self._dim, self._defectNumber))

        # initialize a list to store the pli object for each defect
        self._pli = [object] * self._defectNumber

    def _runMonteCarlo(self, defect):
        # set a parametric function where the first parameter = given defect
        g = ot.ParametricFunction(self._metamodel, [0], [defect])
        g = ot.MemoizeFunction(g)
        g.enableHistory()
        g.clearHistory()
        g.clearCache()
        output = ot.CompositeRandomVector(g, ot.RandomVector(self._distribution))
        event = ot.ThresholdEvent(output, ot.Greater(), self._detectionBoxCox)

        ##### Monte Carlo ########
        algo_MC = ot.ProbabilitySimulationAlgorithm(event)
        algo_MC.setMaximumOuterSampling(self._samplingSize)
        # set negative coef of variation to be sure the stopping criterion is the sampling size
        algo_MC.setMaximumCoefficientOfVariation(-1)
        algo_MC.run()
        return algo_MC.getResult()

    def run(self):
        """
        Compute the indices

        Notes
        -----
        Run the analysis:
            - run a Monte Carlo simulation
            - compute the indices for each defect size

        If, for a defect size, the probability estimate is less than 1e-3 or
        greater than 0.999, then the indices are not computed.
        """
        # initialize a list to store the indices for which the PLI are computed
        self._keepedDefect = np.arange(0, self._defectNumber, 1).tolist()

        for idefect in range(self._defectSizes.shape[0]):

            resultMonteCarlo = self._runMonteCarlo(self._defectSizes[idefect])
            pf = resultMonteCarlo.getProbabilityEstimate()

            if pf > 1e-3 and pf < 0.999:
                self._pli[idefect] = self._definePLIAlgorithm(resultMonteCarlo)
                self._pli[idefect].setGaussKronrod(self._gaussKronrod)
                self._pli[idefect].run()

                # store the result
                self._indices[:, :, idefect] = self._pli[idefect].getIndices()
            else:
                self._keepedDefect.remove(idefect)
                # set not computed indices to nan values
                self._indices[:, :, idefect] = np.zeros(self._indices[:, :, idefect].shape)*np.nan

        if len(self._keepedDefect) != self._defectNumber:
            logging.warning('The indices were estimated only for the following defect: '+ \
                  ''.join(('{:.2f}, ' * len(self._keepedDefect)).format(*self._defectSizes[self._keepedDefect])) + \
                  'because the probability estimate is too small or too big '+\
                  'for other defect values.') 

    def setDefectSizes(self, size):
        """
        Accessor to the defect size where the indices are computed.

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
        # update defect number
        self._defectNumber = self._defectSizes.shape[0]
        # update result matrix
        self._initializeResultMatrix()


    def getDefectSizes(self):
        """
        Accessor to the defect size where the indices are computed.

        Returns
        -------
        defectSize : sequence of float
            The defect sizes where the Monte Carlo simulation is performed to
            compute the POD.
        """
        return self._defectSizes

    def getSamplingSize(self):
        """
        Accessor to the Monte Carlo sampling size.

        Returns
        -------
        size : int
            The size of the Monte Carlo simulation used to compute the POD for
            each defect size.
        """
        return self._samplingSize

    def setSamplingSize(self, size):
        """
        Accessor to the Monte Carlo sampling size.

        Parameters
        ----------
        size : int
            The size of the Monte Carlo simulation used to compute the POD for
            each defect size.
        """
        self._samplingSize = size

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

    def getGaussKronrod(self):
        """
        Accessor to the Gauss Kronrod algorithm used to compute integrals
        """
        return self._gaussKronrod

    def setGaussKronrod(self, algo):
        """
        Accessor to the Gauss Kronrod algorithm used to compute integrals

        Parameters
        ----------
        algo : :py:class:`openturns.GaussKronrod`
            The algorithm
        """
        try:
            self._gaussKronrod = ot.GaussKronrod(algo)
        except NotImplementedError:
            raise AttributeError("The parameter must be a GaussKronrod algorithm.")

    def getPLIObject(self, idefect):
        """
        Accessor to the PLI object for a specific defect.

        Parameters
        ----------
        idefect : int
            The indice of the defect in the given delta list.

        Returns
        -------
        pli : :class:`PLI`
            The PLI base object from which more results can be obtained.
        """
        if idefect in self._keepedDefect:
            return self._pli[idefect]
        else:
            raise Exception("The indices have not been computed for this defect.")

    def getIndices(self, idelta=None, marginal=None, idefect=None):
        """
        Accessor to the indices

        Parameters
        ----------
        idelta : int
            The indice of the delta in the given delta list. Default is None = all.
        marginal : int
            The indice of the perturbed marginal. Default is None = all.
        idefect : int
            The indice of the defect in the given delta list. Default is None = all.

        Returns
        -------
        indices : float, 1d, 2d or 3d array.
            The parameter order of the full matrix is delta, marginal and defect.
            The returned array depends on the given parameter values. 

        """
        if idelta is None and marginal is None and idefect is None:
            return self._indices[:, :, :]
        elif idelta is None and marginal is None:
            return self._indices[:, :, idefect]
        elif idelta is None and idefect is None:
            return self._indices[:, marginal, :]
        elif idefect is None and marginal is None:
            return self._indices[idelta, :, :]
        elif idelta is None:
            return self._indices[:, marginal, idefect]
        elif marginal is None:
            return self._indices[idelta, :, idefect]
        elif idefect is None:
            return self._indices[idelta, marginal, :]
        else:
            return self._indices[idelta, marginal, idefect]

    def drawIndices(self, idefect, confidenceLevel=.95, label=None,
                    hellinger=True, name=None):
        """
        Draw the indices of all margins for a specific defect

        Parameters
        ----------
        idefect : int
            The indice of the defect in the given delta list.
        confidenceLevel : 0 < float < 1 or None
            The wanted confidence level to compute the interval. If set to 'None'
            only the indices are plotted.
        label : list of string
            The labels of each parameters.
        hellinger : bool
            If True, the indices are plotted with respect to the hellinger
            distance between the original PDF and the perturbed PDF.
            Default is True.

        Returns
        -------
        fig : `matplotlib.figure <http://matplotlib.org/api/figure_api.html>`_
            Matplotlib figure object.
        ax : `matplotlib.axes <http://matplotlib.org/api/axes_api.html>`_
            Matplotlib axes object.
        """
        if idefect in self._keepedDefect:
            fig, ax = self._pli[idefect].drawIndices(confidenceLevel=confidenceLevel,
                                                     label=label,
                                                     hellinger=hellinger,
                                                     name=None)
            ax.set_title(self.__class__.__name__ + \
                         ' - defect = {:.3f}'.format(self._defectSizes[idefect]))
            if name is not None:
                fig.savefig(name, bbox_inches='tight', transparent=True)
            return fig, ax
        else:
            raise Exception("The indices have not been computed for this defect.")

    def drawContourIndices(self, marginal, label=None, name=None):
        """
        Draw a contour plot of the indices for a specific marginal

        Parameters
        ----------
        marginal : int
            The indice of the perturbed marginal.
        label : list of string
            The labels of each parameters.

        Returns
        -------
        fig : `matplotlib.figure <http://matplotlib.org/api/figure_api.html>`_
            Matplotlib figure object.
        ax : `matplotlib.axes <http://matplotlib.org/api/axes_api.html>`_
            Matplotlib axes object.
        """ 
        if marginal > self._dim-1:
            raise AttributeError('The marginal parameter must ' +\
                                 'be in the range [0, {}]'.format(self._dim-1))

        if label is None:
            label = 'X{}'.format(marginal)

        extent = (np.min(self._defectSizes[self._keepedDefect]),
                  np.max(self._defectSizes[self._keepedDefect]),
                  np.min(self._delta), np.max(self._delta))

        Z = self._indices[:, marginal, self._keepedDefect]

        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(Z, cmap=plt.cm.RdBu, origin='lower', extent=extent, aspect='auto',
                       vmin=-np.nanmax(np.abs(Z)), vmax=np.nanmax(np.abs(Z))) # drawing the function
        # adding the Contour lines with labels
        cset = ax.contour(Z, 20, linewidths=0.5, colors='k', extent=extent)
        ax.clabel(cset,inline=True,fmt='%0.2f',fontsize=8, colors='k')
        fig.colorbar(im) # adding the colobar on the right
        ax.set_title(self.__class__.__name__ + ' indices - ' + label)
        ax.set_xlabel('Defects')
        ax.set_ylabel('Delta')
        if name is not None:
            fig.savefig(name, bbox_inches='tight', transparent=True)
        return fig, ax



class PLIMean(PLIBase):
    """
    PLI based on a mean perturbation.

    Parameters
    ----------
    POD : :class:`KrigingPOD`, :class:`AdaptiveSignalPOD` or :class:`PolynomialChaosPOD`
        The POD object where the run method has been performed.
    delta : 1d or 2d sequence of float
        The new values of the mean or sigma coefficient. Either 1d if delta
        values are the same for all marginals, or 2d if delta values are defined
        independently for each marginal.
    sigmaScaled : bool
        Change the type of the applied  mean shiftingfor all the variables. 
        If False (default case), the given delta values are the new marginal means.
        If True, newMean = mean + sigma x delta, where sigma
        is the standard deviation of each marginals.
    """
    def __init__(self, POD, delta, sigmaScaled=True):
        PLIBase.__init__(self, POD, delta)
        self._sigmaScaled = sigmaScaled


    def _definePLIAlgorithm(self, resultMonteCarlo):
        return PLIMeanBase(resultMonteCarlo, self._distribution, self._delta,
                           sigmaScaled=self._sigmaScaled)
        

class PLIVariance(PLIBase):
    """
    PLI based on a mean perturbation.

    Parameters
    ----------
    POD : :class:`KrigingPOD`, :class:`AdaptiveSignalPOD` or :class:`PolynomialChaosPOD`
        The POD object where the run method has been performed.
    delta : 1d or 2d sequence of float
        The new values of the mean. Either 1d if delta values are the same for
        all marginals, or 2d if delta values are defined independently for each
        marginal.
    covScaled : bool
        Change the type of the applied variance shifting for all the variables. 
        If False (default case), the given delta values are the new marginal variances.
        If True, newVariance = variance x delta.
    """
    def __init__(self, POD, delta, covScaled=True):
        PLIBase.__init__(self, POD, delta)
        self._covScaled = covScaled

    def _definePLIAlgorithm(self, resultMonteCarlo):
        return PLIVarianceBase(resultMonteCarlo, self._distribution, self._delta,
                               covScaled=self._covScaled)


        
