# -*- Python -*-

__all__ = ["PLI"]

import numpy as np
import openturns as ot
from openturns.viewer import View
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


class PLI:
    """
    PLI base class.

    Notes
    -----
    The Perturbation Law Indices are based upon the modification of the
    probability density function (pdf) of the random inputs, when the
    quantity of interest is a failure probability. An input is considered
    influential if the input pdf modification leads to a broad change in the
    failure probability. These sensitivity indices can be computed using the
    sole set of simulations that has already been used to estimate the
    failure probability, thus limiting the number of calls to the numerical
    model. In this implementation, the sample must come from a Monte Carlo
    simulation.

    The input perturbation is defined to obtain the perturbed density
    function as the closest to the original one, in the sense of the
    Kullback-Leibler divergence. The implemented perturbation includes a
    mean shift and a variance shift, accessible through the derived class.
    The current implementation only allows to modifiy Normal and Uniform
    density functions.

    In order to compare equivalently the indices when the input distributions
    are not the same, it is possible to plot the indices with respect to the
    Hellinger distance.

    These indices have been developed by Paul Lemaitre:
      - Paul Lemaître, Ekatarina Sergienko, Aurélie Arnaud, Nicolas Bousquet,
        Fabrice Gamboa, et al.. Density modification based reliability
        sensitivity analysis. 2012.
      - Paul Lemaitre. Analyse de sensibilité en fiabilité des structures.
        Mécanique des structures [physics.class-ph]. Université de Bordeaux,
        2014. Français.

    See also
    --------
    PLIMean, PLIVariance
    """

    def __init__(self, monteCarloResult, distribution, deltas):
        # the monte carlo result must have its underlying function with
        # the history enabled because the failure sample is obtained using it
        self._monteCarloResult = monteCarloResult
        self.function = ot.MemoizeFunction(
            self._monteCarloResult.getEvent().getFunction()
        )
        if self.function.getOutputHistory().getSize() == 0:
            raise AttributeError(
                "The performance function of the Monte Carlo "
                + "simulation result should be a MemoizeFunction."
            )

        # the original distribution
        if distribution.hasIndependentCopula():
            self._distribution = distribution
        else:
            raise Exception("The distribution must have an independent copula.")
        self._dim = self._distribution.getDimension()

        # the 1d or 2d sequence of deltas
        self._originalDelta = np.vstack(np.array(deltas))
        self._deltaValues = self._originalDelta.copy()
        self._deltaSize = self._deltaValues.shape[0]

        if self._deltaValues.shape[1] != 1 and self._deltaValues.shape[1] != self._dim:
            raise AttributeError(
                "The deltas parameter must be 1d sequence of "
                + "float or 2d sequence of float of dimension "
                + "equal to {}.".format(self._dim)
            )

        # check if the delta values have only one dimension -> copy the columns
        if self._deltaValues.shape[1] == 1:
            self._deltaValues = (
                np.ones((self._deltaValues.shape[0], self._dim)) * self._deltaValues
            )

        # initialize array result
        # rows : delta
        # columns : maginal
        self._pfdelta = np.zeros((self._deltaSize, self._dim))
        self._varPfdelta = np.zeros((self._deltaSize, self._dim))
        self._indices = np.zeros((self._deltaSize, self._dim))
        # for loop to avoid copy id of the distribution collection
        self._estimatorDist = [
            ot.DistributionCollection(self._dim) for i in range(self._deltaSize)
        ]

        # set the gaus Kronrod algorithm
        self._gaussKronrod = ot.GaussKronrod(
            50, 1e-5, ot.GaussKronrodRule(ot.GaussKronrodRule.G7K15)
        )

    def run(self):
        """
        Run the analysis:
        - get the failure sample
        - evaluate the probabilities with the perturbed distributions
        - define the estimator distributions
        """
        # get some results from the monte carlo simulation result
        self._pf = self._monteCarloResult.getProbabilityEstimate()
        self._simulationSize = (
            self._monteCarloResult.getOuterSampling()
            * self._monteCarloResult.getBlockSize()
        )
        # get the failure sample from the monte carlo simulation
        self._failureSample = self._getFailureSample()

        for marginal in range(self._dim):
            for idelta in range(self._deltaSize):
                # compute the probability of failure delta with the perturbed PDF
                self._pfdelta[idelta, marginal] = self._computePfdelta(marginal, idelta)
                self._varPfdelta[idelta, marginal] = self._computeVariancePfdelta(
                    marginal, idelta
                )

                # compute the indices and indices estimator distribution
                self._indices[idelta, marginal] = self._computePLIndices(
                    marginal, idelta
                )
                self._estimatorDist[idelta][
                    marginal
                ] = self._computeEstimatorDistribution(marginal, idelta)

    def _getFailureSample(self):
        """
        Compute the failure sample, using the sample from the OpenTURNS
        Function history, the operator and threshold of the event
        """
        # get the input and output sample
        inputSample = self.function.getInputHistory()
        outputSample = self.function.getOutputHistory()
        operator = (
            self._monteCarloResult.getEvent()
            .getOperator()
            .getImplementation()
            .getClassName()
        )
        threshold = self._monteCarloResult.getEvent().getThreshold()
        if operator in ["Less", "LessOrEqual"]:
            return np.array(inputSample)[
                (np.hstack(np.array(outputSample)) < threshold), :
            ]
        elif operator in ["Greater", "GreaterOrEqual"]:
            return np.array(inputSample)[
                (np.hstack(np.array(outputSample)) > threshold), :
            ]
        else:
            raise NameError(
                'The comparison operator "{}" is not known.'.format(operator)
            )

    def _computePfdelta(self, marginal, idelta):
        """
        Compute the probability of failure for a perturbed distribution.

        Parameters
        ----------
        marginal : int
            The indice of the perturbed marginal.
        idelta : int
            The indice of the delta to be applied in the given delta list.
        """
        return (
            1.0
            / self._simulationSize
            * np.sum(
                self._computePerturbedPDF(
                    self._failureSample, marginal, self._deltaValues[idelta, marginal]
                )
                / self._distribution.computePDF(self._failureSample)
            )
        )

    def _computeVariancePfdelta(self, marginal, idelta):
        """
        Compute the variance of pf_delta (times sqrt(N)). See Lemaitre paper Lemma 3.1.

        Parameters
        ----------
        marginal : int
            The indice of the perturbed marginal.
        idelta : int
            The indice of the delta to be applied in the given delta list.
        """
        return (
            1.0
            / self._simulationSize
            * np.sum(
                (
                    self._computePerturbedPDF(
                        self._failureSample,
                        marginal,
                        self._deltaValues[idelta, marginal],
                    )
                    / self._distribution.computePDF(self._failureSample)
                )
                ** 2
            )
            - self._pfdelta[idelta, marginal] ** 2
        )

    def _computePLIndices(self, marginal, idelta):
        """
        Compute the PL indices.

        Parameters
        ----------
        marginal : int
            The indice of the perturbed marginal.
        idelta : int
            The indice of the delta to be applied in the given delta list.
        """
        if self._pfdelta[idelta, marginal] < self._pf:
            return (self._pfdelta[idelta, marginal] - self._pf) / self._pfdelta[
                idelta, marginal
            ]
        else:
            return (self._pfdelta[idelta, marginal] - self._pf) / self._pf

    def _computeEstimatorDistribution(self, marginal, idelta):
        """
        Define the asymptotic distribution of the indices.

        Parameters
        ----------
        marginal : int
            The indice of the perturbed marginal.
        idelta : int
            The indice of the delta to be applied in the given delta list.

        Returns
        -------
        dist ::py :class:`openturns.Distribution`
            The asymptotic distribution
        """

        covMat = np.array(
            [
                [
                    self._pf * (1 - self._pf),
                    self._pfdelta[idelta, marginal] * (1 - self._pf),
                ],
                [
                    self._pfdelta[idelta, marginal] * (1 - self._pf),
                    self._varPfdelta[idelta, marginal],
                ],
            ]
        )
        if self._pfdelta[idelta, marginal] < self._pf:
            d = np.vstack(
                [
                    -1.0 / self._pfdelta[idelta, marginal],
                    self._pf / self._pfdelta[idelta, marginal] ** 2,
                ]
            )
        else:
            d = np.vstack(
                [-self._pfdelta[idelta, marginal] / self._pf**2, 1.0 / self._pf]
            )

        var = np.dot(d.transpose(), np.dot(covMat, d))
        if np.abs(var) < 1e-6 and var < 0 or var == 0:
            sigma = 1e-30
        else:
            sigma = np.sqrt(var / self._simulationSize)[0, 0]

        return ot.Normal(self._indices[idelta, marginal], sigma)

    def getIndices(self):
        """
        Accessor to the Pertubation Law Indices.

        Returns
        -------
        pli : 2d sequence of float
            The indices for all marginals and all given delta values.
        """
        return self._indices

    def getDeltaSample(self):
        """
        Accessor to applied delta values.

        Returns
        -------
        deltaSample : 2d sequence of float
            The delta values.
        """
        return self._deltaValues

    def computeConfidenceInterval(self, confidenceLevel=0.95):
        """
        Accessor to the confidence interval of the indices.

        Parameters
        ----------
        confidenceLevel : 0 < float < 1
            The wanted confidence level to compute the interval.

        Returns
        -------
        ci : list of 2d sequence of float
            A list of arrays for each marginal containing the lower and upper
            bound of the confidence interval for each delta values.
        """
        # use loop to avoid copy id of numpy array
        ci = [np.zeros((self._deltaSize, 2)) for i in range(self._dim)]
        for marginal in range(self._dim):
            for idelta in range(self._deltaSize):
                ci_marginal = self._estimatorDist[idelta][
                    marginal
                ].computeBilateralConfidenceInterval(confidenceLevel)
                ci[marginal][idelta, 0] = ci_marginal.getLowerBound()[0]
                ci[marginal][idelta, 1] = ci_marginal.getUpperBound()[0]
        return ci

    def getPerturbedProbabilityEstimate(self):
        """
        Accessor to the perturbed probability of failure

        Returns
        -------
        pfdelta : float
            The probability of failure computed with the perturbed density function.
        """
        return self._pfdelta

    def drawIndices(self, confidenceLevel=0.95, label=None, hellinger=False, name=None):
        """
        Draw all indices

        Parameters
        ----------
        confidenceLevel : 0 < float < 1 or None
            The wanted confidence level to compute the interval. If set to 'None'
            only the indices are plotted.
        label : list of string
            The labels of each parameters.
        hellinger : bool
            If True, the indices are plotted with respect to the hellinger
            distance between the original PDF and the perturbed PDF.

        Returns
        -------
        fig : `matplotlib.figure <http://matplotlib.org/api/figure_api.html>`_
            Matplotlib figure object.
        ax : `matplotlib.axes <http://matplotlib.org/api/axes_api.html>`_
            Matplotlib axes object.
        """

        # colors = ot.Drawable.BuildDefaultPalette(self._dim)
        if label is None:
            label = ot.Description.BuildDefault(self._dim, "X")
        else:
            if len(label) != self._dim:
                raise AttributeError("The label dimension must be {}.").format(
                    self._dim
                )

        # compute the confidence interval
        if confidenceLevel is not None:
            ci = self.computeConfidenceInterval(confidenceLevel)

        fig, ax = plt.subplots(figsize=(8, 6))
        for marginal in range(self._dim):
            if confidenceLevel is not None:
                currentLabel = label[marginal] + " + {:.0f}% CI".format(
                    confidenceLevel * 100
                )
            else:
                currentLabel = label[marginal]

            # define the abcsissa of the plot, either the delta values or the
            # hellinger distances if activated
            x_support = np.hstack(self._originalDelta.copy())
            xLabel = "Delta"
            if hellinger:
                for i, d in enumerate(self._deltaValues[:, marginal]):
                    x_support[i] = self._computeHellinger(marginal, d)

                    # if the delta is actually lower than original, then the
                    # abcsissa value is set to the opposite
                    if d < self.getOriginalDelta(marginal):
                        x_support[i] = -x_support[i]
                xLabel = "Hellinger distance"

            (plot,) = ax.plot(
                x_support,
                self._indices[:, marginal],
                marker=".",
                label=currentLabel,
                markersize=9,
            )
            if confidenceLevel is not None:
                ax.fill_between(
                    x_support,
                    ci[marginal][:, 0],
                    ci[marginal][:, 1],
                    facecolor=plot.get_color(),
                    alpha=0.3,
                )
        ax.grid()
        ax.set_xlabel(xLabel)
        ax.set_ylabel("Sensitivity indices")
        ax.set_title(self.__class__.__name__)
        ax.legend(loc="upper center")

        # if hellinger distance, change the negative ticks label to be positive
        if hellinger:
            ax.set_xticklabels([str(t) for t in np.abs(ax.get_xticks())])

        if name is not None:
            fig.savefig(name, bbox_inches="tight", transparent=True)
        return fig, ax

    def drawMarginal1DPDF(
        self,
        marginal,
        idelta,
        showOriginal=True,
        label=None,
        xMin=None,
        xMax=None,
        pointNumber=None,
        name=None,
    ):
        """
        Draw the probability density function of a margin.

        Parameters
        ----------
        marginal : int
            The index of the margin of interest.
        idelta : int
            The index in the delta array.
        showOriginal : bool
            Display on the same figure the original pdf or not.
        x_min : float
            The starting value that is used for meshing the x-axis.
            Defaults uses the quantile associated to the probability level 0.05.
        x_max : float, :math:`x_{max} > x_{min}`
            The ending value that is used for meshing the x-axis.
            Defaults uses the quantile associated to the probability level 0.95.
        n_points : int
            The number of points that is used for meshing the x-axis.
            Defaults uses `DistributionImplementation-DefaultPointNumber` from the
            :py:class:`openturns.ResourceMap`.

        Returns
        -------
        fig : `matplotlib.figure <http://matplotlib.org/api/figure_api.html>`_
            Matplotlib figure object.
        ax : `matplotlib.axes <http://matplotlib.org/api/axes_api.html>`_
            Matplotlib axes object.
        """
        if xMin is None:
            qMin = 0.05
            xMin = self._distribution.getMarginal(marginal).computeQuantile(qMin)[0]
        if xMax is None:
            qMax = 0.95
            xMax = self._distribution.getMarginal(marginal).computeQuantile(qMax)[0]
        if pointNumber is None:
            pointNumber = ot.ResourceMap.GetAsUnsignedInteger(
                "Distribution-DefaultPointNumber"
            )
        if label is None:
            label = "X{}".format(marginal)

        fig, ax = plt.subplots()
        xSample = np.vstack(np.linspace(xMin, xMax, pointNumber))
        pdfSample = self._perturbedMarginalPDF(
            xSample, marginal, self._deltaValues[idelta, marginal]
        )
        plt.plot(
            xSample,
            pdfSample,
            "b-",
            label=label
            + " - Perturbed PDF - "
            + "delta = {:.2e}".format(self._deltaValues[idelta, marginal]),
        )

        if showOriginal:
            View(
                self._distribution.drawMarginal1DPDF(
                    marginal, xMin, xMax, int(pointNumber)
                ),
                plot_kw={"label": label + " - Original PDF"},
                axes=[ax],
            )
            ax.grid()

        ax.grid()
        ax.set_ylabel("PDF")
        ax.set_xlabel(label)
        ax.set_title(label + " - PDF")
        if name is not None:
            fig.savefig(name, bbox_inches="tight", transparent=True)
        return fig, ax

    def _computePerturbedPDF(self, X, marginal, delta):
        """
        Compute the perturbed joint PDF.

        Parameters
        ----------
        X : 2d sequence of float
            The sample where the PDF is computed.
        marginal : int
            The indice of the perturbed marginal.
        delta : float
            The value of the new moment.
        """
        X = np.atleast_2d(X)
        pdf_list = [
            self._distribution.getMarginal(i).computePDF for i in range(self._dim)
        ]
        pdf_list[marginal] = lambda X: self._perturbedMarginalPDF(X, marginal, delta)
        productPDF = 1.0
        for i in range(X.shape[1]):
            productPDF *= np.array(pdf_list[i](np.vstack(X[:, i])))
        return productPDF

    def _perturbedMarginalPDF(self, X):
        raise Exception("Implemented in child class.")

    def getOriginalDelta(self, marginal):
        """
        Accessor to the original delta value

        Parameters
        ----------
        marginal : int
            The indice of the perturbed marginal.
        """
        raise Exception("Implemented in child class.")

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

    def _computeHellinger(self, marginal, delta):
        """
        Compute the Hellinger distance between the original marginal PDF and the
        perturbed marginal PDF.

        Parameters
        ----------
        marginal : int
            The indice of the perturbed marginal.
        delta : float
            The value of the new moment.
        """

        def fun(x, marginal, delta):
            x = np.atleast_2d(x)
            res = np.sqrt(
                np.array(self._distribution.getMarginal(marginal).computePDF(x))
                * np.array(self._perturbedMarginalPDF(x, marginal, delta))
            )
            return res[0]

        func = ot.PythonFunction(1, 1, lambda X: fun(X, marginal, delta))
        h = (
            2
            - 2
            * self._gaussKronrod.integrate(
                func, self._distribution.getMarginal(marginal).getRange()
            )[0]
        )
        return h


class PLIMeanBase(PLI):
    """
    PLI based on a mean perturbation.

    Parameters
    ----------
    monteCarloResult : :py:class:`openturns.SimulationResult`
        The OpenTURNS result object from a Monte Carlo simulation. The function
        used in the simulation must have its history enabled in order to be able
        to get the sample.
    distribution ::py :class:`openturns.Distribution`
        The joint distribution of the input parameters.
    delta : 1d or 2d sequence of float
        The new values of the mean or sigma coefficient. Either 1d if delta
        values are the same for all marginals, or 2d if delta values are defined
        independently for each marginal.
    sigmaScaled : bool
        Change the type of the applied mean shifting for all the variables.
        If False (default case), the given delta values are the new marginal means.
        If True, newMean = mean + sigma x delta, where sigma
        is the standard deviation of each marginals.
    """

    def __init__(self, monteCarloResult, distribution, delta, sigmaScaled=False):
        PLI.__init__(self, monteCarloResult, distribution, delta)

        # if activated,
        if sigmaScaled:
            std = self._distribution.getStandardDeviation()
            mean = self._distribution.getMean()
            self._deltaValues = self._deltaValues * std + mean

    def _perturbedMarginalPDF(self, X, marginal, delta):
        """
        Compute the perturbed marginal PDF.

        Parameters
        ----------
        X : 2d sequence of float
            The sample where the PDF is computed.
        marginal : int
            The indice of the perturbed marginal.
        delta : float
            The value of the new mean.
        """
        X = np.atleast_2d(X)

        marginalDist = self._distribution.getMarginal(marginal)

        # check the case if the delta match the original mean
        if delta == marginalDist.getMean()[0]:
            return np.array(marginalDist.computePDF(X))
        else:
            if marginalDist.getImplementation().getClassName() == "Normal":
                std = marginalDist.getStandardDeviation()[0]
                return (
                    1.0
                    / (std * np.sqrt(2 * np.pi))
                    * np.exp(-0.5 * ((X - delta) / std) ** 2)
                )

            elif marginalDist.getImplementation().getClassName() == "Uniform":
                # define the function Mx'/Mx - delta, must be = 0 to find optimal lambda
                def MprimeOverM(lamb, a, b, delta):
                    a = float(a)
                    if lamb == 0:
                        return (a + b) / 2 - delta
                    else:
                        return (
                            np.exp(lamb * b) * (lamb * b - 1)
                            + np.exp(lamb * a) * (1 - lamb * a)
                        ) / (lamb * (np.exp(lamb * b) - np.exp(lamb * a))) - delta

                a, b = marginalDist.getParameter()
                # compute the optimal lambda solving the M'/M - delta = 0
                optimalLambda = fsolve(
                    MprimeOverM,
                    0,
                    args=(a, b, delta),
                    epsfcn=marginalDist.getStandardDeviation()[0] * 0.01,
                )

                # return the analytical expression of the distribution
                return (
                    optimalLambda
                    / (np.exp(optimalLambda * b) - np.exp(optimalLambda * a))
                    * np.exp(optimalLambda * X)
                    * np.logical_and(X <= b, X >= a)
                )
            else:
                raise NotImplementedError(
                    "Only Normal and Uniform distribution can be used."
                )

    def getOriginalDelta(self, marginal):
        """
        Accessor to the original mean value

        Parameters
        ----------
        marginal : int
            The indice of the perturbed marginal.
        """
        return self._distribution.getMarginal(marginal).getMean()[0]


class PLIVarianceBase(PLI):
    """
    PLI based on a variance perturbation.

    Parameters
    ----------
    monteCarloResult ::py :class:`openturns.SimulationResult`
        The OpenTURNS result object from a Monte Carlo simulation. The function
        used in the simulation must have its history enabled in order to be able
        to get the sample.
    distribution ::py :class:`openturns.Distribution`
        The joint distribution of the input parameters.
    delta : 1d or 2d sequence of float
        The new values of the variance of coefficient. Either 1d if delta values
        are the same for all marginals, or 2d if delta values are defined
        independently for each marginal.
    covScaled : bool
        Change the type of the applied variance shifting for all the variables.
        If False (default case), the given delta values are the new marginal variances.
        If True, newVariance = (mean x delta + std)^2, it corresponds with an
        increase of the coefficient of variation by delta : newCov = cov + delta.
    """

    def __init__(self, monteCarloResult, distribution, delta, covScaled=False):
        PLI.__init__(self, monteCarloResult, distribution, delta)

        # check if the delta values are positive
        if ~(self._deltaValues > 0).all() and not covScaled:
            raise AttributeError("The delta values must be positive.")

        # if activated,
        if covScaled:
            std = np.array(self._distribution.getStandardDeviation())
            mean = np.array(self._distribution.getMean())
            self._deltaValues = (self._deltaValues * mean + std) ** 2

    def _perturbedMarginalPDF(self, X, marginal, delta):
        """
        Compute the perturbed marginal PDF.

        Parameters
        ----------
        X : 2d sequence of float
            The sample where the PDF is computed.
        marginal : int
            The indice of the perturbed marginal.
        delta : float
            The value of the new variance.
        """
        X = np.atleast_2d(X)

        marginalDist = self._distribution.getMarginal(marginal)

        # check the case if the delta match the original variance
        if delta == marginalDist.getCovariance()[0, 0]:
            return np.array(marginalDist.computePDF(X))
        else:
            if marginalDist.getImplementation().getClassName() == "Normal":
                mean = marginalDist.getMean()[0]
                std = np.sqrt(delta)
                return (
                    1.0
                    / (std * np.sqrt(2 * np.pi))
                    * np.exp(-0.5 * ((X - mean) / std) ** 2)
                )

            elif marginalDist.getImplementation().getClassName() == "Uniform":

                # set the delta vector which the right hand of the constraints
                # see eq 10 in Lemaitre paper
                mean = marginalDist.getMean()[0]
                deltaRHS = [mean, delta + mean**2]

                # find the optimal lambda for a given delta values
                optimalLambda = self.optimizeLambda(marginal, deltaRHS)
                psi = np.log(self.computeIntegral(marginal, optimalLambda, 0))
                return np.exp(
                    optimalLambda[0] * X + optimalLambda[1] * X**2 - psi
                ) * marginalDist.computePDF(X)
            else:
                raise NotImplementedError(
                    "Only Normal and Uniform distribution can be used."
                )

    def getOriginalDelta(self, marginal):
        """
        Accessor to the original variance value

        Parameters
        ----------
        marginal : int
            The indice of the perturbed marginal.
        """
        return self._distribution.getMarginal(marginal).getCovariance()[0, 0]

    def pdfExp(self, X, marginal, lamb, degree):
        """
        Common function used to compute integrals

        Parameters
        ----------
        X : 2d sequence of float
            The sample where the PDF is computed.
        marginal : int
            The indice of the perturbed marginal.
        lamb : sequence of float of dim 2
            The optimization parameters
        degree : int
            The wanted degree of the X values in the equation.
        """
        X = np.atleast_2d(X)
        return (
            X**degree
            * np.array(self._distribution.getMarginal(marginal).computePDF(X))
            * np.exp(lamb[0] * X + lamb[1] * X**2)
        )

    def computeIntegral(self, marginal, lamb, degree):
        """
        Compute the integrals using GaussKronrod algorithm
        """
        def func(X):
            return self.pdfExp(X, marginal, lamb, degree)[0]
        funcOT = ot.PythonFunction(1, 1, func)
        return self._gaussKronrod.integrate(
            funcOT, self._distribution.getMarginal(marginal).getRange()
        )[0]

    def H(self, marginal, lamb, deltaRHS):
        """
        Compute the Lagrange function

        Parameters
        ----------
        marginal : int
            The indice of the perturbed marginal.
        lamb : sequence of float of dim 2
            The optimization parameters
        deltaRHS : sequence of float of dim 2
            The values of the mean and variance + mean^2
        """
        expPsi = self.computeIntegral(marginal, lamb, 0)
        psi = np.log(expPsi)
        lamb = np.hstack(lamb)
        deltaRHS = np.hstack(deltaRHS)
        return psi - np.sum(lamb * deltaRHS)

    def gradH(self, marginal, lamb, deltaRHS):
        """
        Compute the gradient of the Lagrange function
        """
        expPsi = self.computeIntegral(marginal, lamb, 0)
        if expPsi == 0:
            raise ZeroDivisionError("Exponential Psi = 0")
        r1 = self.computeIntegral(marginal, lamb, 1)
        r2 = self.computeIntegral(marginal, lamb, 2)
        gradH1 = r1 / expPsi - deltaRHS[0]
        gradH2 = r2 / expPsi - deltaRHS[1]
        return np.vstack([gradH1, gradH2])

    def hessianH(self, marginal, lamb):
        """
        Compute the hessian of the Lagrange function
        """
        expPsi = self.computeIntegral(marginal, lamb, 0)
        r1 = self.computeIntegral(marginal, lamb, 1)
        r2 = self.computeIntegral(marginal, lamb, 2)
        r3 = self.computeIntegral(marginal, lamb, 3)
        r4 = self.computeIntegral(marginal, lamb, 4)
        hessH11 = r2 / expPsi - r1**2 / expPsi**2
        hessH12 = r3 / expPsi - r1 * r2 / expPsi**2
        hessH22 = r4 / expPsi - r2**2 / expPsi**2
        return np.atleast_2d([[hessH11, hessH12], [hessH12, hessH22]])

    def optimizeLambda(self, marginal, deltaRHS):
        """
        Compute the lambda values

        Parameters
        ----------
        marginal : int
            The indice of the perturbed marginal.
        deltaRHS : sequence of float of dim 2
            The values of the mean and variance + mean^2
        """

        # define the optimization function which the Lagrange function
        # and using the gradient and the hessian
        optimFunc = ot.PythonFunction(
            2,
            1,
            lambda lamb: [self.H(marginal, lamb, deltaRHS)],
            gradient=lambda lamb: self.gradH(marginal, lamb, deltaRHS),
            hessian=lambda lamb: self.hessianH(marginal, lamb),
        )

        # define the optimization problem
        optimPb = ot.OptimizationProblem(
            optimFunc,
            ot.Function(),
            ot.Function(),
            ot.Interval([-1000.0] * 2, [1000.0] * 2),
        )

        # solve the problem using SLSQP from NLopt
        optim = ot.NLopt(optimPb, "LD_SLSQP")
        optim.setStartingPoint([0, 0])
        optim.run()
        # return the lambda values, solution of the problem
        return optim.getResult().getOptimalPoint()
