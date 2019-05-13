# -*- coding: utf-8 -*-
# -*- Python -*-

import openturns as ot
import numpy as np
from scipy.interpolate import interp1d
from ._decorator import DocInherit, keepingArgs

__all__ = []

class KrigingBase():
    """
    Base class for the KrigingPOD and AdaptiveSignalPOD which use the kriging
    metamodel. Both classes inherit methods from KrigingBase. 
    """
    def __init__(self):
        pass

    def getPODModel(self):
        """
        Accessor to the POD model.

        Returns
        -------
        PODModel : :py:class:`openturns.Function`
            The function which computes the probability of detection for a given
            defect value.
        """
        return self._PODmodel

    def getPODCLModel(self, confidenceLevel=0.95):
        """
        Accessor to the POD model at a given confidence level.

        Parameters
        ----------
        confidenceLevel : float
            The confidence level the POD must be computed. Default is 0.95

        Returns
        -------
        PODModelCl : :py:class:`openturns.Function`
            The function which computes the probability of detection for a given
            defect value at the confidence level given as parameter.
        """
        # Compute the quantile at the given confidence level for each
        # defect quantile and build the interpolate function.
        PODQuantile = self._PODPerDefect.computeQuantilePerComponent(
                                                            1. - confidenceLevel)
        interpModel = interp1d(self._defectSizes, PODQuantile, kind='linear')
        PODmodelCl = ot.PythonFunction(1, 1, interpModel)

        return PODmodelCl

    @DocInherit # decorator to inherit the docstring from POD class
    @keepingArgs # decorator to keep the real signature
    def computeDetectionSize(self, probabilityLevel, confidenceLevel=None):
        if confidenceLevel is None:
            return self._computeDetectionSize(self.getPODModel(),
                                          None,
                                          probabilityLevel,
                                          confidenceLevel,
                                          np.min(self._defectSizes),
                                          np.max(self._defectSizes))
        elif confidenceLevel is not None:
            return self._computeDetectionSize(self.getPODModel(),
                                          self.getPODCLModel(confidenceLevel),
                                          probabilityLevel,
                                          confidenceLevel,
                                          np.min(self._defectSizes),
                                          np.max(self._defectSizes))

    @DocInherit # decorator to inherit the docstring from POD class
    @keepingArgs # decorator to keep the real signature
    def drawPOD(self, probabilityLevel=None, confidenceLevel=None, defectMin=None,
                defectMax=None, nbPt=100, name=None):

        if defectMin is None:
            defectMin = np.min(self._defectSizes)
        else:
            if defectMin < np.min(self._defectSizes):
                raise ValueError('DefectMin must be greater than the minimum ' + \
                                 'of the given defect sizes.')
            if defectMin > np.max(self._defectSizes):
                raise ValueError('DefectMin must be lower than the maximum ' + \
                                 'of the given defect sizes.')
        if defectMax is None:
            defectMax = np.max(self._defectSizes)
        else:
            if defectMax > np.max(self._defectSizes):
                raise ValueError('DefectMax must be lower than the maximum ' + \
                                 'of the given defect sizes.')
            if defectMax < np.min(self._defectSizes):
                raise ValueError('DefectMax must be greater than the minimum ' + \
                                 'of the given defect sizes.')

        if confidenceLevel is None:
            fig, ax = self._drawPOD(self.getPODModel(), None,
                                probabilityLevel, confidenceLevel, defectMin,
                                defectMax, nbPt, name)
        elif confidenceLevel is not None:
            fig, ax = self._drawPOD(self.getPODModel(), self.getPODCLModel(confidenceLevel),
                    probabilityLevel, confidenceLevel, defectMin,
                    defectMax, nbPt, name)

        ax.set_title('POD - Kriging model')
        if name is not None:
            fig.savefig(name, bbox_inches='tight', transparent=True)

        return fig, ax

    @DocInherit # decorator to inherit the docstring from POD class
    @keepingArgs # decorator to keep the real signature
    def drawValidationGraph(self, name=None):

        y_loo = self._computeLOO(self._input, self._signals, self._krigingResult)
        fig, ax = self._drawValidationGraph(self._signals, y_loo)
        ax.set_title("Validation of the Kriging model")
        ax.set_ylabel('Predicted leave one out signals')

        if name is not None:
            fig.savefig(name, bbox_inches='tight', transparent=True)
        return fig, ax

    def getQ2(self):
        """
        Accessor to the Q2 value. 

        Returns
        -------
        Q2 : float
            The Q2 value computed analytically using Dubrule (1983) technique.
        """
        return self._Q2

    def getKrigingResult(self):
        """
        Accessor to the kriging result.

        Returns
        -------
        result : :py:class:`openturns.KrigingResult`
            The kriging result.
        """
        if self._krigingResult is None:
            print('The run method must be launched first.')
        else:
            return self._krigingResult

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


    def getInitialStartSize(self):
        """
        Accessor to the size of the multi start algorithm.

        Returns
        -------
        size : int
            The number of multi start using the TNC algorithm to find the 
            covariance parameters. Default is 100.
        """
        return self._initialStartSize

    def setInitialStartSize(self, size):
        """
        Accessor to the size of the multi start algorithm.

        Parameters
        ----------
        size : int
            The number of multi start using the TNC algorithm to find the 
            covariance parameters.
        """
        self._initialStartSize = size

    def getDefectSizes(self):
        """
        Accessor to the defect size where POD is computed.

        Returns
        -------
        defectSize : sequence of float
            The defect sizes where the Monte Carlo simulation is performed to
            compute the POD.
        """
        return self._defectSizes

    def setDefectSizes(self, size):
        """
        Accessor to the defect size where POD is computed.

        Parameters
        ----------
        defectSize : sequence of float
            The defect sizes where the Monte Carlo simulation is performed to
            compute the POD.
        """
        size = np.hstack(np.array(size))
        size.sort()
        self._defectSizes = size.copy()
        minMin = self._input[:, 0].getMin()[0]
        maxMax = self._input[:, 0].getMax()[0]
        if size.max() > maxMax or size.min() < minMin:
            raise ValueError('Defect sizes must range between ' + \
                             '{:0.4f} '.format(np.ceil(minMin*10000)/10000) + \
                             'and {:0.4f}.'.format(np.floor(maxMax*10000)/10000))
        self._defectNumber = self._defectSizes.shape[0]

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
        if self._distribution is None:
            print('The run method must be launched first.')
        else:
            return self._distribution

    def setBasis(self, basis):
        """
        Accessor to the kriging basis. 

        Parameters
        ----------
        basis : :py:class:`openturns.Basis`
            The basis used as trend in the kriging model.
        """
        try:
            ot.Basis(basis)
        except NotImplementedError:
            raise Exception('The given parameter is not a Basis.')
        self._basis = basis

    def getBasis(self):
        """
        Accessor to the kriging basis. 

        Returns
        -------
        basis : :py:class:`openturns.Basis`
            The basis used as trend in the kriging model. Default is a linear
            basis for the defect and constant for the other parameters.
        """
        if self._basis is None:
            print('The run method must be launched first.')
        else:
            return self._basis

    def setCovarianceModel(self, covarianceModel):
        """
        Accessor to the kriging covariance model. 

        Parameters
        ----------
        covarianceModel : :py:class:`openturns.CovarianceModel`
            The covariance model in the kriging model.
        """
        try:
            ot.CovarianceModel(covarianceModel)
        except NotImplementedError:
            raise Exception('The given parameter is not a CovarianceModel.')
        self._covarianceModel = covarianceModel

    def getCovarianceModel(self):
        """
        Accessor to the kriging covariance model. 

        Returns
        -------
        covarianceModel : :py:class:`openturns.CovarianceModel`
            The covariance model in the kriging model. Default is an anisotropic
            Squared exponential covariance model.
        """
        if self._covarianceModel is None:
            print('The run method must be launched first.')
        else:
            return self._covarianceModel

    def getVerbose(self):
        """
        Accessor to the verbosity.

        Returns
        -------
        verbose : bool
            Enable or disable the verbosity. Default is True. 
        """
        return self._verbose

    def setVerbose(self, verbose):
        """
        Accessor to the verbosity.

        Parameters
        ----------
        verbose : bool
            Enable or disable the verbosity.
        """
        if type(verbose) is not bool:
            raise TypeError('The parameter is not a bool.')
        else:
            self._verbose = verbose


    def _buildKrigingAlgo(self, inputSample, outputSample):
        """
        Build the functional chaos algorithm without running it.
        """
        if self._basis is None:
            # create linear basis only for the defect parameter (1st parameter),
            # constant otherwise
            input = ['x'+str(i) for i in range(self._dim)]
            functions = []
            # constant
            functions.append(ot.SymbolicFunction(input, ['1']))
            # linear for the first parameter only
            functions.append(ot.SymbolicFunction(input, [input[0]]))
            self._basis = ot.Basis(functions)

        if self._covarianceModel is None:
            # anisotropic squared exponential covariance model
            self._covarianceModel = ot.SquaredExponential([1] * self._dim)

        algoKriging = ot.KrigingAlgorithm(inputSample, outputSample,
                                          self._covarianceModel, self._basis, True)
        return algoKriging

    def _computePODSamplePerDefect(self, defect, detection, krigingResult,
                                  distribution, simulationSize, samplingSize):
        """
        Compute the POD sample for a defect size.
        """

        dim = distribution.getDimension()
        # create a distibution with a dirac distribution for the defect size
        diracDist = [ot.Dirac(defect)]
        diracDist += [distribution.getMarginal(i+1) for i in range(dim-1)]
        distribution = ot.ComposedDistribution(diracDist)

        # create a sample for the Monte Carlo simulation and confidence interval
        MC_sample = distribution.getSample(samplingSize)
        # Kriging_RV = ot.KrigingRandomVector(krigingResult, MC_sample)
        # Y_sample = Kriging_RV.getSample(simulationSize)
        Y_sample = self._randomVectorSampling(krigingResult, MC_sample,
                                        simulationSize, samplingSize)

        # compute the POD for all simulation size
        POD_MCPG_a = np.mean(Y_sample > detection, axis=1)
        # compute the variance of the MC simulation using TCL
        VAR_TCL = np.array(POD_MCPG_a)*(1-np.array(POD_MCPG_a)) / Y_sample.shape[1]
        # Create distribution of the POD estimator for all simulation 
        POD_PG_dist = []
        for i in range(simulationSize):
            if VAR_TCL[i] > 0:
                POD_PG_dist += [ot.Normal(POD_MCPG_a[i],np.sqrt(VAR_TCL[i]))]
            else:
                if POD_MCPG_a[i] < 1:
                    POD_PG_dist += [ot.Dirac([0.])]
                else:
                    POD_PG_dist += [ot.Dirac([1.])]
        POD_PG_alea = ot.Mixture(POD_PG_dist)
        # get a sample of these distributions
        POD_PG_sample = POD_PG_alea.getSample(simulationSize * samplingSize)

        return POD_PG_sample

    def _randomVectorSampling(self, krigingResult, sample, simulationSize, samplingSize):
        """
        Kriging Random vector perso
        """
        
        # only compute the variance
        variance = np.hstack([krigingResult.getConditionalCovariance(
                            sample[i])[0,0] for i in range(samplingSize)])
        pred = krigingResult.getConditionalMean(sample)

        normalSample = ot.Normal().getSample(simulationSize)
        # with numpy broadcasting
        randomVector = np.array(normalSample)* np.sqrt(variance) + np.array(pred)
        return randomVector


    def _estimKrigingTheta(self, algoKriging, lowerBound, upperBound, size):
        """
        Estimate the kriging theta values with an initial random search using
        a Sobol sequence of size samples.
        """

        if size > 0:
            # create uniform distribution of the parameters bounds
            dim = len(lowerBound)
            distBoundCol = []
            for i in range(dim):
                distBoundCol += [ot.Uniform(lowerBound[i], upperBound[i])]
            distBound = ot.ComposedDistribution(distBoundCol)

            # set the bounds
            searchInterval = ot.Interval(lowerBound, upperBound)
            algoKriging.setOptimizationBounds(searchInterval)
            # Generate starting points with a low discrepancy sequence
            startingPoint = ot.LowDiscrepancyExperiment(ot.SobolSequence(),
                                                        distBound, size).generate()

            algoKriging.setOptimizationAlgorithm(ot.MultiStart(ot.TNC(), startingPoint))
        else:
            algoKriging.setOptimizeParameters(False)

        return algoKriging


    def _computeLOO(self, inputSample, outputSample, krigingResult):
        """
        Compute the Leave One out prediction analytically.
        """
        inputSample = np.array(inputSample)
        outputSample = np.array(outputSample)

        # get covariance model
        cov = krigingResult.getCovarianceModel()
        # get input transformation
        t = krigingResult.getTransformation()
        # check if the transformation was enabled or not and if so transform
        # the input sample
        if t.getInputDimension() == inputSample.shape[1]:
            normalized_inputSample = np.array(t(inputSample))
        else:
            normalized_inputSample = inputSample

        K = cov.discretize(normalized_inputSample)
        # get coefficient and compute trend
        basis = krigingResult.getBasisCollection()[0]
        F1 = krigingResult.getTrendCoefficients()[0]
        size = inputSample.shape[0]
        p = F1.getDimension()
        F = np.ones((size, p))
        for i in range(p):
            F[:, i] = np.hstack(basis.build(i)(normalized_inputSample))
        # Calcul de y_loo
        Z = np.zeros((p, p))
        S = np.vstack([np.hstack([K, F]), np.hstack([F.T, Z])])
        S_inv = np.linalg.inv(S)
        B = S_inv[:size:, :size:]
        B_but_its_diag = B * (np.ones(B.shape) - np.eye(size))
        B_diag = np.atleast_2d(np.diag(B)).T
        y_loo = (- np.dot(B_but_its_diag / B_diag, outputSample)).ravel()
        return y_loo

    def _computeQ2(self, inputSample, outputSample, krigingResult):
        """
        Compute the Q2 using the analytical loo prediction.
        """
        y_loo = self._computeLOO(inputSample, outputSample, krigingResult)
        # Calcul du Q2
        delta = (np.hstack(outputSample) - y_loo)
        return 1 - np.mean(delta**2)/np.var(outputSample)

