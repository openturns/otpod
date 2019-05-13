# -*- coding: utf-8 -*-
# -*- Python -*-

__all__ = ['UnivariateLinearModelPOD']

import openturns as ot
import math as m
from ._pod import POD
from ._math_tools import computeBoxCox, DataHandling, computeLinearParametersCensored, \
                         computeR2
from statsmodels.regression.linear_model import OLS
import numpy as np
from ._decorator import DocInherit, keepingArgs
from ._progress_bar import updateProgress


class UnivariateLinearModelPOD(POD):

    """
    Linear regression based POD.

    **Available constructors:**

    UnivariateLinearModelPOD(*analysis=analysis, detection=detection*)

    UnivariateLinearModelPOD(*inputSample, outputSample, detection, noiseThres,
    saturationThres, resDistFact, boxCox*)

    Parameters
    ----------
    analysis : :class:`UnivariateLinearModelAnalysis`
        Linear analysis object.
    inputSample : 2-d sequence of float
        Vector of the defect sizes, of dimension 1.
    outputSample : 2-d sequence of float
        Vector of the signals, of dimension 1.
    detection : float
        Detection value of the signal.
    noiseThres : float
        Value for low censored data. Default is None.
    saturationThres : float
        Value for high censored data. Default is None
    resDistFact : :py:class:`openturns.DistributionFactory`
        Distribution hypothesis followed by the residuals. Default is None.
    boxCox : bool or float
        Enable or not the Box Cox transformation. If boxCox is a float, the Box
        Cox transformation is enabled with the given value. Default is False.

    Notes
    -----
    This class aims at building the POD based on a linear regression
    model. If a linear analysis has been launched, it can be used as prescribed 
    in the first constructor. It can be noticed that, in this case, with the
    default parameters of the linear analysis, the POD will corresponds with the
    linear regression model associated to a Gaussian hypothesis on the residuals.


    Otherwise, all parameters can be given as in the second constructor.

    Following the given distribution in *resDistFact*, the POD model is built
    different hypothesis:

    - if *resDistFact = None*, it corresponds with Berens-Binomial. This
      is the default case. 
    - if *resDistFact* = :py:class:`openturns.NormalFactory`, it corresponds with Berens-Gauss.
    - if *resDistFact* = {:py:class:`openturns.KernelSmoothing`,
      :py:class:`openturns.WeibullFactory`, ...}, the confidence interval is
      built by bootstrap.

    If bootstrap is used, a progress bar is shown if the verbosity is enabled.
    It can be disabled using the method *setVerbose*.
    """

    def __init__(self, inputSample=None, outputSample=None, detection=None, noiseThres=None,
                 saturationThres=None, resDistFact=None, boxCox=False,
                 analysis=None):

        #  Constructor with analysis given, check if analysis is only given with detection.
        self._analysis = analysis
        if self._analysis is not None:
            try:
                assert (inputSample is None)
                assert (outputSample is None)
                assert (noiseThres is None)
                assert (saturationThres is None)
                assert (resDistFact is None)
                assert (detection is not None)
            except:
                raise AttributeError('The constructor available with a linear '+\
                                     'analysis as parameter must only have ' + \
                                     'the detection parameter.')

            # get back informations from analysis on input parameters
            inputSample = self._analysis.getInputSample()
            outputSample = self._analysis.getOutputSample()
            noiseThres = self._analysis.getNoiseThreshold()
            saturationThres = self._analysis.getSaturationThreshold()
            boxCox = self._analysis.getBoxCoxParameter()
            self._resDistFact = self._analysis._resDistFact
        else:
            # residuals distribution factory attributes
            self._resDistFact = resDistFact

        self._verbose = True

        # initialize the POD class
        super(UnivariateLinearModelPOD, self).__init__(inputSample, outputSample,
                                 detection, noiseThres, saturationThres, boxCox)
        # inherited attributes
        # self._simulationSize
        # self._detection
        # self._inputSample
        # self._outputSample
        # self._noiseThres
        # self._saturationThres        
        # self._lambdaBoxCox
        # self._boxCox
        # self._size
        # self._dim
        # self._censored
        
        # assertion input dimension is 1
        assert (self._dim == 1), "Dimension inputSample must be 1."


    def run(self):
        """
        Build the POD models.

        Notes
        -----
        This method build the linear model for the uncensored or censored case
        depending of the input parameters. Then it builds the POD model
        following the given residuals distribution factory.
        """

        # as it is required to build several time the linear model for the case
        # with bootstrap, the _run method of POD is not used here.
        results = _computeLinearModel(self._inputSample, self._outputSample,
                                      self._detection, self._noiseThres,
                                      self._saturationThres, self._boxCox,
                                      self._censored)
        # get results
        self._defects = results['defects']
        self._signals = results['signals']
        self._intercept = results['intercept']
        self._slope = results['slope']
        self._stderr = results['stderr']
        self._residuals = results['residuals']
        self._lambdaBoxCox = results['lambdaBoxCox']
        self._graphBoxCox = results['graphBoxCox']
        # return the box cox detection even if box cox was not enabled. In this
        # case detection = detectionBoxCox
        self._detectionBoxCox = results['detection']

        ######################### build linear model ###########################
        # define the linear model
        def LinModel(x):
            return self._intercept + self._slope * x
        self._linearModel = LinModel


        ######################## build PODModel function #######################
        if self._resDistFact is None:
            # Berens Binomial
            PODfunction = self._PODbinomialModel(self._residuals,
                                                 self._linearModel)
        elif self._resDistFact.getClassName() == 'NormalFactory':
            PODfunction = self._PODgaussModel(self._defects,
                                              self._stderr,
                                              self._linearModel)
        else:
            # Linear regression model + bootstrap
            PODfunction = self._PODbootstrapModel(self._residuals,
                                                  self._linearModel)

        self._PODmodel = ot.PythonFunction(1, 1, PODfunction)


        ############## build PODModel function with conf interval ##############
        # Berens binomial : build directly in the get method
        if self._resDistFact is not None:
            if self._resDistFact.getClassName() == 'NormalFactory':
                # Linear regression with gaussian residuals hypothesis : build POD 
                # collection function.
                # The final PODmodelCl is built in the get method.
                self._PODcollDict= self._PODgaussModelCl(self._defects,
                                    self._intercept, self._slope,
                                    self._stderr, self._detectionBoxCox)
            else:
                # Linear regression model + bootstrap : build the collection of 
                # functions which is time consuming.
                # The final PODmodelCl is built in the get method.
                self._PODcollDict = self._PODbootstrapModelCl()

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

        if self._resDistFact is None:
            # Berens Binomial
            PODfunction = self._PODbinomialModelCl(self._residuals,
                                                   self._linearModel,
                                                   confidenceLevel)
        else:
            # Linear regression model + gaussian residuals or + bootstrap
            def PODfunction(x):
                samplePODDef = ot.Sample(self._simulationSize, 1)
                for i in range(self._simulationSize):
                    samplePODDef[i] = [self._PODcollDict[i](x[0])]
                return samplePODDef.computeQuantilePerComponent(1. - confidenceLevel)

        PODmodelCl = ot.PythonFunction(1, 1, PODfunction)

        return PODmodelCl

    def getR2(self):
        """
        Accessor to the R2 value. 

        Returns
        -------
        R2 : float
            The R2 value.
        """
        return computeR2(self._signals, self._residuals)

    @DocInherit # decorator to inherit the docstring from POD class
    @keepingArgs # decorator to keep the real signature
    def computeDetectionSize(self, probabilityLevel, confidenceLevel=None):
        return self._computeDetectionSize(self.getPODModel(),
                                          self.getPODCLModel(confidenceLevel),
                                          probabilityLevel,
                                          confidenceLevel)

    @DocInherit # decorator to inherit the docstring from POD class
    @keepingArgs # decorator to keep the real signature
    def drawPOD(self, probabilityLevel=None, confidenceLevel=None, defectMin=None,
                defectMax=None, nbPt=100, name=None):

        if confidenceLevel is None:
            fig, ax = self._drawPOD(self.getPODModel(), None,
                                probabilityLevel, confidenceLevel, defectMin,
                                defectMax, nbPt, name)
        elif confidenceLevel is not None:
            fig, ax = self._drawPOD(self.getPODModel(), self.getPODCLModel(confidenceLevel),
                    probabilityLevel, confidenceLevel, defectMin,
                    defectMax, nbPt, name)

        if self._resDistFact is None:
            ax.set_title('POD - Linear regression model - Binomial')
        else:
            if self._resDistFact.getClassName() =='NormalFactory':
                ax.set_title('POD - Linear regression model - Gauss')
            else:
                ax.set_title('POD - Linear regression model - ' + \
                              str(self._resDistFact.getClassName()))

        if name is not None:
            fig.savefig(name, bbox_inches='tight', transparent=True)

        return fig, ax

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


################################################################################
####################### Linear regression Binomial #############################
################################################################################

    def _PODbinomialModel(self, residuals, linearModel):
        empiricalDist = ot.UserDefined(residuals)
        # function to compute the POD(defect)
        def PODmodel(x):
            def_threshold = self._detectionBoxCox - linearModel(x[0])
            # Nb of residuals > threshold(defect) / N
            return [empiricalDist.computeComplementaryCDF(def_threshold)]
        return PODmodel

    def _PODbinomialModelCl(self, residuals, linearModel, confLevel):
        empiricalDist = ot.UserDefined(residuals)
        sizeResiduals = residuals.getSize()
        def PODmodelCl(x):
            # Nb of residuals > threshold - linModel(defect)
            def_threshold = self._detectionBoxCox - linearModel(x[0])
            NbDepDef = m.trunc(sizeResiduals * empiricalDist.computeComplementaryCDF(def_threshold))
            # Particular case : NbDepDef == sizeResiduals
            if NbDepDef == sizeResiduals:
                pod = confLevel**(1. / sizeResiduals)
            else:
                # 1 - quantile(confLevel) of distribution Beta(r, s)
                pod = 1-ot.DistFunc.qBeta(sizeResiduals - NbDepDef, NbDepDef + 1, confLevel)
            return [pod]
        return PODmodelCl

################################################################################
####################### Linear regression Gauss ################################
################################################################################

    def _PODgaussModel(self, defects, stderr, linearModel):
        X = ot.Sample(defects.getSize(), [1, 0])
        X[:, 1] = defects
        X = ot.Matrix(X)
        # compute the prediction variance of the linear regression model
        def predictionVariance(x):
            Y = ot.Point([1.0, x])
            gramX = X.computeGram()
            return stderr**2 * (1. + ot.dot(Y, gramX.solveLinearSystem(Y)))
        # function to compute the POD(defect)
        def PODmodel(x):
            t = (self._detectionBoxCox - linearModel(x[0])) / np.sqrt(predictionVariance(x[0]))
            # DistFunc.pNormal(t,True) = complementary CDF of the Normal(0,1)
            return [ot.DistFunc.pNormal(t,True)]
        return PODmodel

    def _PODgaussModelCl(self, defects, intercept, slope, stderr, detection):

        class buildPODModel():
            def __init__(self, intercept, slope, sigmaEpsilon, detection):

                self.intercept = intercept
                self.slope = slope
                self.sigmaEpsilon = sigmaEpsilon
                self.detection = detection

            def PODmodel(self, x):
                t = (self.detection - (self.intercept + 
                              self.slope * x)) / self.sigmaEpsilon
                return ot.DistFunc.pNormal(t,True)

        N = defects.getSize()
        X = ot.Sample(N, [1, 0])
        X[:, 1] = defects
        X = ot.Matrix(X)
        covMatrix = X.computeGram(True).solveLinearSystem(ot.IdentityMatrix(2))
        sampleNormal = ot.Normal([0,0], ot.CovarianceMatrix(
                    covMatrix.getImplementation())).getSample(self._simulationSize)
        sampleSigmaEpsilon = (ot.Chi(N-2).inverse()*np.sqrt(N-2)*stderr).getSample(
                                                            self._simulationSize)

        PODcoll = []
        for i in range(self._simulationSize):
            sigmaEpsilon = sampleSigmaEpsilon[i][0]
            interceptSimu = sampleNormal[i][0] * sigmaEpsilon + intercept
            slopeSimu = sampleNormal[i][1] * sigmaEpsilon + slope
            PODcoll.append(buildPODModel(interceptSimu, slopeSimu, sigmaEpsilon,
                                         detection).PODmodel)
        return PODcoll

################################################################################
####################### Linear regression bootstrap ############################
################################################################################

    def _PODbootstrapModel(self, residuals, linearModel):
        empiricalDist = self._resDistFact.build(residuals)
        # function to compute the POD(defects)
        def PODmodel(x):
            def_threshold = self._detectionBoxCox - linearModel(x[0])
            # Nb of residuals > threshold(defect) / N
            return [empiricalDist.computeComplementaryCDF(def_threshold)]
        return PODmodel

    def _PODbootstrapModelCl(self):

        class buildPODModel():
            def __init__(self, inputSample, outputSample, detection, noiseThres,
                            saturationThres, resDistFact, boxCox, censored):

                results = _computeLinearModel(inputSample, outputSample, detection,
                                        noiseThres, saturationThres, boxCox, censored)

                self.intercept = results['intercept']
                self.slope = results['slope']
                self.residuals = results['residuals']
                self.detectionBoxCox = results['detection']
                self.resDist = resDistFact.build(self.residuals)

            def PODmodel(self, x):
                defectThres = self.detectionBoxCox - (self.intercept + 
                              self.slope * x)
                return self.resDist.computeComplementaryCDF(defectThres)


        data = ot.Sample(self._size, 2)
        data[:, 0] = self._inputSample
        data[:, 1] = self._outputSample
        # bootstrap of the data
        bootstrapExp = ot.BootstrapExperiment(data)
        PODcoll = []
        for i in range(self._simulationSize):
        # generate a sample with replacement within data of the same size
            bootstrapData = bootstrapExp.generate()
            # compute the linear models
            model = buildPODModel(bootstrapData[:,0], bootstrapData[:,1],
                                  self._detection, self._noiseThres,
                                  self._saturationThres, self._resDistFact,
                                  self._boxCox, self._censored)

            PODcoll.append(model.PODmodel)
            if self._verbose:
                updateProgress(i, self._simulationSize, 'Computing POD (bootstrap)')

        return PODcoll

################################################################################
####################### Compute linear regression  #############################
################################################################################

def _computeLinearModel(inputSample, outputSample, detection, noiseThres,
                        saturationThres, boxCox, censored):
    """
    Run filerCensoredData and build the linear regression model.
    It is defined as a simple function because it is also needed in a loop for
    the bootstrap based POD.
    """

    #################### Filter censored data ##############################
    if censored:
        # Filter censored data
        defects, defectsNoise, defectsSat, signals = \
            DataHandling.filterCensoredData(inputSample, outputSample,
                          noiseThres, saturationThres)
    else:
        defects, signals = inputSample, outputSample

    defectsSize = defects.getSize()

    ###################### Box Cox transformation ##########################
    # Compute Box Cox if enabled
    if boxCox:
        if signals.getMin()[0] < 0:
            shift = - signals.getMin()[0] + 100
        else:
            shift = 0.

        # optimization required, get optimal lambda without graph
        lambdaBoxCox, graphBoxCox = computeBoxCox(defects, signals, shift)

        # Transformation of data
        boxCoxTransform = ot.BoxCoxTransform([lambdaBoxCox])
        signals = boxCoxTransform(signals + shift)
        if censored:
            if noiseThres is not None:
                noiseThres = boxCoxTransform([noiseThres + shift])[0]
            if saturationThres is not None:
                saturationThres = boxCoxTransform([saturationThres + shift])[0]
        detectionBoxCox = boxCoxTransform([detection + shift])[0]
    else:
        detectionBoxCox = detection
        lambdaBoxCox = None
        graphBoxCox = None

    ######################### Linear Regression model ######################
    # Linear regression with statsmodels module
    # Create the X matrix : [1, inputSample]
    X = ot.Sample(defectsSize, [1, 0])
    X[:, 1] = defects
    algoLinear = OLS(np.array(signals), np.array(X)).fit()

    intercept = algoLinear.params[0]
    slope = algoLinear.params[1]
    # get standard error estimates (residuals standard deviation)
    stderr = np.sqrt(algoLinear.scale)
    # get residuals from algoLinear
    residuals = ot.Sample(np.vstack(algoLinear.resid))

    if censored:
        # define initial starting point for MLE optimization
        initialStartMLE = [intercept, slope, stderr]
        # MLE optimization
        res = computeLinearParametersCensored(initialStartMLE, defects,
            defectsNoise, defectsSat, signals, noiseThres, saturationThres)
        intercept = res[0]
        slope = res[1]
        stderr = res[2]
        residuals = signals - (intercept + slope * defects)

    return {'defects':defects, 'signals':signals, 'intercept':intercept,
            'slope':slope, 'stderr':stderr, 'residuals':residuals,
            'detection':detectionBoxCox, 'lambdaBoxCox':lambdaBoxCox,
            'graphBoxCox':graphBoxCox}
