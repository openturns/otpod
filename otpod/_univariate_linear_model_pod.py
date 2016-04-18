# -*- coding: utf-8 -*-
# -*- Python -*-

__all__ = ['UnivariateLinearModelPOD']

import openturns as ot
import math as m
from ._pod import POD
from ._math_tools import computeBoxCox, computeLinearParametersCensored
from ._math_tools import DataHandling
from statsmodels.regression.linear_model import OLS
import numpy as np


class _Results():
    """
    This class contains the result of the run. Instances are created
    for uncensored data or if needed for censored data.
    """
    def __init__(self):
        pass

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

        # if self._resDistFact is not None:
        #     if self._resDistFact.getClassName() == 'NormalFactory':
        #         raise Exception('Not yet implemented.')

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
        # self._resultsUnc
        # self._resultsCens
        
        # assertion input dimension is 1
        assert (self._dim == 1), "InputSample must be of dimension 1."


    def run(self):
        """
        Build the POD models.

        Notes
        -----
        This method build the linear model for the uncensored and censored case
        if required. Then it builds the POD model following the given residuals
        distribution factory.
        """

        # as it is required to build several time the linear model for the case
        # with bootstrap, the _run method of POD is not used here.
        results = _computeLinearModel(self._inputSample, self._outputSample,
                                      self._detection, self._noiseThres,
                                      self._saturationThres, self._boxCox,
                                      self._censored)
        defects = results['defects']
        # contains intercept, slope, stderr, residuals
        self._resultsUnc = results['uncensored']
        self._resultsCens = results['censored']
        # return the box cox detection even if box cox was not enabled. In this
        # case detection = detectionBoxCox
        self._detectionBoxCox = results['detection']

        ######################### build linear model ###########################
        # define the linear model
        def LinModel(x):
            return self._resultsUnc.intercept + self._resultsUnc.slope * x
        self._resultsUnc.linearModel = LinModel

        if self._censored:
             # define the linear model
            def LinModelCensored(x):
                return self._resultsCens.intercept + self._resultsCens.slope * x
            self._resultsCens.linearModel = LinModelCensored

        ######################## build PODModel function #######################
        if self._resDistFact is None:
            # Berens Binomial
            PODfunction = self._PODbinomialModel(self._resultsUnc.residuals,
                                                 self._resultsUnc.linearModel)
        elif self._resDistFact.getClassName() == 'NormalFactory':
            PODfunction = self._PODgaussModel(defects,
                                              self._resultsUnc.stderr,
                                              self._resultsUnc.linearModel)
        else:
            # Linear regression model + bootstrap
            PODfunction = self._PODbootstrapModel(self._resultsUnc.residuals,
                                                  self._resultsUnc.linearModel)

        self._resultsUnc.PODmodel = ot.PythonFunction(1, 1, PODfunction)

        # Create POD model for the censored case
        if self._censored:
            if self._resDistFact is None:
                # Berens Binomial
                PODfunction = self._PODbinomialModel(self._resultsCens.residuals,
                                                     self._resultsCens.linearModel)
            elif self._resDistFact.getClassName() == 'NormalFactory':
                PODfunction = self._PODgaussModel(defects,
                                                  self._resultsCens.stderr,
                                                  self._resultsCens.linearModel)
            else:
                # Linear regression model + bootstrap
                PODfunction = self._PODbootstrapModel(self._resultsCens.residuals,
                                                      self._resultsCens.linearModel)

            self._resultsCens.PODmodel = ot.PythonFunction(1, 1, PODfunction)


        ############## build PODModel function with conf interval ##############
        # Berens binomial : build directly in the get method
        if self._resDistFact is not None:
            if self._resDistFact.getClassName() == 'NormalFactory':
                # Linear regression with gaussian residuals hypothesis : build POD 
                # collection function for uncensored and censore if required.
                # The final PODmodelCl is built in the get method.
                self._PODcollDict= {'uncensored': self._PODgaussModelCl(defects,
                                    self._resultsUnc.intercept, self._resultsUnc.slope,
                                    self._resultsUnc.stderr, self._detectionBoxCox)}
                if self._censored:
                    self._PODcollDict['censored'] = self._PODgaussModelCl(defects,
                                    self._resultsCens.intercept, self._resultsCens.slope,
                                    self._resultsCens.stderr, self._detectionBoxCox)
            else:
                # Linear regression model + bootstrap : build the collection of function
                # for uncensored and censored case which is time consuming.
                # The final PODmodelCl is built in the get method.
                self._PODcollDict = self._PODbootstrapModelCl()



    def getPODModel(self, model='uncensored'):
        """
        Accessor to the POD model.

        Parameters
        ----------
        model : string
            The linear regression model to be used, either *uncensored* or
            *censored* if censored threshold were given. Default is *uncensored*.

        Returns
        -------
        PODModel : :py:class:`openturns.NumericalMathFunction`
            The function which computes the probability of detection for a given
            defect value.
        """

        # Check if the censored model exists when asking for it 
        if model == "censored" and not self._censored:
            raise NameError('POD model for censored data is not available.')

        if model == "uncensored":
            PODmodel = self._resultsUnc.PODmodel
        elif model == "censored":
            PODmodel = self._resultsCens.PODmodel
        else:
            raise NameError("model can be 'uncensored' or 'censored'.")

        return PODmodel

    def getPODCLModel(self, model='uncensored', confidenceLevel=0.95):
        """
        Accessor to the POD model at a given confidence level.

        Parameters
        ----------
        model : string
            The linear regression model to be used, either *uncensored* or
            *censored* if censored threshold were given. Default is *uncensored*.
        confidenceLevel : float
            The confidence level the POD must be computed. Default is 0.95

        Returns
        -------
        PODModelCl : :py:class:`openturns.NumericalMathFunction`
            The function which computes the probability of detection for a given
            defect value at the confidence level given as parameter.
        """

        # Check is the censored model exists when asking for it 
        if model == "censored" and not self._censored:
            raise NameError('Linear model for censored data is not available.')

        if model == "uncensored":
            if self._resDistFact is None:
                # Berens Binomial
                PODfunction = self._PODbinomialModelCl(self._resultsUnc.residuals,
                                                       self._resultsUnc.linearModel,
                                                       confidenceLevel)
            else:
                # Linear regression model + gaussian residuals or + bootstrap
                def PODfunction(x):
                    samplePODDef = ot.NumericalSample(self._simulationSize, 1)
                    for i in range(self._simulationSize):
                        samplePODDef[i] = [self._PODcollDict['uncensored'][i](x[0])]
                    return samplePODDef.computeQuantilePerComponent(1. - confidenceLevel)

            PODmodelCl = ot.PythonFunction(1, 1, PODfunction)

        elif model == "censored":
        # Create model conf interval for the censored case
            if self._resDistFact is None:
                # Berens Binomial
                PODfunction = self._PODbinomialModelCl(self._resultsCens.residuals,
                                                       self._resultsCens.linearModel,
                                                       confidenceLevel)
            else:
                # Linear regression model + gaussian residuals or + bootstrap
                def PODfunction(x):
                    samplePODDef = ot.NumericalSample(self._simulationSize, 1)
                    for i in range(self._simulationSize):
                        samplePODDef[i] = [self._PODcollDict['censored'][i](x[0])]
                    return samplePODDef.computeQuantilePerComponent(1. - confidenceLevel)

            PODmodelCl = ot.PythonFunction(1, 1, PODfunction)
        else:
            raise NameError("model can be 'uncensored' or 'censored'.")

        return PODmodelCl

    def computeDetectionSize(self, probabilityLevel, confidenceLevel=None):
        """
        Compute the detection size for a given probability level.

        Parameters
        ----------
        probabilityLevel : float
            The probability level for which the defect size is computed.
        confidenceLevel : float
            The confidence level associated to the given probability level the
            defect size is computed.

        Returns
        -------
        result : collection of :py:class:`openturns.NumericalPointWithDescription`
            A list of NumericalPointWithDescription containing the detection size
            computing for each case.
        """

        defectMin = self._inputSample.getMin()[0]
        defectMax = self._inputSample.getMax()[0]
        result = ot.NumericalPointWithDescriptionCollection()

        # compute 'a90' for uncensored model
        model = self.getPODModel()
        aProbLevel = ot.NumericalPointWithDescription(1, ot.Brent().solve(model,
                                        probabilityLevel, defectMin, defectMax))
        aProbLevel.setDescription(['a'+str(probabilityLevel*100)])
        result.add(aProbLevel)

        # compute 'a90_95' for uncensored model
        if confidenceLevel is not None:
            model = self.getPODCLModel(confidenceLevel=confidenceLevel)
            aProbLevelConfLevel = ot.NumericalPointWithDescription(1, ot.Brent().solve(model,
                                        probabilityLevel, defectMin, defectMax))
            aProbLevelConfLevel.setDescription(['a'+str(probabilityLevel*100)+'/'\
                                                +str(confidenceLevel*100)])
            result.add(aProbLevelConfLevel)

        if self._censored:
            # compute 'a90' for censored model
            model = self.getPODModel('censored')
            aProbLevel = ot.NumericalPointWithDescription(1, ot.Brent().solve(model,
                                            probabilityLevel, defectMin, defectMax))
            aProbLevel.setDescription(['a'+str(probabilityLevel*100)+' (censored)'])
            result.add(aProbLevel)

            # compute 'a90_95' for censored model
            if confidenceLevel is not None:
                model = self.getPODCLModel('censored', confidenceLevel)
                aProbLevelConfLevel = ot.NumericalPointWithDescription(1, ot.Brent().solve(model,
                                            probabilityLevel, defectMin, defectMax))
                aProbLevelConfLevel.setDescription(['a'+str(probabilityLevel*100)+'/'\
                                                    +str(confidenceLevel*100)+' (censored)'])
                result.add(aProbLevelConfLevel)
        
        return result

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
        X = ot.NumericalSample(defects.getSize(), [1, 0])
        X[:, 1] = defects
        X = ot.Matrix(X)
        # compute the prediction variance of the linear regression model
        def predictionVariance(x):
            Y = ot.NumericalPoint([1.0, x])
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
        X = ot.NumericalSample(N, [1, 0])
        X[:, 1] = defects
        X = ot.Matrix(X)
        covMatrix = X.computeGram(True).solveLinearSystem(ot.IdentityMatrix(2))
        sampleNormal = ot.Normal([0,0], ot.CovarianceMatrix(
                    covMatrix.getImplementation())).getSample(self._simulationSize)
        sampleSigmaEpsilon = (ot.Chi(N-2).inverse()*np.sqrt(N-2)*stderr).getSample(self._simulationSize)

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

                self.detection = results['detection']
                self.resultsUnc = results['uncensored']
                self.resultsCens = results['censored']
                self.resultsUnc.resDist = resDistFact.build(self.resultsUnc.residuals)
                if censored:
                    self.resultsCens.resDist = resDistFact.build(self.resultsCens.residuals)

            def PODmodel(self, x):
                defectThres = self.detection - (self.resultsUnc.intercept + 
                              self.resultsUnc.slope * x)
                return self.resultsUnc.resDist.computeComplementaryCDF(defectThres)

            def PODmodelCens(self, x):
                defectThres = self.detection - (self.resultsCens.intercept +
                              self.resultsCens.slope * x)
                return self.resultsCens.resDist.computeComplementaryCDF(defectThres)

        data = ot.NumericalSample(self._size, 2)
        data[:, 0] = self._inputSample
        data[:, 1] = self._outputSample
        # bootstrap of the data
        bootstrapExp = ot.BootstrapExperiment(data)
        PODcollUnc = []
        PODcollCens = []
        for i in range(self._simulationSize):
        # generate a sample with replacement within data of the same size
            bootstrapData = bootstrapExp.generate()
            # compute the linear models
            model = buildPODModel(bootstrapData[:,0], bootstrapData[:,1],
                                  self._detection, self._noiseThres,
                                  self._saturationThres, self._resDistFact,
                                  self._boxCox, self._censored)

            PODcollUnc.append(model.PODmodel)

            # computing in the censored case
            if self._censored:
                PODcollCens.append(model.PODmodelCens)

        return {'uncensored':PODcollUnc, 'censored':PODcollCens}


################################################################################
####################### Compute linear regression  #############################
################################################################################

def _computeLinearModel(inputSample, outputSample, detection, noiseThres,
                        saturationThres, boxCox, censored):
    """
    run the same code as in the linear analysis class but without the test
    this is much faster doing it.
    It is also needed for the POD bootstrap method.
    """

    ## create result container
    resultsUnc = _Results()
    resultsCens = _Results()
    #################### Filter censored data ##############################
    if censored:
        # check if one sided censoring
        if noiseThres is None:
            noiseThres = -ot.sys.float_info.max
        if saturationThres is None:
            saturationThres = ot.sys.float_info.max
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
        # optimization required, get optimal lambda without graph
        lambdaBoxCox, graph = computeBoxCox(defects, signals)

        # Transformation of data
        boxCoxTransform = ot.BoxCoxTransform([lambdaBoxCox])
        signals = boxCoxTransform(signals)
        if censored:
            if noiseThres != -ot.sys.float_info.max:
                noiseThres = boxCoxTransform([noiseThres])[0]
            if saturationThres != ot.sys.float_info.max:
                saturationThres = boxCoxTransform([saturationThres])[0]
        detectionBoxCox = boxCoxTransform([detection])[0]
    else:
        noiseThres = noiseThres
        saturationThres = saturationThres
        detectionBoxCox = detection

    ######################### Linear Regression model ######################
    # Linear regression with statsmodels module
    # Create the X matrix : [1, inputSample]
    X = ot.NumericalSample(defectsSize, [1, 0])
    X[:, 1] = defects
    algoLinear = OLS(np.array(signals), np.array(X)).fit()

    resultsUnc.intercept = algoLinear.params[0]
    resultsUnc.slope = algoLinear.params[1]
    # get standard error estimates (residuals standard deviation)
    resultsUnc.stderr = np.sqrt(algoLinear.scale)
    # get residuals from algoLinear
    resultsUnc.residuals = ot.NumericalSample(np.vstack(algoLinear.resid))

    if censored:
        # define initial starting point for MLE optimization
        initialStartMLE = [resultsUnc.intercept, resultsUnc.slope,
                           resultsUnc.stderr]
        # MLE optimization
        res = computeLinearParametersCensored(initialStartMLE, defects,
            defectsNoise, defectsSat, signals, noiseThres, saturationThres)
        resultsCens.intercept = res[0]
        resultsCens.slope = res[1]
        resultsCens.stderr = res[2]
        resultsCens.residuals = signals - (resultsCens.intercept + resultsCens.slope * defects)

    return {'defects':defects, 'signals':signals, 'uncensored':resultsUnc,
            'censored':resultsCens, 'detection':detectionBoxCox}