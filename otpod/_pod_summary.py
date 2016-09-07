# -*- coding: utf-8 -*-
# -*- Python -*-

__all__ = ['PODSummary']

import openturns as ot
import numpy as np
from ._univariate_linear_model_analysis import UnivariateLinearModelAnalysis
from ._univariate_linear_model_pod import UnivariateLinearModelPOD
from ._quantile_regression_pod import QuantileRegressionPOD
from ._polynomial_chaos_pod import PolynomialChaosPOD
from ._kriging_pod import KrigingPOD
import logging
import os


class PODSummary():
    """
    Run the analysis and compute POD with several methods.

    **Available constructor:**

    PODSummary(*inputSample, outputSample, detection, noiseThres,
    saturationThres, boxCox*)

    Parameters
    ----------
    inputSample : 2-d sequence of float
        Vector of the input values. The first column must correspond with the
        defect sizes.
    outputSample : 2-d sequence of float
        Vector of the signals, of dimension 1.
    detection : float
        Detection value of the signal.
    noiseThres : float
        Value for low censored data. Default is None.
    saturationThres : float
        Value for high censored data. Default is None
    boxCox : bool or float
        Enable or not the Box Cox transformation. If boxCox is a float, the Box
        Cox transformation is enabled with the given value. Default is False.

    Warnings
    --------
    The first column of the input sample must corresponds with the defects sample.

    Notes
    -----
    This class aims at running the linear analysis and computing the POD with
    different models:

    - Linear regression model with Gaussian residuals hypothesis,
    - Linear regression model with no hypothesis on the residuals (binomial),
    - Linear regression model with with kernel smoothing on the residuals,
    - Quantile regression,
    - Polynomial chaos,
    - kriging if the dimension of the input sample is greater than 1.

    Each method can be deactivated using the method *setMethodActive* and using
    the key corresponding to the method.

    All results can be displayed and saved thanks to the methods *printResults*, 
    *saveResults* and *saveGraphs*. For each method, the probability level and
    confidence level can be specified in order to compute the defect size to 
    the wanted probability level. 

    The verbosity is enabled by default but it can be disabled using the method
    *setVerbose*.
    """

    def __init__(self, inputSample=None, outputSample=None, detection=None, noiseThres=None,
                 saturationThres=None, boxCox=False):


        self._inputSample = ot.NumericalSample(np.vstack(inputSample))
        self._signals = ot.NumericalSample(np.vstack(outputSample))
        self._detection = detection
        self._noiseThres = noiseThres
        self._saturationThres = saturationThres
        self._boxCox = boxCox

        # Add flag to tell if censored data must taken into account or not.
        if self._noiseThres is not None or self._saturationThres is not None:
            # flag to tell censoring is enabled
            self._censored = True
        else:
            self._censored = False

        self._dim = self._inputSample.getDimension()

        self._verbose = True
        self._simulationSize = 1000
        self._samplingSize = 5000

        self._PODgauss = None
        self._PODbin = None
        self._PODks = None
        self._PODqr = None
        self._PODchaos = None
        self._PODkriging = None

        self._activeMethods = {'LinearGauss':True, 'LinearBinomial':True,
                         'LinearKernelSmoothing':True, 'QuantileRegression':True,
                         'PolynomialChaos':True, 'Kriging':True}

    def run(self):
        """
        Run all active methods.
        """

        # run the univariate linear model analysis with gaussian residuals hypothesis
        if self._verbose:
            print "\nStart univariate linear model analysis..."
        self._analysis = UnivariateLinearModelAnalysis(self._inputSample[:, 0],
                                                 self._signals, self._noiseThres,
                                                 self._saturationThres,
                                                 ot.NormalFactory(), self._boxCox)

        # run the univariate linear model with gaussian residuals
        if self._activeMethods['LinearGauss']:
            if self._verbose:
                print "\nStart univariate linear model POD with Gaussian residuals..."
            self._PODgauss = UnivariateLinearModelPOD(self._inputSample[:, 0], self._signals,
                                                self._detection, self._noiseThres,
                                                self._saturationThres,
                                                ot.NormalFactory(), self._boxCox)
            self._PODgauss.setVerbose(self._verbose)
            self._PODgauss.setSimulationSize(self._simulationSize)
            self._PODgauss.run()


        # run the univariate linear model with no hypothesis on the residuals
        if self._activeMethods['LinearBinomial']:
            if self._verbose:
                print "\nStart univariate linear model POD with no hypothesis on the residuals..."
            self._PODbin = UnivariateLinearModelPOD(self._inputSample[:, 0], self._signals,
                                                self._detection, self._noiseThres,
                                                self._saturationThres,
                                                None, self._boxCox)
            self._PODbin.setVerbose(self._verbose)
            self._PODbin.run()

        # run the univariate linear model with kernel smoothing on the residuals
        if self._activeMethods['LinearKernelSmoothing']:
            if self._verbose:
                print "\nStart univariate linear model POD with kernel smoothing on the residuals..."
            self._PODks = UnivariateLinearModelPOD(self._inputSample[:, 0], self._signals,
                                                self._detection, self._noiseThres,
                                                self._saturationThres,
                                                ot.KernelSmoothing(), self._boxCox)
            self._PODks.setVerbose(self._verbose)
            self._PODks.setSimulationSize(self._simulationSize)
            self._PODks.run()

        # run the quantile regression 
        if self._activeMethods['QuantileRegression']:
            if self._verbose:
                print "\nStart quantile regression POD..."
            self._PODqr = QuantileRegressionPOD(self._inputSample[:, 0], self._signals,
                                                self._detection, self._noiseThres,
                                                self._saturationThres, self._boxCox)
            self._PODqr.setVerbose(self._verbose)
            self._PODqr.setSimulationSize(self._simulationSize)
            self._PODqr.run()


        # run the polynomial chaos
        if self._activeMethods['PolynomialChaos']:
            if self._verbose:
                print "\nStart polynomial chaos POD..."
            self._PODchaos = PolynomialChaosPOD(self._inputSample, self._signals,
                                       self._detection, self._noiseThres,
                                       self._saturationThres, self._boxCox)
            self._PODchaos.setVerbose(self._verbose)
            self._PODchaos.setSimulationSize(self._simulationSize)
            self._PODchaos.setSamplingSize(self._samplingSize)
            self._PODchaos.run()

        # run the kriging
        if self._dim > 1 and self._activeMethods['Kriging']:
            if self._verbose:
                print "\nStart kriging POD..."
            self._PODkriging = KrigingPOD(self._inputSample, self._signals,
                               self._detection, self._noiseThres,
                               self._saturationThres, self._boxCox)
            self._PODkriging.setVerbose(self._verbose)
            self._PODkriging.setSimulationSize(self._simulationSize)
            self._PODkriging.setSamplingSize(self._samplingSize)
            self._PODkriging.run()

        
    def getMethodActive(self):
        """
        Accessor to the dictionnary of active methods.

        Returns
        -------
        activeDict : dict
            The dictionnary containing the bool telling if the methods is
            activated or not.
        """
        return self._activeMethods

    def setMethodActive(self, method, activation):
        """
        Accessor to the dictionnary of active methods.

        Parameters
        ----------
        method : string
            The key of the method to activate or deactivate.
        activation : bool
            Set to True to activate and False to deactivate.
        """
        if not self._activeMethods.has_key(method):
            raise NameError(method + ' is not an admissible keys.')
        if type(activation) is not bool:
            raise ValueError('The given activation parameter is not a boolean.')
        else:
            self._activeMethods[method] = activation

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

    def getSamplingSize(self):
        """
        Accessor to the Monte Carlo sampling size.

        Returns
        -------
        size : int
            The size of the Monte Carlo simulation used to compute the POD for
            each defect size for polynomial chaos and kriging.
        """
        return self._samplingSize

    def getLinearGaussPOD(self):
        """
        Accessor to the linear model POD object with Gaussian hypothesis.

        Returns
        -------
        algorithm : UnivariateLinearModelPOD
            The UnivariateLinearModelPOD object that is used to compute the POD.
        """
        if not self._activeMethods['LinearGauss']:
            raise Exception('LinearGauss is not activated.')
        else:
            if self._PODgauss is None:
                raise Exception('The run method must be lauched.')
            else:
                return self._PODgauss

    def getLinearBinomialPOD(self):
        """
        Accessor to the linear model POD object with no hypothesis on the residuals.

        Returns
        -------
        algorithm : UnivariateLinearModelPOD
            The UnivariateLinearModelPOD object that is used to compute the POD.
        """
        if not self._activeMethods['LinearBinomial']:
            raise Exception('LinearBinomial is not activated.')
        else:
            if self._PODbin is None:
                raise Exception('The run method must be lauched.')
            else:
                return self._PODbin

    def getLinearKernelSmoothingPOD(self):
        """
        Accessor to the linear model POD object with kernel smoothing on the residuals.

        Returns
        -------
        algorithm : UnivariateLinearModelPOD
            The UnivariateLinearModelPOD object that is used to compute the POD.
        """
        if not self._activeMethods['LinearKernelSmoothing']:
            raise Exception('LinearKernelSmoothing is not activated.')
        else:
            if self._PODks is None:
                raise Exception('The run method must be lauched.')
            else:
                return self._PODks

    def getQuantileRegressionPOD(self):
        """
        Accessor to the quantile regression POD object.

        Returns
        -------
        algorithm : QuantileRegressionPOD
            The QuantileRegressionPOD object that is used to compute the POD.
        """
        if not self._activeMethods['QuantileRegression']:
            raise Exception('QuantileRegression is not activated.')
        else:
            if self._PODqr is None:
                raise Exception('The run method must be lauched.')
            else:
                return self._PODqr

    def getPolynomialChaosPOD(self):
        """
        Accessor to the polynomial chaos POD object.

        Returns
        -------
        algorithm : PolynomialChaosPOD
            The PolynomialChaosPOD object that is used to compute the POD.
        """
        if not self._activeMethods['PolynomialChaos']:
            raise Exception('PolynomialChaos is not activated.')
        else:
            if self._PODchaos is None:
                raise Exception('The run method must be lauched.')
            else:
                return self._PODchaos

    def getKrigingPOD(self):
        """
        Accessor to the kriging POD object.

        Returns
        -------
        algorithm : KrigingPOD
            The KrigingPOD object that is used to compute the POD.
        """
        if not self._activeMethods['Kriging']:
            raise Exception('Kriging is not activated.')
        elif self._dim == 1:
            raise Exception('Kriging cannot be used when input dimension is 1.')
        else:
            if self._PODkriging is None:
                raise Exception('The run method must be lauched.')
            else:
                return self._PODkriging

    def setSamplingSize(self, size):
        """
        Accessor to the Monte Carlo sampling size.

        Parameters
        ----------
        size : int
            The size of the Monte Carlo simulation used to compute the POD for
            each defect size for polynomial chaos and kriging.
        """
        self._samplingSize = size

    def printResults(self, probabilityLevel=0.9, confidenceLevel=0.95):
        """
        Print all results in the terminal.

        Parameters
        ----------
        probabilityLevel : float
            The probability level for which the defect size is computed.
            default is 0.9.
        confidenceLevel : float
            The confidence level associated to the given probability level the
            defect size is computed. Default is 0.95.

        Notes
        -----
        The probability level and confidence level can be specified in order to
        display the defect size for different probability level.
        """

        # build list of results to be displayed
        self._buildPrintResults(probabilityLevel, confidenceLevel)

        validationResult = '\n'.join(['{:<42} {:>10} {:>10} {:>10}'.format(*line) \
                            for line in self._dataValidation])

        PODResult = '\n'.join(['{:<47} {:>13} {:>13}'.format(*line) for line in self._dataPOD])

        self._analysis.printResults()
        ndash = 80
        print ''
        print '-' * ndash
        print '         Model validation results'
        print '-' * ndash
        print validationResult
        print '-' * ndash
        print ''
        print '-' * ndash
        print '         POD results'
        print '-' * ndash
        print PODResult
        print '-' * ndash
        print ''

        if self._censored:
            msg = 'For '
            if self._activeMethods['QuantileRegression']:
                msg = msg + 'quantile regression, '
            if self._activeMethods['PolynomialChaos']:
                msg = msg + 'polynomial chaos, '
            if self._dim > 1 and self._activeMethods['Kriging']:
                msg = msg + 'kriging, '
            msg = msg + 'results are given for filtered data.'
            logging.info(msg)


    def saveResults(self, name, probabilityLevel=0.9, confidenceLevel=0.95):
        """
        Save all analysis test results in a file.

        Parameters
        ----------
        name : string
            Name of the file or full path name.
        probabilityLevel : float
            The probability level for which the defect size is computed.
            default is 0.9.
        confidenceLevel : float
            The confidence level associated to the given probability level the
            defect size is computed. Default is 0.95.

        Notes
        -----
        The probability level and confidence level can be specified in order to
        display the defect size for different probability level.

        The file can be saved as a csv file. Separations are made with tabulations.

        If *name* is the file name, then it is saved in the current working
        directory.
        """

        self._buildPrintResults(probabilityLevel, confidenceLevel)

        # attention, i asks for a private attribute of the analysis class.
        regressionResult = '\n'.join(['{}\t{}\t{}'.format(*line) for
                                line in self._analysis._dataRegression])

        residualsResult = '\n'.join(['{}\t{}\t{}'.format(*line) for
                                line in self._analysis._dataResiduals])

        validationResult = '\n'.join(['{}\t{}\t{}\t{}'.format(*line) for
                                line in self._dataValidation])

        podResult = '\n'.join(['{}\t{}\t{}'.format(*line) for
                                line in self._dataPOD])

        with open(name, 'w') as fd:
            fd.write('Linear model analysis results\n\n')
            fd.write(regressionResult)
            fd.write('\n\nResiduals analysis results\n\n')
            fd.write(residualsResult)
            fd.write('Model validation results\n\n')
            fd.write(validationResult)
            fd.write('\n\nPOD results\n\n')
            fd.write(podResult)

    def _buildPrintResults(self, probabilityLevel, confidenceLevel):

        # number of digits to be displayed
        n_digits = 2

        self._dataValidation = \
           [["", "", "Uncensored", "Censored"],
            ["", "", "", ""],
            ["", "R2", "Q2", "R2"],
            ["", "", "", ""]]

        self._dataPOD = [["", 'a'+str(int(probabilityLevel*100)), 
                          'a'+str(int(probabilityLevel*100))+'/'\
                                    +str(int(confidenceLevel*100))]]

        if self._activeMethods['LinearGauss'] or self._activeMethods['LinearBinomial'] \
            or self._activeMethods['LinearKernelSmoothing']:
            # validation result
            R2Unc = round(self._analysis.getR2()[0], n_digits)
            if self._censored:
                R2Cen = round(self._analysis.getR2()[1], n_digits)
            else:
                R2Cen = ""
            linearValid = ["Linear Regression (> 0.8):", R2Unc, "", R2Cen]
            self._dataValidation.append(linearValid)

            # POD results
            self._dataPOD.append(["Linear Regression", "", ""])
            if self._activeMethods['LinearGauss']:
                try:
                    detectSize = self._PODgauss.computeDetectionSize(probabilityLevel, confidenceLevel)
                except:
                    detectSize = [-1, -1]
                    logging.warn('Detection size for linear model with Gaussian '+\
                                 'residuals cannot be computed.')
                self._dataPOD.append(["Gaussian residuals :", round(detectSize[0], n_digits),
                                                round(detectSize[1], n_digits)])

            if self._activeMethods['LinearBinomial']:
                try:
                    detectSize = self._PODbin.computeDetectionSize(probabilityLevel, confidenceLevel)
                except:
                    detectSize = [-1, -1]
                    logging.warn('Detection size for linear model with no '+\
                                 'hypothesis on the residuals cannot be computed.')
                self._dataPOD.append(["No residuals hypothesis :", round(detectSize[0], n_digits),
                                                round(detectSize[1], n_digits)])

            if self._activeMethods['LinearKernelSmoothing']:
                try:
                    detectSize = self._PODks.computeDetectionSize(probabilityLevel, confidenceLevel)
                except:
                    detectSize = [-1, -1]
                    logging.warn('Detection size for linear model with kernel '+\
                                 'smoothing on the residuals cannot be computed.')
                self._dataPOD.append(["Kernel smoothing on residuals :", round(detectSize[0], n_digits),
                                                round(detectSize[1], n_digits)])

            self._dataPOD.append(["", "", ""])

        if self._activeMethods['QuantileRegression']:
            # validation result
            quantRegValid = ["Quantile Regression at level "+str(round(probabilityLevel,2))+\
                             " (> 0.6):", round(self._PODqr.getR2(probabilityLevel),
                             n_digits), "", ""]
            self._dataValidation.append(quantRegValid)

            # POD results
            try:
                detectSize = self._PODqr.computeDetectionSize(probabilityLevel, confidenceLevel)
            except:
                detectSize = [-1, -1]
                logging.warn('Detection size for quantile regression cannot be computed.')
            self._dataPOD.append(["Quantile Regression :", round(detectSize[0], n_digits),
                                            round(detectSize[1], n_digits)])

        if self._activeMethods['PolynomialChaos']:
            # validation result
            chaosValid = ["Polynomial Chaos (> 0.8):", round(self._PODchaos.getR2(),
                n_digits), round(self._PODchaos.getQ2(), n_digits), ""]
            self._dataValidation.append(chaosValid)

            # POD results
            try:
                detectSize = self._PODchaos.computeDetectionSize(probabilityLevel, confidenceLevel)
            except:
                detectSize = [-1, -1]
                logging.warn('Detection size for polynomial chaos cannot be computed.')
            self._dataPOD.append(["Polynomial chaos :", round(detectSize[0], n_digits),
                                                 round(detectSize[1], n_digits)])

        if self._dim >1 and self._activeMethods['Kriging']:
            # validation result
            krigingValid = ["Kriging (> 0.8):", "", round(self._PODkriging.getQ2(),
                                                                n_digits), ""]
            self._dataValidation.append(krigingValid)

            # POD results
            try:
                detectSize = self._PODkriging.computeDetectionSize(probabilityLevel, confidenceLevel)
            except:
                detectSize = [-1, -1]
                logging.warn('Detection size for kriging cannot be computed.')
            self._dataPOD.append(["Kriging :", round(detectSize[0], n_digits),
                                                 round(detectSize[1], n_digits)])

    def drawGraphs(self, directory=None, extension='png', probabilityLevel=None,
                   confidenceLevel=None):
        """
        draw and save all possible graphs

        Parameters
        ----------
        directory : string
            Directory where to save the graphs. Default is the working directory.
        extension : string
            File extension of the graphs. Default is 'png'.
        probabilityLevel : float
            The probability level for which the defect size is computed.
            default is None.
        confidenceLevel : float
            The confidence level associated to the given probability level the
            defect size is computed. Default is None.
        """

        if directory is None:
            directory = os.getcwd()

        fig = []
        
        if self._boxCox:
            f, ax = self._analysis.drawBoxCoxLikelihood(os.path.join(directory,
                                            'BoxCox_likelihood.') + extension)
            fig.append(f)

        f, ax = self._analysis.drawLinearModel(model='uncensored', name=os.path.join(directory,
                                                'Linear_model.') + extension)
        fig.append(f)
        f, ax = self._analysis.drawResiduals(model='uncensored', name=os.path.join(directory,
                                                    'Residuals.') + extension)
        fig.append(f)
        f, ax = self._analysis.drawResidualsDistribution(model='uncensored', name=os.path.join(directory,
                                         'Residuals_distribution.') + extension)
        fig.append(f)
        f, ax = self._analysis.drawResidualsQQplot(model='uncensored', name=os.path.join(directory,
                                            'Residuals_QQ_plot.') + extension)
        fig.append(f)

        if self._censored:
            f, ax = self._analysis.drawLinearModel(model='censored', name=os.path.join(directory,
                                'Linear_model_censored.') + extension)
            fig.append(f)
            f, ax = self._analysis.drawResiduals(model='censored', name=os.path.join(directory,
                                'Residuals_censored.') + extension)
            fig.append(f)
            f, ax = self._analysis.drawResidualsDistribution(model='censored', name=os.path.join(directory,
                                'Residuals_distribution_censored.') + extension)
            fig.append(f)
            f, ax = self._analysis.drawResidualsQQplot(model='censored', name=os.path.join(directory,
                                'Residuals_QQ_plot_censored.') + extension)
            fig.append(f)


        if self._activeMethods['LinearGauss']:
            try:
                f, ax = self._PODgauss.drawPOD(probabilityLevel, confidenceLevel,
                    name=os.path.join(directory,'POD_Gauss.') + extension)
                fig.append(f)
            except:
                logging.warn('POD for linear model with Gaussian residuals '+\
                             'cannot be drawn for the given parameters.')
        if self._activeMethods['LinearBinomial']:
            try:
                f, ax = self._PODbin.drawPOD(probabilityLevel, confidenceLevel,
                    name=os.path.join(directory,'POD_Binomial.') + extension)
                fig.append(f)
            except:
                logging.warn('POD for linear model with no hypothesis on the '+\
                             'residuals cannot be drawn for the given parameters.')
        if self._activeMethods['LinearKernelSmoothing']:
            try:
                f, ax = self._PODks.drawPOD(probabilityLevel, confidenceLevel,
                    name=os.path.join(directory,'POD_Kernel_Smoothing.') + extension)
                fig.append(f)
            except:
                logging.warn('POD for linear model with kernel smoothing on the '+\
                             'residuals cannot be drawn for the given parameters.')
        if self._activeMethods['QuantileRegression']:
            try:
                f, ax = self._PODqr.drawPOD(probabilityLevel, confidenceLevel,
                    name=os.path.join(directory,'POD_Quantile_Regression.') + extension)
                fig.append(f)
            except:
                logging.warn('POD for quantile regression cannot be drawn for the given parameters.')
            f, ax = self._PODqr.drawLinearModel(probabilityLevel, name=os.path.join(directory,
                                                'Quantile_regression_model.') + extension)
            fig.append(f)
        if self._activeMethods['PolynomialChaos']:
            try:
                f, ax = self._PODchaos.drawPOD(probabilityLevel, confidenceLevel,
                    name=os.path.join(directory,'POD_Polynomial_chaos.') + extension)
                fig.append(f)
            except:
                logging.warn('POD for polynomial chaos cannot be drawn for the given parameters.')
            f, ax = self._PODchaos.drawValidationGraph(name=os.path.join(directory,
                                    'Validation_graph_Polynomial_chaos.') + extension)
            fig.append(f)
        if self._dim >1 and self._activeMethods['Kriging']:
            try:
                f, ax = self._PODkriging.drawPOD(probabilityLevel, confidenceLevel,
                    name=os.path.join(directory,'POD_Kriging.') + extension)
                fig.append(f)
            except:
                logging.warn('POD for kriging cannot be drawn for the given parameters.')
            f, ax = self._PODkriging.drawValidationGraph(name=os.path.join(directory,
                                    'Validation_graph_Kriging.') + extension)
            fig.append(f)

        return fig
