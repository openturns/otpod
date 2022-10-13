# -*- Python -*-

__all__ = ["AdaptiveHitMissPOD"]

import os
import openturns as ot
import numpy as np
from ._pod import POD
from scipy.interpolate import interp1d
from ._progress_bar import updateProgress
from ._decorator import DocInherit, keepingArgs
import logging
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix

try:
    from pkg_resources import parse_version
except ImportError:
    from distutils.version import LooseVersion as parse_version


class AdaptiveHitMissPOD(POD):
    """
    Adaptive algorithm for hit miss data type.

    **Available constructor:**

    AdaptiveHitMissPOD(*inputDOE, outputDOE, physicalModel, nMorePoints,
    detection, noiseThres, saturationThres*)

    Parameters
    ----------
    inputDOE : 2-d sequence of float
        Vector of the input values. The first column must correspond with the
        defect sizes.
    outputDOE : 2-d sequence of float
        Vector of the signals, of dimension 1.
    physicalModel : :py:class:`openturns.Function`
        True model used to compute the real hit miss value of the signal value
        to be added to the DOE.
    nMorePoints : positive int
        The number of points to add to the DOE, computed by the *physicalModel*.
    detection : float
        Detection value of the signal if the physical model does not return a
        hit miss value.
    noiseThres : float
        Value for low censored data. Default is None.
    saturationThres : float
        Value for high censored data. Default is None

    Warnings
    --------
    The first column of the input sample must corresponds with the defect sizes.

    Notes
    -----
    This class aims at building the POD based on a classifier model where the
    design of experiments is iteratively enriched. The initial design of
    experiments is given as input parameters. The enrichment criterion is based
    on the misclassification empirical risk. The criterion is computed on several
    candidate points. The sample of candidate points is created using
    a low discrepancy sequence (Sobol') if the input distribution has an
    independant copula, otherwise a Monte Carlo experiment is used. The stopping
    criterion is only based on the number of points that must be added to the
    design of experiments.

    The classifier algorithms availables are the SVC and the random forests. The
    choice of the algorithm can be defined using *setClassifierType*. The default
    algorithm is the random forests.

    The physical model can return either the hit miss value (0 or 1) or the signal
    value. In this case, the detection value must be given and the physical
    model is transformed in order to provide a hit miss value.

    The POD are computed by a Monte Carlo simulation for several defect values.
    The accuracy of the Monte Carlo simulation is taken into account using the TCL.
    The return POD model corresponds with an interpolate function built
    with the POD values computed for the given defect sizes. The default values
    are 20 defect sizes between the minimum and maximum value of the defect sample.
    The defect sizes can be changed using the method *setDefectSizes*.

    A progress bar is shown if the verbosity is enabled. It can be disabled using
    the method *setVerbose*.
    """

    def __init__(
        self,
        inputDOE,
        outputDOE,
        physicalModel=None,
        nMorePoints=0,
        detection=None,
        noiseThres=None,
        saturationThres=None,
    ):

        # initialize the POD class
        boxCox = False
        super(AdaptiveHitMissPOD, self).__init__(
            inputDOE, outputDOE, detection, noiseThres, saturationThres, boxCox
        )
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

        self._distribution = None
        self._classifierType = "rf"  # random forest classifier or svc
        self._ClassifierParameters = [[100], [None], [2], [0]]
        self._classifierModel = None
        self._confMat = None
        self._pmax = 0.52
        self._pmin = 0.45
        self._initialStartSize = 1000
        self._samplingSize = 10000  # Number of MC simulations to compute POD
        self._candidateSize = 5000
        self._nMorePoints = nMorePoints
        self._verbose = True
        self._graph = False  # flag to print or not the POD curves at each iteration
        self._probabilityLevel = None  # default graph option
        self._confidenceLevel = None  # default graph option
        self._graphDirectory = None  # graph directory for saving

        self._normalDist = ot.Normal()

        if self._censored:
            logging.info(
                "Censored data are not taken into account : the "
                + "kriging model is only built on filtered data."
            )

        # Run the preliminary run of the POD class
        result = self._run(
            self._inputSample,
            self._outputSample,
            self._detection,
            self._noiseThres,
            self._saturationThres,
            self._boxCox,
            self._censored,
        )

        # get some results
        self._input = result["inputSample"]
        self._signals = result["signals"]
        self._detectionBoxCox = result["detectionBoxCox"]
        self._boxCoxTransform = result["boxCoxTransform"]
        self._shift = result["shift"]

        # define the defect sizes for the interpolation function if not defined
        self._defectNumber = 20
        self._defectSizes = np.linspace(
            self._input[:, 0].getMin()[0],
            self._input[:, 0].getMax()[0],
            self._defectNumber,
        )

        if detection is None:
            # case where the physical model already returns 0 or 1
            self._physicalModel = physicalModel
        else:
            # case where the physical model returns a true signal value
            # the physical model is turned into a binary model with respect
            # to the detection value.
            if parse_version(ot.__version__) < parse_version("1.18"):
                self._physicalModel = ot.IndicatorFunction(
                    physicalModel, ot.Greater(), self._detection
                )
            else:
                self._physicalModel = ot.IndicatorFunction(
                    ot.LevelSet(physicalModel, ot.Greater(), self._detection)
                )
            self._signals = np.array(
                np.array(self._signals) > self._detection, dtype="int"
            )

    def run(self):
        """
        Launch the algorithm and build the POD models.

        Notes
        -----
        This method launches the iterative algorithm. Once the algorithm stops,
        it builds the POD models : Monte Carlo simulation are performed for each
        defect sizes with the final classifier model. Eventually, the sample is
        used to compute the mean POD and the POD at the confidence level.
        """

        # Create an initial uniform distribution if not given
        if self._distribution is None:
            inputMin = self._input.getMin()
            inputMin[0] = np.min(self._defectSizes)
            inputMax = self._input.getMax()
            inputMax[0] = np.max(self._defectSizes)
            marginals = [ot.Uniform(inputMin[i], inputMax[i]) for i in range(self._dim)]
            self._distribution = ot.ComposedDistribution(marginals)

        # Create the design of experiments of the candidate points where the
        # criterion is computed
        if self._distribution.hasIndependentCopula():
            # without copula use low discrepancy experiment as first doe
            doeCandidate = ot.LowDiscrepancyExperiment(
                ot.SobolSequence(), self._distribution, self._candidateSize
            ).generate()
        else:
            # else simple Monte Carlo distribution on Uniform distribution
            doeCandidate = self._distribution.getSample(self._candidateSize)

        doeCandidate = np.array(doeCandidate)
        # build initial classifier model
        # build the kriging model without optimization

        if self._verbose:
            print("Building the classifier")

        n_ini = int(self._input.getSize())
        self._input = np.array(self._input)
        self._signals = np.hstack(self._signals)

        n_added_points = 0
        algo_iteration = 0

        ## Cas de la classif par svc
        if self._classifierType == "svc":
            algo_temp = list(
                map(
                    lambda C, kernel, degree, probability: svm.SVC(
                        C=C,
                        kernel=kernel,
                        degree=degree,
                        gamma="auto",
                        probability=probability,
                        coef0=1,
                    ),
                    *self._ClassifierParameters
                )
            )[0]

        ## Cas de la classif par fro
        if self._classifierType == "rf":
            algo_temp = list(
                map(
                    lambda n_estimators, max_depth, min_samples_split, random_state: ExtraTreesClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=random_state,
                    ),
                    *self._ClassifierParameters
                )
            )[0]

        algo_temp.fit(self._input, self._signals)

        list_classifiers = []
        f_iter = algo_temp.predict_proba
        list_classifiers.append(f_iter)
        self._classifierModel = f_iter

        plt.ion()
        # Start the improvment loop
        if self._verbose and self._nMorePoints > 0:
            print("Start the improvement loop")

        while n_added_points < self._nMorePoints:

            # calcul de ce qu il y a dans l' exp de la proba
            probs = f_iter(doeCandidate)[:, 1]

            # recuperation des indices ou la p p_min < proba(x) < p_max
            ind_p1 = np.where(probs < self._pmax)[0]
            ind_p2 = np.where(probs >= self._pmin)[0]
            ind_p = np.intersect1d(ind_p2, ind_p1)
            ind = ind_p

            # s'il n'a pas d indices on elargit p_min = 0.45, p_max=0.55
            if len(ind) == 0:
                ind_p1 = np.where(probs < 0.1)[0]
                ind_p2 = np.where(probs >= 0.8)[0]
                ind_p = np.intersect1d(ind_p2, ind_p1)
                ind = ind_p

            ind_rank = np.argsort(probs[ind])
            quant = [
                0,
                int(len(ind) / 4.0),
                int(len(ind) / 2.0),
                int(3.0 * len(ind) / 4.0),
                len(ind) - 1,
            ]

            ind_bis = ind_rank[quant]
            x_new = doeCandidate[ind[ind_bis], :]
            z_new = np.hstack(self._physicalModel(x_new))

            n_new_temp = len(self._input) + len(x_new)

            # si on depasse le nombre de points, on s arrete
            if n_new_temp > (n_ini + self._nMorePoints):
                x_new = x_new[: self._nMorePoints + n_ini - len(self._input), :]
                z_new = z_new[: self._nMorePoints + n_ini - len(self._input)]

            self._input = np.vstack((self._input, x_new))
            self._signals = np.hstack((self._signals, z_new))

            n_added_points = n_new_temp - n_ini
            algo_iteration = algo_iteration + 1

            if self._classifierType == "svc":
                algo_temp = list(
                    map(
                        lambda C, kernel, degree, probability: svm.SVC(
                            C=C,
                            kernel=kernel,
                            degree=degree,
                            gamma="auto",
                            probability=probability,
                            coef0=1,
                        ),
                        *self._ClassifierParameters
                    )
                )[0]

            if self._classifierType == "rf":
                algo_temp = list(
                    map(
                        lambda n_estimators, max_depth, min_samples_split, random_state: ExtraTreesClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            random_state=random_state,
                        ),
                        *self._ClassifierParameters
                    )
                )[0]

            # Apprentissage avec self._input,self._signals
            algo_temp.fit(self._input, self._signals)

            self._confMat = np.zeros((2, 2))
            for classifier in list_classifiers:
                conf_temp = 1.0 * confusion_matrix(
                    self._signals, classifier(self._input)[:, 1] >= 0.5
                )
                conf_temp = 1.0 * conf_temp / conf_temp.sum(axis=0)
                self._confMat = conf_temp + self._confMat

            self._confMat = 1.0 * self._confMat / len(list_classifiers)
            classif_algo_temp = algo_temp.predict_proba

            p11 = self._confMat[1, 1]
            p10 = self._confMat[1, 0]

            def agg_classifier(x_in):
                c = p11 - p10
                p1_bayes = 1.0 / c * (classif_algo_temp(x_in)[:, 1] - p10)
                p1_bayes = np.vstack(
                    np.min(
                        np.array(
                            [
                                np.max(
                                    np.array([p1_bayes, np.zeros(len(p1_bayes))]),
                                    axis=0,
                                ),
                                np.ones(len(p1_bayes)),
                            ]
                        ),
                        axis=0,
                    )
                )
                return (np.array([1 - p1_bayes, p1_bayes]).T)[0]

            f_iter = agg_classifier
            list_classifiers.append(f_iter)
            self._classifierModel = f_iter

            if self._verbose:
                updateProgress(n_added_points - 1, self._nMorePoints, "Adding points")

            if self._graph:
                self._PODPerDefect = self._computePOD(self._defectSizes, agg_classifier)
                # create the interpolate function of the POD model
                meanPOD = self._PODPerDefect.computeMean()
                interpModel = interp1d(
                    self._defectSizes, np.array(meanPOD), kind="linear"
                )
                self._PODmodel = ot.PythonFunction(1, 1, interpModel)
                # The POD at confidence level is built in getPODCLModel() directly
                fig, ax = self.drawPOD(self._probabilityLevel, self._confidenceLevel)
                plt.draw()
                plt.pause(0.001)
                plt.show()
                if self._graphDirectory is not None:
                    fig.savefig(
                        os.path.join(self._graphDirectory, "AdaptiveHitMissPOD_")
                        + str(algo_iteration),
                        bbox_inches="tight",
                        transparent=True,
                    )

        self._input = ot.Sample(self._input)
        self._signals = ot.Sample(np.vstack(self._signals))
        # Compute the sample predicted for each defect sizes
        self._PODPerDefect = self._computePOD(self._defectSizes, self._classifierModel)
        # compute the POD for all defect sizes
        meanPOD = self._PODPerDefect.computeMean()
        # create the interpolate function of the POD model
        interpModel = interp1d(self._defectSizes, np.array(meanPOD), kind="linear")
        self._PODmodel = ot.PythonFunction(1, 1, interpModel)

        # The POD at confidence level is built in getPODCLModel() directly

        # remove the interactive plotting
        plt.ioff()

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
            1.0 - confidenceLevel
        )
        interpModel = interp1d(self._defectSizes, PODQuantile, kind="linear")
        PODmodelCl = ot.PythonFunction(1, 1, interpModel)

        return PODmodelCl

    @DocInherit  # decorator to inherit the docstring from POD class
    @keepingArgs  # decorator to keep the real signature
    def computeDetectionSize(self, probabilityLevel, confidenceLevel=None):
        if confidenceLevel is None:
            return self._computeDetectionSize(
                self.getPODModel(),
                None,
                probabilityLevel,
                confidenceLevel,
                np.min(self._defectSizes),
                np.max(self._defectSizes),
            )
        elif confidenceLevel is not None:
            return self._computeDetectionSize(
                self.getPODModel(),
                self.getPODCLModel(confidenceLevel),
                probabilityLevel,
                confidenceLevel,
                np.min(self._defectSizes),
                np.max(self._defectSizes),
            )

    @DocInherit  # decorator to inherit the docstring from POD class
    @keepingArgs  # decorator to keep the real signature
    def drawPOD(
        self,
        probabilityLevel=None,
        confidenceLevel=None,
        defectMin=None,
        defectMax=None,
        nbPt=100,
        name=None,
    ):

        if defectMin is None:
            defectMin = np.min(self._defectSizes)
        else:
            if defectMin < np.min(self._defectSizes):
                raise ValueError(
                    "DefectMin must be greater than the minimum "
                    + "of the given defect sizes."
                )
            if defectMin > np.max(self._defectSizes):
                raise ValueError(
                    "DefectMin must be lower than the maximum "
                    + "of the given defect sizes."
                )
        if defectMax is None:
            defectMax = np.max(self._defectSizes)
        else:
            if defectMax > np.max(self._defectSizes):
                raise ValueError(
                    "DefectMax must be lower than the maximum "
                    + "of the given defect sizes."
                )
            if defectMax < np.min(self._defectSizes):
                raise ValueError(
                    "DefectMax must be greater than the minimum "
                    + "of the given defect sizes."
                )

        if confidenceLevel is None:
            fig, ax = self._drawPOD(
                self.getPODModel(),
                None,
                probabilityLevel,
                confidenceLevel,
                defectMin,
                defectMax,
                nbPt,
                name,
            )
        elif confidenceLevel is not None:
            fig, ax = self._drawPOD(
                self.getPODModel(),
                self.getPODCLModel(confidenceLevel),
                probabilityLevel,
                confidenceLevel,
                defectMin,
                defectMax,
                nbPt,
                name,
            )

        ax.set_title("POD - Classifier model")
        if name is not None:
            fig.savefig(name, bbox_inches="tight", transparent=True)

        return fig, ax

    def getConfusionMatrix(self):
        """
        Accessor to the confusion matrix.
        """
        if self._confMat is None:
            print("The run method must be launched first.")
        else:
            return self._confMat

    def getPMin(self):
        """
        Accessor to the lower probability bound for the point selections.
        """
        return self._pmin

    def setPMin(self, pmin):
        """
        Accessor to the lower probability bound for the point selections.
        """
        if pmin < 0 or pmin > 1:
            raise ValueError("pmin must range between 0 and 1.")
        else:
            self._pmin = pmin

    def getPMax(self):
        """
        Accessor to the upper probability bound for the point selections.
        """
        return self._pmax

    def setPMax(self, pmax):
        """
        Accessor to the upper probability bound for the point selections.
        """
        if pmax < 0 or pmax > 1:
            raise ValueError("pmax must range between 0 and 1.")
        else:
            self._pmax = pmax

    def getClassifierType(self):
        """
        Accessor to the classifier type.
        """
        return self._classifierType

    def setClassifierType(self, classifier):
        """
        Accessor to the classifier type.
        """
        if classifier != "rf" and classifier != "svc":
            raise ValueError("Classifier must be 'rf or 'svc'.")
        else:
            self._classifierType = classifier
            if classifier == "svc":
                self._ClassifierParameters = [[1.0], ["poly"], [3], [True]]
            if classifier == "rf":
                self._ClassifierParameters = [[100], [None], [2], [0]]

    def getClassifierParameters(self):
        """
        Accessor to the classifier parameters.
        """
        return self._ClassifierParameters

    def setClassifierParameters(self, parameters):
        """
        Accessor to the classifier parameters.
        """
        self._ClassifierParameters = parameters

    def getOutputDOE(self):
        """
        Accessor to the final output values of the DOE.
        """
        if self._boxCox:
            invBoxCox = self._boxCoxTransform.getInverse()
            return invBoxCox(self._signals) - self._shift
        else:
            return self._signals

    def getInputDOE(self):
        """
        Accessor to the final input values of the DOE.
        """
        return self._input

    def getCandidateSize(self):
        """
        Accessor to the number of candidate points.

        Returns
        -------
        size : int
            The number of candidate points on which the criterion is computed.
        """
        return self._candidateSize

    def setCandidateSize(self, size):
        """
        Accessor to the number of candidate points.

        Parameters
        ----------
        size : int
            The number of candidate points on which the criterion is computed
        """
        self._candidateSize = size

    def getGraphActive(self):
        """
        Accessor to the graph verbosity.

        Returns
        -------
        graphVerbose : bool
            Enable or disable the display of the POD graph at each iteration. Default
            is False.
        """
        return self._graph

    def setGraphActive(
        self, graphVerbose, probabilityLevel=None, confidenceLevel=None, directory=None
    ):
        """
        Accessor to the graph verbosity.

        Parameters
        ----------
        graphVerbose : bool
            Enable or disable the display of the POD graph at each iteration.
        probabilityLevel : float
            The probability level for which the defect size is computed. Default
            is None.
        confidenceLevel : float
            The confidence level associated to the given probability level the
            defect size is computed. Default is None.
        directory : string
            Directory where to save the graphs as png files.
        """
        if type(graphVerbose) is not bool:
            raise TypeError("The parameter 'graphVerbose' is not a bool.")
        elif type(directory) is not str and directory is not None:
            raise TypeError("The parameter 'directory' is not a string.")
        else:
            self._graph = graphVerbose
            self._probabilityLevel = probabilityLevel
            self._confidenceLevel = confidenceLevel
            self._graphDirectory = directory

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
            raise ValueError(
                "Defect sizes must range between "
                + "{:0.4f} ".format(np.ceil(minMin * 10000) / 10000)
                + "and {:0.4f}.".format(np.floor(maxMax * 10000) / 10000)
            )
        self._defectNumber = self._defectSizes.shape[0]

    def setDistribution(self, distribution):
        """
        Accessor to the parameters distribution.

        Parameters
        ----------
        distribution : :py:class:`openturns.ComposedDistribution`
            The input parameters distribution.
        """
        try:
            ot.ComposedDistribution(distribution)
        except NotImplementedError:
            raise Exception("The given parameter is not a ComposedDistribution.")
        self._distribution = distribution

    def getDistribution(self):
        """
        Accessor to the parameters distribution.

        Returns
        -------
        distribution : :py:class:`openturns.ComposedDistribution`
            The input parameters distribution, default is a Uniform distribution
            for all parameters.
        """
        if self._distribution is None:
            print("The run method must be launched first.")
        else:
            return self._distribution

    def getClassifier(self):
        """
        Accessor to the classifier model.

        Returns
        -------
        result : classifier
            The classifier model, either random forest or svm.
        """
        if self._classifierModel is None:
            print("The run method must be launched first.")
        else:
            return self._classifierModel

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
            raise TypeError("The parameter is not a bool.")
        else:
            self._verbose = verbose

    def _mergeDefectInX(self, defect, X):
        """
        defect : scalar of the defect value
        X : sample without the defect column
        """
        size = X.getSize()
        dim = X.getDimension() + 1
        samplePred = ot.Sample(size, dim)
        samplePred[:, 0] = ot.Sample(size, [defect])
        samplePred[:, 1:] = X
        return samplePred

    def _computePOD(self, defectSizes, algoClassifier):
        """
        Compute the POD sample for all defect sizes in a vectorized way.
        """
        # create the input sample that must be computed by the metamodels
        samplePred = self._distribution.getSample(self._samplingSize)[:, 1:]
        fullSamplePred = ot.Sample(self._samplingSize * self._defectNumber, self._dim)
        for i, defect in enumerate(defectSizes):
            fullSamplePred[
                self._samplingSize * i: self._samplingSize * (i + 1), :
            ] = self._mergeDefectInX(defect, samplePred)

        classifierSample = algoClassifier(np.array(fullSamplePred))[:, 1]
        classifierSample = np.reshape(
            classifierSample, (self._samplingSize, self._defectNumber), "F"
        )
        return ot.Sample(classifierSample)
