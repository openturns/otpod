import openturns as ot

ot.TBB.Disable()
import otpod
import numpy as np

inputSample = ot.Sample(
    [
        [4.59626812e00, 7.46143339e-02, 1.02231538e00, 8.60042277e01],
        [4.14315790e00, 4.20801346e-02, 1.05874908e00, 2.65757364e01],
        [4.76735111e00, 3.72414824e-02, 1.05730385e00, 5.76058433e01],
        [4.82811977e00, 2.49997658e-02, 1.06954641e00, 2.54461380e01],
        [4.48961094e00, 3.74562922e-02, 1.04943946e00, 6.19483646e00],
        [5.05605334e00, 4.87599783e-02, 1.06520409e00, 3.39024904e00],
        [5.69679328e00, 7.74915877e-02, 1.04099514e00, 6.50990466e01],
        [5.10193991e00, 4.35520544e-02, 1.02502536e00, 5.51492592e01],
        [4.04791970e00, 2.38565932e-02, 1.01906882e00, 2.07875350e01],
        [4.66238956e00, 5.49901237e-02, 1.02427200e00, 1.45661275e01],
        [4.86634219e00, 6.04693570e-02, 1.08199374e00, 1.05104730e00],
        [4.13519347e00, 4.45225831e-02, 1.01900124e00, 5.10117047e01],
        [4.92541940e00, 7.87692335e-02, 9.91868726e-01, 8.32302238e01],
        [4.70722074e00, 6.51799251e-02, 1.10608515e00, 3.30181002e01],
        [4.29040932e00, 1.75426222e-02, 9.75678838e-01, 2.28186756e01],
        [4.89291400e00, 2.34997929e-02, 1.07669835e00, 5.38926138e01],
        [4.44653744e00, 7.63175936e-02, 1.06979154e00, 5.19109415e01],
        [3.99977452e00, 5.80430585e-02, 1.01850716e00, 7.61988190e01],
        [3.95491570e00, 1.09302814e-02, 1.03687664e00, 6.09981789e01],
        [5.16424368e00, 2.69026464e-02, 1.06673711e00, 2.88708887e01],
        [5.30491620e00, 4.53802273e-02, 1.06254792e00, 3.03856837e01],
        [4.92809155e00, 1.20616369e-02, 1.00700410e00, 7.02512744e00],
        [4.68373805e00, 6.26028935e-02, 1.05152117e00, 4.81271603e01],
        [5.32381954e00, 4.33013582e-02, 9.90522007e-01, 6.56015973e01],
        [4.35455857e00, 1.23814619e-02, 1.01810539e00, 1.10769534e01],
    ]
)

signals = ot.Sample(
    [
        [37.305445],
        [35.466919],
        [43.187991],
        [45.305165],
        [40.121222],
        [44.609524],
        [45.14552],
        [44.80595],
        [35.414039],
        [39.851778],
        [42.046049],
        [34.73469],
        [39.339349],
        [40.384559],
        [38.718623],
        [46.189709],
        [36.155737],
        [31.768369],
        [35.384313],
        [47.914584],
        [46.758537],
        [46.564428],
        [39.698493],
        [45.636588],
        [40.643948],
    ]
)

detection = 38.0


####### Test on the Sobol indices ###################
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
ot.RandomGenerator.SetState(ot.RandomGeneratorState(ot.Indices([0] * 768), 0))
POD = otpod.KrigingPOD(inputSample, signals, detection)
covarianceModel = ot.SquaredExponential([5.03148, 13.9442, 20, 20], [15.1697])
POD.setCovarianceModel(covarianceModel)
POD.setInitialStartSize(0)
# no need to compute accurate POD
POD.setSamplingSize(100)
POD.setSimulationSize(50)
POD.run()

N = 300
# Test for default defect size
sobol = otpod.SobolIndices(POD, N)
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
sobol.run()
sobol_result = sobol.getSensitivityResult()
firstAgg1 = sobol_result.getAggregatedFirstOrderIndices()
totalAgg1 = sobol_result.getAggregatedTotalOrderIndices()
firstOrder1 = sobol_result.getFirstOrderIndices(5)
totalOrder1 = sobol_result.getTotalOrderIndices(5)


def test_1_FA():
    np.testing.assert_almost_equal(firstAgg1, [0.04, 0.04, 1.32], decimal=1)


def test_1_TA():
    np.testing.assert_almost_equal(
        totalAgg1, [6.03e-05, -4.81e-05, 9.75e-01], decimal=2
    )


def test_1_FO_5():
    np.testing.assert_almost_equal(firstOrder1, [0.04, 0.04, 1.3], decimal=2)


def test_1_TO_5():
    np.testing.assert_almost_equal(
        totalOrder1, [6.57e-05, -4.92e-05, 9.75e-01], decimal=2
    )


# Test 2 for one specific defect size
sobol.setDefectSizes([4.5])
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
sobol.run()
sobol_result = sobol.getSensitivityResult()
firstAgg2 = sobol_result.getAggregatedFirstOrderIndices()
totalAgg2 = sobol_result.getAggregatedTotalOrderIndices()


def test_2_FA():
    np.testing.assert_almost_equal(firstAgg2, [0.04, 0.04, 1.32], decimal=2)


def test_2_TA():
    np.testing.assert_almost_equal(
        totalAgg2, [6.38e-05, -4.86e-05, 9.75e-01], decimal=2
    )


# Test 3 with Martinez method
sobol.setSensitivityMethod("Martinez")
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
sobol.run()
sobol_result = sobol.getSensitivityResult()
firstAgg3 = sobol_result.getAggregatedFirstOrderIndices()
totalAgg3 = sobol_result.getAggregatedTotalOrderIndices()


def test_3_FA():
    np.testing.assert_almost_equal(firstAgg3, [0.04, 0.04, 1.0], decimal=2)


def test_3_TA():
    np.testing.assert_almost_equal(totalAgg3, [9.54e-06, 1.06e-06, 9.63e-01], decimal=2)


# Test 4 with Jansen method
sobol.setSensitivityMethod("Jansen")
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
sobol.run()
sobol_result = sobol.getSensitivityResult()
firstAgg4 = sobol_result.getAggregatedFirstOrderIndices()
totalAgg4 = sobol_result.getAggregatedTotalOrderIndices()


def test_4_FA():
    np.testing.assert_almost_equal(firstAgg4, [-0.13, -0.13, 1.0], decimal=2)


def test_4_TA():
    np.testing.assert_almost_equal(totalAgg4, [9.54e-06, 1.08e-06, 1.13e00], decimal=2)


# Test 5 with Jansen method
sobol.setSensitivityMethod("MauntzKucherenko")
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
sobol.run()
sobol_result = sobol.getSensitivityResult()
firstAgg5 = sobol_result.getAggregatedFirstOrderIndices()
totalAgg5 = sobol_result.getAggregatedTotalOrderIndices()


def test_5_FA():
    np.testing.assert_almost_equal(firstAgg5, [2.58e-04, -7.04e-05, 1.28e00], decimal=2)


def test_5_TA():
    np.testing.assert_almost_equal(
        totalAgg5, [8.59e-05, -2.66e-05, 9.75e-01], decimal=2
    )


################################################################################
# Test 6 WITH CHAOS
chaosPOD = otpod.PolynomialChaosPOD(inputSample, signals, detection)
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
chaosPOD.setSamplingSize(200)
chaosPOD.setSimulationSize(50)
chaosPOD.run()
N = 100
# Test for default defect size
sobolChaos = otpod.SobolIndices(chaosPOD, N)
sobolChaos.setSimulationSize(100)
ot.RandomGenerator.SetSeed(0)
sobolChaos.run()
sobol_result = sobolChaos.getSensitivityResult()
firstAgg6 = sobol_result.getAggregatedFirstOrderIndices()
totalAgg6 = sobol_result.getAggregatedTotalOrderIndices()


def test_6_FA():
    np.testing.assert_almost_equal(
        firstAgg6, [0.0994924, -0.0318261, -0.111692], decimal=2
    )


def test_6_TA():
    np.testing.assert_almost_equal(totalAgg6, [1.03563, 0.969766, 0.902102], decimal=2)
