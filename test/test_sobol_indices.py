import openturns as ot
ot.TBB.Disable()
import otpod
import numpy as np
from distutils.version import LooseVersion

inputSample = ot.NumericalSample(
    [[4.59626812e+00, 7.46143339e-02, 1.02231538e+00, 8.60042277e+01],
    [4.14315790e+00, 4.20801346e-02, 1.05874908e+00, 2.65757364e+01],
    [4.76735111e+00, 3.72414824e-02, 1.05730385e+00, 5.76058433e+01],
    [4.82811977e+00, 2.49997658e-02, 1.06954641e+00, 2.54461380e+01],
    [4.48961094e+00, 3.74562922e-02, 1.04943946e+00, 6.19483646e+00],
    [5.05605334e+00, 4.87599783e-02, 1.06520409e+00, 3.39024904e+00],
    [5.69679328e+00, 7.74915877e-02, 1.04099514e+00, 6.50990466e+01],
    [5.10193991e+00, 4.35520544e-02, 1.02502536e+00, 5.51492592e+01],
    [4.04791970e+00, 2.38565932e-02, 1.01906882e+00, 2.07875350e+01],
    [4.66238956e+00, 5.49901237e-02, 1.02427200e+00, 1.45661275e+01],
    [4.86634219e+00, 6.04693570e-02, 1.08199374e+00, 1.05104730e+00],
    [4.13519347e+00, 4.45225831e-02, 1.01900124e+00, 5.10117047e+01],
    [4.92541940e+00, 7.87692335e-02, 9.91868726e-01, 8.32302238e+01],
    [4.70722074e+00, 6.51799251e-02, 1.10608515e+00, 3.30181002e+01],
    [4.29040932e+00, 1.75426222e-02, 9.75678838e-01, 2.28186756e+01],
    [4.89291400e+00, 2.34997929e-02, 1.07669835e+00, 5.38926138e+01],
    [4.44653744e+00, 7.63175936e-02, 1.06979154e+00, 5.19109415e+01],
    [3.99977452e+00, 5.80430585e-02, 1.01850716e+00, 7.61988190e+01],
    [3.95491570e+00, 1.09302814e-02, 1.03687664e+00, 6.09981789e+01],
    [5.16424368e+00, 2.69026464e-02, 1.06673711e+00, 2.88708887e+01],
    [5.30491620e+00, 4.53802273e-02, 1.06254792e+00, 3.03856837e+01],
    [4.92809155e+00, 1.20616369e-02, 1.00700410e+00, 7.02512744e+00],
    [4.68373805e+00, 6.26028935e-02, 1.05152117e+00, 4.81271603e+01],
    [5.32381954e+00, 4.33013582e-02, 9.90522007e-01, 6.56015973e+01],
    [4.35455857e+00, 1.23814619e-02, 1.01810539e+00, 1.10769534e+01]])

signals = ot.NumericalSample(
    [[ 37.305445], [ 35.466919], [ 43.187991], [ 45.305165], [ 40.121222], [ 44.609524],
     [ 45.14552 ], [ 44.80595 ], [ 35.414039], [ 39.851778], [ 42.046049], [ 34.73469 ],
     [ 39.339349], [ 40.384559], [ 38.718623], [ 46.189709], [ 36.155737], [ 31.768369],
     [ 35.384313], [ 47.914584], [ 46.758537], [ 46.564428], [ 39.698493], [ 45.636588],
     [ 40.643948]])

detection = 38.


####### Test on the Sobol indices ###################
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
ot.RandomGenerator.SetState(ot.RandomGeneratorState(ot.Indices([0]*768), 0))
POD = otpod.KrigingPOD(inputSample, signals, detection)
if LooseVersion(ot.__version__) <= '1.6':
    covColl = ot.CovarianceModelCollection(4)
    scale = [5.03148,13.9442,20,20]
    for i in range(4):
        c = ot.SquaredExponential(1, scale[i])
        c.setAmplitude([15.1697])
        covColl[i]  = c
    covarianceModel = ot.ProductCovarianceModel(covColl)
else:
    covarianceModel = ot.SquaredExponential([5.03148,13.9442,20,20], [15.1697])
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
# print firstAgg1
# print totalAgg1
# print firstOrder1
# print totalOrder1
def test_1_FA():
    np.testing.assert_almost_equal(firstAgg1, [0.799186,0.0398873,-0.00431286], decimal=2)
def test_1_TA():
    np.testing.assert_almost_equal(totalAgg1, [0.994125,0.214024,0.00581788], decimal=2)
def test_1_FO_5():
    np.testing.assert_almost_equal(firstOrder1, [0.940412,0.00702777,0.000245034], decimal=2)
def test_1_TO_5():
    np.testing.assert_almost_equal(totalOrder1, [1.02143,0.139784,0.000318769], decimal=2)

# Test 2 for one specific defect size
sobol.setDefectSizes([4.5])
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
sobol.run()
sobol_result = sobol.getSensitivityResult()
firstAgg2 = sobol_result.getAggregatedFirstOrderIndices()
totalAgg2 = sobol_result.getAggregatedTotalOrderIndices()
# print firstAgg2
# print totalAgg2
def test_2_FA():
    np.testing.assert_almost_equal(firstAgg2, [0.925856,0.00606244,-0.00565195], decimal=2)
def test_2_TA():
    np.testing.assert_almost_equal(totalAgg2, [1.08168,0.197621,0.00388022], decimal=2)

# Test 3 with Martinez method
sobol.setSensitivityMethod("Martinez")
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
sobol.run()
sobol_result = sobol.getSensitivityResult()
firstAgg3 = sobol_result.getAggregatedFirstOrderIndices()
totalAgg3 = sobol_result.getAggregatedTotalOrderIndices()
# print firstAgg3
# print totalAgg3
def test_3_FA():
    np.testing.assert_almost_equal(firstAgg3, [0.841662,-0.0310924,-0.0428869], decimal=2)
def test_3_TA():
    np.testing.assert_almost_equal(totalAgg3, [1.07519,0.202521,0.00315248], decimal=2)

# Test 4 with Jansen method
sobol.setSensitivityMethod("Jansen")
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
sobol.run()
sobol_result = sobol.getSensitivityResult()
firstAgg4 = sobol_result.getAggregatedFirstOrderIndices()
totalAgg4 = sobol_result.getAggregatedTotalOrderIndices()
# print firstAgg4
# print totalAgg4
def test_4_FA():
    np.testing.assert_almost_equal(firstAgg4, [0.829998,-0.0502771,-0.0548323], decimal=2)
def test_4_TA():
    np.testing.assert_almost_equal(totalAgg4, [1.12427,0.203706,0.00314908], decimal=2)


# Test 5 with Jansen method
sobol.setSensitivityMethod("MauntzKucherenko")
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
sobol.run()
sobol_result = sobol.getSensitivityResult()
firstAgg5 = sobol_result.getAggregatedFirstOrderIndices()
totalAgg5 = sobol_result.getAggregatedTotalOrderIndices()
# print firstAgg5
# print totalAgg5
def test_5_FA():
    np.testing.assert_almost_equal(firstAgg5, [0.925982,0.00618882,-0.00552557], decimal=2)
def test_5_TA():
    np.testing.assert_almost_equal(totalAgg5, [1.08168,0.197624,0.00388355], decimal=2)


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
    np.testing.assert_almost_equal(firstAgg6, [0.691873,0.0394016,9.98788e-05], decimal=2)
def test_6_TA():
    np.testing.assert_almost_equal(totalAgg6, [0.859487,0.24918,0.0108658], decimal=2)
