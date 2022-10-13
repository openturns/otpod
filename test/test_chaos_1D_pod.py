import openturns as ot

ot.TBB.Disable()
import otpod
import numpy as np

beta0 = 2.52
beta1 = 43.28
sigmaEps = 1.95
lamb = 0.3
N = 100


def generate_data():

    defectDist = ot.Uniform(0.11, 0.59)
    epsilon = ot.Normal(0, sigmaEps)

    def computeSignal(a):
        a = np.array(a)
        size = a.size
        y = beta0 + beta1 * a + np.array(epsilon.getSample(size))
        return y

    defectSupport = defectDist.getSample(N)
    signal = computeSignal(defectSupport)
    invBoxCox = ot.InverseBoxCoxTransform(lamb)
    signalInvBoxCox = invBoxCox(signal)
    return defectSupport, signalInvBoxCox


ot.RandomGenerator.SetSeed(0)
defects, signals = generate_data()

detection = 200.0
noiseThres = 60.0
saturationThres = 1700.0

####### Test on the POD models ###################
# Test polynomial chaos without Box Cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD1 = otpod.PolynomialChaosPOD(defects, signals, detection, boxCox=False)
POD1.setSamplingSize(300)
POD1.setSimulationSize(100)
POD1.run()
detectionSize1 = POD1.computeDetectionSize(0.9, 0.95)


def test_1_a90():
    np.testing.assert_almost_equal(detectionSize1[0], 0.361646416941)


def test_1_a95():
    np.testing.assert_almost_equal(detectionSize1[1], 0.377797311045)


def test_1_R2_90():
    np.testing.assert_almost_equal(POD1.getR2(), 0.848236884469)


def test_1_Q2_90():
    np.testing.assert_almost_equal(POD1.getQ2(), 0.84004097148)


# Test polynomial chaos with censored data without Box Cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD2 = otpod.PolynomialChaosPOD(
    defects, signals, detection, noiseThres, saturationThres, boxCox=False
)
POD2.setSamplingSize(300)
POD2.setSimulationSize(100)
POD2.run()
detectionSize2 = POD2.computeDetectionSize(0.9, 0.95)


def test_2_a90():
    np.testing.assert_almost_equal(detectionSize2[0], 0.334598985609)


def test_2_a95():
    np.testing.assert_almost_equal(detectionSize2[1], 0.351002668123)


def test_2_R2_90():
    np.testing.assert_almost_equal(POD2.getR2(), 0.855357879893)


def test_2_Q2_90():
    np.testing.assert_almost_equal(POD2.getQ2(), 0.847884860997)


# Test polynomial chaos with Box Cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD3 = otpod.PolynomialChaosPOD(defects, signals, detection, boxCox=True)
POD3.setSamplingSize(300)
POD3.setSimulationSize(100)
POD3.run()
detectionSize3 = POD3.computeDetectionSize(0.9, 0.95)


def test_3_a90():
    np.testing.assert_almost_equal(detectionSize3[0], 0.313797361708)


def test_3_a95():
    np.testing.assert_almost_equal(detectionSize3[1], 0.327468037601)


def test_3_R2_90():
    np.testing.assert_almost_equal(POD3.getR2(), 0.890005925785)


def test_3_Q2_90():
    np.testing.assert_almost_equal(POD3.getQ2(), 0.886608537592)


# Test polynomial chaos with censored data with Box Cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD4 = otpod.PolynomialChaosPOD(
    defects, signals, detection, noiseThres, saturationThres, boxCox=True
)
POD4.setSamplingSize(300)
POD4.setSimulationSize(100)
POD4.run()
detectionSize4 = POD4.computeDetectionSize(0.9, 0.95)


def test_4_a90():
    np.testing.assert_almost_equal(detectionSize4[0], 0.29077057016094426)


def test_4_a95():
    np.testing.assert_almost_equal(detectionSize4[1], 0.3127756005340909)


def test_4_R2_90():
    np.testing.assert_almost_equal(POD4.getR2(), 0.8716647905822066)


def test_4_Q2_90():
    np.testing.assert_almost_equal(POD4.getQ2(), 0.86811804068316134)
