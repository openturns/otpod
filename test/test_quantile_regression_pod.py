import openturns as ot
ot.TBB.Disable()
import otpod
import numpy as np

beta0 = 2.52
beta1 = 43.28
sigmaEps= 1.95
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

detection = 200.
noiseThres = 60.
saturationThres = 1700.

####### Test on the POD models ###################
# Test quantile regression without Box Cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD1 = otpod.QuantileRegressionPOD(defects, signals, detection, boxCox=False)
POD1.setSimulationSize(10)
POD1.run()
detectionSize1 = POD1.computeDetectionSize(0.9, 0.95)
def test_1_a90():
    np.testing.assert_almost_equal(detectionSize1[0], 0.287376656987)
def test_1_a95():
    np.testing.assert_almost_equal(detectionSize1[1], 0.302782872277)
def test_1_R2_90():
    np.testing.assert_almost_equal(POD1.getR2(0.9), 0.572190802359)

# Test quantile regression with censored data without Box Cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD2 = otpod.QuantileRegressionPOD(defects, signals, detection, noiseThres, saturationThres, boxCox=False)
POD2.setSimulationSize(10)
POD2.run()
detectionSize2 = POD2.computeDetectionSize(0.9, 0.95)
def test_2_a90():
    np.testing.assert_almost_equal(detectionSize2[0], 0.287014269172)
def test_2_a95():
    np.testing.assert_almost_equal(detectionSize2[1], 0.31470276646, decimal=5)
def test_2_R2_90():
    np.testing.assert_almost_equal(POD2.getR2(0.9), 0.601476185955)


# Test quantile regression with Box Cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD3 = otpod.QuantileRegressionPOD(defects, signals, detection, boxCox=True)
POD3.setSimulationSize(10)
POD3.run()
detectionSize3 = POD3.computeDetectionSize(0.9, 0.95)
def test_3_a90():
    np.testing.assert_almost_equal(detectionSize3[0], 0.314034473666, decimal=4)
def test_3_a95():
    np.testing.assert_almost_equal(detectionSize3[1], 0.344053946523, decimal=5)
def test_3_R2_90():
    np.testing.assert_almost_equal(POD3.getR2(0.9), 0.628868102499)

# Test quantile regression with censored data with Box Cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD4 = otpod.QuantileRegressionPOD(defects, signals, detection, noiseThres, saturationThres, boxCox=True)
POD4.setSimulationSize(10)
POD4.run()
detectionSize4 = POD4.computeDetectionSize(0.9, 0.95)
def test_4_a90():
    np.testing.assert_almost_equal(detectionSize4[0], 0.285941338048)
def test_4_a95():
    np.testing.assert_almost_equal(detectionSize4[1], 0.317139410914)
def test_4_R2_90():
    np.testing.assert_almost_equal(POD4.getR2(0.9), 0.565484415155)
