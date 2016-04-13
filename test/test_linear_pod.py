import openturns as ot
import otpod
import numpy as np

beta0 = 2.52
beta1 = 43.28
sigmaEps= 1.95
lamb = 0.3
N = 100

def generate_data():
    ot.RandomGenerator.SetSeed(0)
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

defects, signals = generate_data()

detection = 200.
noiseThres = 60
saturationThres = 1700


# Test linear regression with no hypothesis on residuals and Box Cox
ot.RandomGenerator.SetSeed(0)
POD = otpod.UnivariateLinearModelPOD(defects, signals, detection, boxCox=True)
POD.run()
def test_1_PODModel():
    PODmodel = POD.getPODModel()
    np.testing.assert_almost_equal(PODmodel([0.3])[0], 0.8700000000000001)
def test_1_PODModelCl():
    PODmodelCl = POD.getPODCLModel()
    np.testing.assert_almost_equal(PODmodelCl([0.3])[0], 0.8128338902211437)


# Test linear regression with no hypothesis on residuals, censored data and Box Cox
ot.RandomGenerator.SetSeed(0)
POD = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, saturationThres, boxCox=True)
POD.run()
def test_2_PODModel():
    PODmodel = POD.getPODModel()
    np.testing.assert_almost_equal(PODmodel([0.3])[0], 0.9425287356321839)
def test_2_PODModelCl():
    PODmodelCl = POD.getPODCLModel()
    np.testing.assert_almost_equal(PODmodelCl([0.3])[0], 0.8978734696051756)