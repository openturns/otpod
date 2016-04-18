import openturns as ot
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
# Test linear regression with no hypothesis on residuals and Box Cox
ot.RandomGenerator.SetSeed(0)
POD1 = otpod.UnivariateLinearModelPOD(defects, signals, detection, boxCox=True)
POD1.run()
def test_1_PODModel():
    PODmodel = POD1.getPODModel()
    np.testing.assert_almost_equal(PODmodel([0.3])[0], 0.8700000000000001)
def test_1_PODModelCl():
    PODmodelCl = POD1.getPODCLModel()
    np.testing.assert_almost_equal(PODmodelCl([0.3])[0], 0.8128338902211437)
def test_1_PODModelCl09():
    PODmodelCl = POD1.getPODCLModel(confidenceLevel=0.9)
    np.testing.assert_almost_equal(PODmodelCl([0.3])[0], 0.827366884824)


# Test linear regression with no hypothesis on residuals, censored data and Box Cox
ot.RandomGenerator.SetSeed(0)
POD2 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, saturationThres, boxCox=True)
POD2.run()
def test_2_PODModel():
    PODmodel = POD2.getPODModel()
    np.testing.assert_almost_equal(PODmodel([0.3])[0], 0.908045977011)
def test_2_PODModelCl():
    PODmodelCl = POD2.getPODCLModel()
    np.testing.assert_almost_equal(PODmodelCl([0.3])[0], 0.840198204355)
def test_2_PODModelCl09():
    PODmodelCl = POD2.getPODCLModel(confidenceLevel=0.9)
    np.testing.assert_almost_equal(PODmodelCl([0.3])[0], 0.854928609849)


############### Test on the detection size #####################################

######### Test with the Linear regression and binomial hypothesis ##############
detectionSize1 = POD1.computeDetectionSize(0.9, 0.95)
def test_1_a90():
    np.testing.assert_almost_equal(detectionSize1[0], 0.309875091327)
def test_1_a9095():
    np.testing.assert_almost_equal(detectionSize1[1], 0.331125479592)

detectionSize2 = POD2.computeDetectionSize(0.9, 0.95)
def test_2_a90():
    np.testing.assert_almost_equal(detectionSize2[0], 0.299650947187)
def test_2_a9095():
    np.testing.assert_almost_equal(detectionSize2[1], 0.314205304121)


# Test linear regression with no hypothesis on residuals, low censored data and Box Cox
ot.RandomGenerator.SetSeed(0)
POD3 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, None, boxCox=True)
POD3.run()
detectionSize3 = POD3.computeDetectionSize(0.9, 0.95)
def test_3_a90():
    np.testing.assert_almost_equal(detectionSize3[0], 0.298335917914)
def test_3_a9095():
    np.testing.assert_almost_equal(detectionSize3[1], 0.30669905666)


# Test linear regression with no hypothesis on residuals, high censored data and Box Cox
ot.RandomGenerator.SetSeed(0)
POD4 = otpod.UnivariateLinearModelPOD(defects, signals, detection, None, saturationThres, boxCox=True)
POD4.run()
detectionSize4 = POD4.computeDetectionSize(0.9, 0.95)
def test_4_a90():
    np.testing.assert_almost_equal(detectionSize4[0], 0.310853812252)
def test_4_a9095():
    np.testing.assert_almost_equal(detectionSize4[1], 0.321207886529)


# Test without Box Cox
ot.RandomGenerator.SetSeed(0)
POD7 = otpod.UnivariateLinearModelPOD(defects, signals, detection, boxCox=False)
POD7.run()
detectionSize7 = POD7.computeDetectionSize(0.9, 0.95)
def test_7_a90():
    np.testing.assert_almost_equal(detectionSize7[0], 0.315363719618)
def test_7_a9095():
    np.testing.assert_almost_equal(detectionSize7[1], 0.336508146999)

# Test linear regression with no hypothesis on residuals, censored data
ot.RandomGenerator.SetSeed(0)
POD8 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, saturationThres, boxCox=False)
POD8.run()
detectionSize8 = POD8.computeDetectionSize(0.9, 0.95)
def test_8_a90():
    np.testing.assert_almost_equal(detectionSize8[0], 0.322675144764)
def test_8_a9095():
    np.testing.assert_almost_equal(detectionSize8[1], 0.338746591291)


# Test linear regression with no hypothesis on residuals, low censored data
ot.RandomGenerator.SetSeed(0)
POD9 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, None, boxCox=False)
POD9.run()
detectionSize9 = POD9.computeDetectionSize(0.9, 0.95)
def test_9_a90():
    np.testing.assert_almost_equal(detectionSize9[0], 0.325755096299)
def test_9_a9095():
    np.testing.assert_almost_equal(detectionSize9[1], 0.343380433369)


# Test linear regression with no hypothesis on residuals, high censored data
ot.RandomGenerator.SetSeed(0)
POD10 = otpod.UnivariateLinearModelPOD(defects, signals, detection, None, saturationThres, boxCox=False)
POD10.run()
detectionSize10 = POD10.computeDetectionSize(0.9, 0.95)
def test_10_a90():
    np.testing.assert_almost_equal(detectionSize10[0], 0.309059536753)
def test_10_a9095():
    np.testing.assert_almost_equal(detectionSize10[1], 0.332040815487)


######### Test with the Linear regression and kernel smoothing #################
resDistFact = ot.KernelSmoothing()
# Test with Box Cox
ot.RandomGenerator.SetSeed(0)
POD13 = otpod.UnivariateLinearModelPOD(defects, signals, detection, resDistFact=resDistFact, boxCox=True)
POD13.setSimulationSize(100)
POD13.run()
detectionSize13 = POD13.computeDetectionSize(0.9, 0.95)
def test_13_a90():
    np.testing.assert_almost_equal(detectionSize13[0], 0.315383858389)
def test_13_a9095():
    np.testing.assert_almost_equal(detectionSize13[1], 0.331458994153)

# Test with censored data and box cox
ot.RandomGenerator.SetSeed(0)
POD14 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=True)
POD14.setSimulationSize(100)
POD14.run()
detectionSize14 = POD14.computeDetectionSize(0.9, 0.95)
def test_14_a90():
    np.testing.assert_almost_equal(detectionSize14[0], 0.306196744634)
def test_14_a9095():
    np.testing.assert_almost_equal(detectionSize14[1], 0.323242622884, decimal=6)


# Test linear regression with no hypothesis on residuals, low censored data and Box Cox
ot.RandomGenerator.SetSeed(0)
POD15 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, None, resDistFact=resDistFact, boxCox=True)
POD15.setSimulationSize(100)
POD15.run()
detectionSize15 = POD15.computeDetectionSize(0.9, 0.95)
def test_15_a90():
    np.testing.assert_almost_equal(detectionSize15[0], 0.303907054908)
def test_15_a9095():
    np.testing.assert_almost_equal(detectionSize15[1], 0.319609972955)


# Test linear regression with no hypothesis on residuals, high censored data and Box Cox
ot.RandomGenerator.SetSeed(0)
POD16 = otpod.UnivariateLinearModelPOD(defects, signals, detection, None, saturationThres, resDistFact=resDistFact, boxCox=True)
POD16.setSimulationSize(100)
POD16.run()
detectionSize16 = POD16.computeDetectionSize(0.9, 0.95)
def test_16_a90():
    np.testing.assert_almost_equal(detectionSize16[0], 0.314714793438)
def test_16_a9095():
    np.testing.assert_almost_equal(detectionSize16[1], 0.330851193848)


# Test from the linear analysis
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, resDistFact=resDistFact, boxCox=True)
ot.RandomGenerator.SetSeed(0)
POD17 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD17.setSimulationSize(100)
POD17.run()
detectionSize17 = POD17.computeDetectionSize(0.9, 0.95)
def test_17_a90():
    np.testing.assert_almost_equal(detectionSize17[0], 0.315383858393)
def test_17_a9095():
    np.testing.assert_almost_equal(detectionSize17[1], 0.33145899415)


# Test from the linear analysis with censored data
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=True)
ot.RandomGenerator.SetSeed(0)
POD18 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD18.setSimulationSize(100)
POD18.run()
detectionSize18 = POD18.computeDetectionSize(0.9, 0.95)
def test_18_a90():
    np.testing.assert_almost_equal(detectionSize18[0], 0.306196744634)
def test_18_a9095():
    np.testing.assert_almost_equal(detectionSize18[1], 0.323242622884, decimal=6)



# Test without Box Cox
ot.RandomGenerator.SetSeed(0)
POD19 = otpod.UnivariateLinearModelPOD(defects, signals, detection, resDistFact=resDistFact, boxCox=False)
POD19.setSimulationSize(100)
POD19.run()
detectionSize19 = POD19.computeDetectionSize(0.9, 0.95)
def test_19_a90():
    np.testing.assert_almost_equal(detectionSize19[0], 0.327600094017)
def test_19_a9095():
    np.testing.assert_almost_equal(detectionSize19[1], 0.339010405225)

# Test linear regression with no hypothesis on residuals, censored data
ot.RandomGenerator.SetSeed(0)
POD20 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=False)
POD20.setSimulationSize(100)
POD20.run()
detectionSize20 = POD20.computeDetectionSize(0.9, 0.95)
def test_20_a90():
    np.testing.assert_almost_equal(detectionSize20[0], 0.332304242822)
def test_20_a9095():
    np.testing.assert_almost_equal(detectionSize20[1], 0.345249438995)


# Test linear regression with no hypothesis on residuals, low censored data
ot.RandomGenerator.SetSeed(0)
POD21 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, None, resDistFact=resDistFact, boxCox=False)
POD21.setSimulationSize(100)
POD21.run()
detectionSize21 = POD21.computeDetectionSize(0.9, 0.95)
def test_21_a90():
    np.testing.assert_almost_equal(detectionSize21[0], 0.334893306287)
def test_21_a9095():
    np.testing.assert_almost_equal(detectionSize21[1], 0.346600799157)


# Test linear regression with no hypothesis on residuals, high censored data
ot.RandomGenerator.SetSeed(0)
POD22 = otpod.UnivariateLinearModelPOD(defects, signals, detection, None, saturationThres, resDistFact=resDistFact, boxCox=False)
POD22.setSimulationSize(100)
POD22.run()
detectionSize22 = POD22.computeDetectionSize(0.9, 0.95)
def test_22_a90():
    np.testing.assert_almost_equal(detectionSize22[0], 0.324956744187)
def test_22_a9095():
    np.testing.assert_almost_equal(detectionSize22[1], 0.336622929806)


# Test from the linear analysis
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, resDistFact=resDistFact, boxCox=False)
ot.RandomGenerator.SetSeed(0)
POD23 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD23.setSimulationSize(100)
POD23.run()
detectionSize23 = POD23.computeDetectionSize(0.9, 0.95)
def test_23_a90():
    np.testing.assert_almost_equal(detectionSize23[0], 0.327600094017)
def test_23_a9095():
    np.testing.assert_almost_equal(detectionSize23[1], 0.339010405225)


# Test from the linear analysis
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=False)
ot.RandomGenerator.SetSeed(0)
POD24 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD24.setSimulationSize(100)
POD24.run()
detectionSize24 = POD24.computeDetectionSize(0.9, 0.95)
def test_24_a90():
    np.testing.assert_almost_equal(detectionSize24[0], 0.332304242822)
def test_24_a9095():
    np.testing.assert_almost_equal(detectionSize24[1], 0.345249438995)



######### Test with the Linear regression and Normal factory #################
resDistFact = ot.NormalFactory()
# Test with Box Cox
ot.RandomGenerator.SetSeed(0)
POD25 = otpod.UnivariateLinearModelPOD(defects, signals, detection, resDistFact=resDistFact, boxCox=True)
POD25.run()
detectionSize25 = POD25.computeDetectionSize(0.9, 0.95)
def test_25_a90():
    np.testing.assert_almost_equal(detectionSize25[0], 0.313117629683)
def test_25_a9095():
    np.testing.assert_almost_equal(detectionSize25[1], 0.324672954397)

# Test with censored data and box cox
ot.RandomGenerator.SetSeed(0)
POD26 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=True)
POD26.run()
detectionSize26 = POD26.computeDetectionSize(0.9, 0.95)
def test_26_a90():
    np.testing.assert_almost_equal(detectionSize26[0], 0.311596807823)
def test_26_a9095():
    np.testing.assert_almost_equal(detectionSize26[1], 0.323809926781, decimal=6)


# Test linear regression with no hypothesis on residuals, low censored data and Box Cox
ot.RandomGenerator.SetSeed(0)
POD27 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, None, resDistFact=resDistFact, boxCox=True)
POD27.run()
detectionSize27 = POD27.computeDetectionSize(0.9, 0.95)
def test_27_a90():
    np.testing.assert_almost_equal(detectionSize27[0], 0.31307250797)
def test_27_a9095():
    np.testing.assert_almost_equal(detectionSize27[1], 0.325364972586)


# Test linear regression with no hypothesis on residuals, high censored data and Box Cox
ot.RandomGenerator.SetSeed(0)
POD28 = otpod.UnivariateLinearModelPOD(defects, signals, detection, None, saturationThres, resDistFact=resDistFact, boxCox=True)
POD28.run()
detectionSize28 = POD28.computeDetectionSize(0.9, 0.95)
def test_28_a90():
    np.testing.assert_almost_equal(detectionSize28[0], 0.309757773325)
def test_28_a9095():
    np.testing.assert_almost_equal(detectionSize28[1], 0.321470244842)


# Test from the linear analysis
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, resDistFact=resDistFact, boxCox=True)
ot.RandomGenerator.SetSeed(0)
POD29 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD29.run()
detectionSize29 = POD29.computeDetectionSize(0.9, 0.95)
def test_29_a90():
    np.testing.assert_almost_equal(detectionSize29[0], 0.313117628628)
def test_29_a9095():
    np.testing.assert_almost_equal(detectionSize29[1], 0.324672953337)


# Test from the linear analysis with censored data
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=True)
ot.RandomGenerator.SetSeed(0)
POD30 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD30.run()
detectionSize30 = POD30.computeDetectionSize(0.9, 0.95)
def test_30_a90():
    np.testing.assert_almost_equal(detectionSize30[0], 0.311596807823)
def test_30_a9095():
    np.testing.assert_almost_equal(detectionSize30[1], 0.323809926781, decimal=6)


# Test without Box Cox
ot.RandomGenerator.SetSeed(0)
POD31 = otpod.UnivariateLinearModelPOD(defects, signals, detection, resDistFact=resDistFact, boxCox=False)
POD31.run()
detectionSize31 = POD31.computeDetectionSize(0.9, 0.95)
def test_31_a90():
    np.testing.assert_almost_equal(detectionSize31[0], 0.331990846766)
def test_31_a9095():
    np.testing.assert_almost_equal(detectionSize31[1], 0.349279588446)

# Test linear regression with no hypothesis on residuals, censored data
ot.RandomGenerator.SetSeed(0)
POD32 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=False)
POD32.run()
detectionSize32 = POD32.computeDetectionSize(0.9, 0.95)
def test_32_a90():
    np.testing.assert_almost_equal(detectionSize32[0], 0.327103582077)
def test_32_a9095():
    np.testing.assert_almost_equal(detectionSize32[1], 0.342861356319)


# Test linear regression with no hypothesis on residuals, low censored data
ot.RandomGenerator.SetSeed(0)
POD33 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, None, resDistFact=resDistFact, boxCox=False)
POD33.run()
detectionSize33 = POD33.computeDetectionSize(0.9, 0.95)
def test_33_a90():
    np.testing.assert_almost_equal(detectionSize33[0], 0.338759336615)
def test_33_a9095():
    np.testing.assert_almost_equal(detectionSize33[1], 0.355642257962)


# Test linear regression with no hypothesis on residuals, high censored data
ot.RandomGenerator.SetSeed(0)
POD34 = otpod.UnivariateLinearModelPOD(defects, signals, detection, None, saturationThres, resDistFact=resDistFact, boxCox=False)
POD34.run()
detectionSize34 = POD34.computeDetectionSize(0.9, 0.95)
def test_34_a90():
    np.testing.assert_almost_equal(detectionSize34[0], 0.319447937432)
def test_34_a9095():
    np.testing.assert_almost_equal(detectionSize34[1], 0.33529012435)


# Test from the linear analysis
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, resDistFact=resDistFact, boxCox=False)
ot.RandomGenerator.SetSeed(0)
POD35 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD35.run()
detectionSize35 = POD35.computeDetectionSize(0.9, 0.95)
def test_35_a90():
    np.testing.assert_almost_equal(detectionSize35[0], 0.331990846766)
def test_35_a9095():
    np.testing.assert_almost_equal(detectionSize35[1], 0.349279588446)


# Test from the linear analysis
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=False)
ot.RandomGenerator.SetSeed(0)
POD36 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD36.run()
detectionSize36 = POD36.computeDetectionSize(0.9, 0.95)
def test_36_a90():
    np.testing.assert_almost_equal(detectionSize36[0], 0.327103582077)
def test_36_a9095():
    np.testing.assert_almost_equal(detectionSize36[1], 0.342861356319)
