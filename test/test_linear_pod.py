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
# Test linear regression with no hypothesis on residuals and Box Cox
np.random.seed(0)
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
np.random.seed(0)
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
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD3 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, None, boxCox=True)
POD3.run()
detectionSize3 = POD3.computeDetectionSize(0.9, 0.95)
def test_3_a90():
    np.testing.assert_almost_equal(detectionSize3[0], 0.298335917914)
def test_3_a9095():
    np.testing.assert_almost_equal(detectionSize3[1], 0.30669905666)


# Test linear regression with no hypothesis on residuals, high censored data and Box Cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD4 = otpod.UnivariateLinearModelPOD(defects, signals, detection, None, saturationThres, boxCox=True)
POD4.run()
detectionSize4 = POD4.computeDetectionSize(0.9, 0.95)
def test_4_a90():
    np.testing.assert_almost_equal(detectionSize4[0], 0.310853812252)
def test_4_a9095():
    np.testing.assert_almost_equal(detectionSize4[1], 0.321207886529)


# Test without Box Cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD7 = otpod.UnivariateLinearModelPOD(defects, signals, detection, boxCox=False)
POD7.run()
detectionSize7 = POD7.computeDetectionSize(0.9, 0.95)
def test_7_a90():
    np.testing.assert_almost_equal(detectionSize7[0], 0.315363719618)
def test_7_a9095():
    np.testing.assert_almost_equal(detectionSize7[1], 0.336508146999)

# Test linear regression with no hypothesis on residuals, censored data
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD8 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, saturationThres, boxCox=False)
POD8.run()
detectionSize8 = POD8.computeDetectionSize(0.9, 0.95)
def test_8_a90():
    np.testing.assert_almost_equal(detectionSize8[0], 0.322675144764)
def test_8_a9095():
    np.testing.assert_almost_equal(detectionSize8[1], 0.338746591291)


# Test linear regression with no hypothesis on residuals, low censored data
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD9 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, None, boxCox=False)
POD9.run()
detectionSize9 = POD9.computeDetectionSize(0.9, 0.95)
def test_9_a90():
    np.testing.assert_almost_equal(detectionSize9[0], 0.325755096299)
def test_9_a9095():
    np.testing.assert_almost_equal(detectionSize9[1], 0.343380433369)


# Test linear regression with no hypothesis on residuals, high censored data
np.random.seed(0)
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
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD13 = otpod.UnivariateLinearModelPOD(defects, signals, detection, resDistFact=resDistFact, boxCox=True)
POD13.setSimulationSize(100)
POD13.run()
detectionSize13 = POD13.computeDetectionSize(0.9, 0.95)
def test_13_a90():
    np.testing.assert_almost_equal(detectionSize13[0], 0.31541137427928123, decimal=5)
def test_13_a9095():
    np.testing.assert_almost_equal(detectionSize13[1], 0.33314702909334737, decimal=5)

# Test with censored data and box cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD14 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=True)
POD14.setSimulationSize(100)
POD14.run()
detectionSize14 = POD14.computeDetectionSize(0.9, 0.95)
def test_14_a90():
    np.testing.assert_almost_equal(detectionSize14[0], 0.3062333393379461, decimal=5)
def test_14_a9095():
    np.testing.assert_almost_equal(detectionSize14[1], 0.3229732037016516, decimal=5)


# Test linear regression with no hypothesis on residuals, low censored data and Box Cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD15 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, None, resDistFact=resDistFact, boxCox=True)
POD15.setSimulationSize(100)
POD15.run()
detectionSize15 = POD15.computeDetectionSize(0.9, 0.95)
def test_15_a90():
    np.testing.assert_almost_equal(detectionSize15[0], 0.3039399788079371, decimal=5)
def test_15_a9095():
    np.testing.assert_almost_equal(detectionSize15[1], 0.31965901134536634, decimal=5)


# Test linear regression with no hypothesis on residuals, high censored data and Box Cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD16 = otpod.UnivariateLinearModelPOD(defects, signals, detection, None, saturationThres, resDistFact=resDistFact, boxCox=True)
POD16.setSimulationSize(100)
POD16.run()
detectionSize16 = POD16.computeDetectionSize(0.9, 0.95)
def test_16_a90():
    np.testing.assert_almost_equal(detectionSize16[0], 0.31474659240318204, decimal=5)
def test_16_a9095():
    np.testing.assert_almost_equal(detectionSize16[1], 0.3327904293170683, decimal=5)


# Test from the linear analysis
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, resDistFact=resDistFact, boxCox=True)
ot.RandomGenerator.SetSeed(0)
POD17 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD17.setSimulationSize(100)
POD17.run()
detectionSize17 = POD17.computeDetectionSize(0.9, 0.95)
def test_17_a90():
    np.testing.assert_almost_equal(detectionSize17[0], 0.31541137427928123, decimal=5)
def test_17_a9095():
    np.testing.assert_almost_equal(detectionSize17[1], 0.33314702909334737, decimal=5)


# Test from the linear analysis with censored data
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=True)
ot.RandomGenerator.SetSeed(0)
POD18 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD18.setSimulationSize(100)
POD18.run()
detectionSize18 = POD18.computeDetectionSize(0.9, 0.95)
def test_18_a90():
    np.testing.assert_almost_equal(detectionSize18[0], 0.3062333393379461, decimal=5)
def test_18_a9095():
    np.testing.assert_almost_equal(detectionSize18[1], 0.3229732037016516, decimal=5)



# Test without Box Cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD19 = otpod.UnivariateLinearModelPOD(defects, signals, detection, resDistFact=resDistFact, boxCox=False)
POD19.setSimulationSize(100)
POD19.run()
detectionSize19 = POD19.computeDetectionSize(0.9, 0.95)
def test_19_a90():
    np.testing.assert_almost_equal(detectionSize19[0], 0.32764225985585627, decimal=5)
def test_19_a9095():
    np.testing.assert_almost_equal(detectionSize19[1], 0.33928149971670996, decimal=5)

# Test linear regression with no hypothesis on residuals, censored data
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD20 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=False)
POD20.setSimulationSize(100)
POD20.run()
detectionSize20 = POD20.computeDetectionSize(0.9, 0.95)
def test_20_a90():
    np.testing.assert_almost_equal(detectionSize20[0], 0.3323508693323901, decimal=5)
def test_20_a9095():
    np.testing.assert_almost_equal(detectionSize20[1], 0.34262733110886456, decimal=5)


# Test linear regression with no hypothesis on residuals, low censored data
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD21 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, None, resDistFact=resDistFact, boxCox=False)
POD21.setSimulationSize(100)
POD21.run()
detectionSize21 = POD21.computeDetectionSize(0.9, 0.95)
def test_21_a90():
    np.testing.assert_almost_equal(detectionSize21[0], 0.3349379996736849, decimal=5)
def test_21_a9095():
    np.testing.assert_almost_equal(detectionSize21[1], 0.34532971301553483, decimal=5)


# Test linear regression with no hypothesis on residuals, high censored data
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD22 = otpod.UnivariateLinearModelPOD(defects, signals, detection, None, saturationThres, resDistFact=resDistFact, boxCox=False)
POD22.setSimulationSize(100)
POD22.run()
detectionSize22 = POD22.computeDetectionSize(0.9, 0.95)
def test_22_a90():
    np.testing.assert_almost_equal(detectionSize22[0], 0.32500154645463586, decimal=5)
def test_22_a9095():
    np.testing.assert_almost_equal(detectionSize22[1], 0.3369901285582817, decimal=5)


# Test from the linear analysis
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, resDistFact=resDistFact, boxCox=False)
ot.RandomGenerator.SetSeed(0)
POD23 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD23.setSimulationSize(100)
POD23.run()
detectionSize23 = POD23.computeDetectionSize(0.9, 0.95)
def test_23_a90():
    np.testing.assert_almost_equal(detectionSize23[0], 0.32764225985585627, decimal=5)
def test_23_a9095():
    np.testing.assert_almost_equal(detectionSize23[1], 0.33928149971670996, decimal=5)


# Test from the linear analysis
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=False)
ot.RandomGenerator.SetSeed(0)
POD24 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD24.setSimulationSize(100)
POD24.run()
detectionSize24 = POD24.computeDetectionSize(0.9, 0.95)
def test_24_a90():
    np.testing.assert_almost_equal(detectionSize24[0], 0.3323508693323901, decimal=5)
def test_24_a9095():
    np.testing.assert_almost_equal(detectionSize24[1], 0.34262733110886456, decimal=5)



######### Test with the Linear regression and Normal factory #################
resDistFact = ot.NormalFactory()
# Test with Box Cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD25 = otpod.UnivariateLinearModelPOD(defects, signals, detection, resDistFact=resDistFact, boxCox=True)
POD25.run()
detectionSize25 = POD25.computeDetectionSize(0.9, 0.95)
def test_25_a90():
    np.testing.assert_almost_equal(detectionSize25[0], 0.313117629683)
def test_25_a9095():
    np.testing.assert_almost_equal(detectionSize25[1], 0.324672954397)

# Test with censored data and box cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD26 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=True)
POD26.run()
detectionSize26 = POD26.computeDetectionSize(0.9, 0.95)
def test_26_a90():
    np.testing.assert_almost_equal(detectionSize26[0], 0.311596807823)
def test_26_a9095():
    np.testing.assert_almost_equal(detectionSize26[1], 0.323809926781, decimal=6)


# Test linear regression with no hypothesis on residuals, low censored data and Box Cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD27 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, None, resDistFact=resDistFact, boxCox=True)
POD27.run()
detectionSize27 = POD27.computeDetectionSize(0.9, 0.95)
def test_27_a90():
    np.testing.assert_almost_equal(detectionSize27[0], 0.31307250797)
def test_27_a9095():
    np.testing.assert_almost_equal(detectionSize27[1], 0.325364972586)


# Test linear regression with no hypothesis on residuals, high censored data and Box Cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD28 = otpod.UnivariateLinearModelPOD(defects, signals, detection, None, saturationThres, resDistFact=resDistFact, boxCox=True)
POD28.run()
detectionSize28 = POD28.computeDetectionSize(0.9, 0.95)
def test_28_a90():
    np.testing.assert_almost_equal(detectionSize28[0], 0.309757773325)
def test_28_a9095():
    np.testing.assert_almost_equal(detectionSize28[1], 0.321470244842)


# Test from the linear analysis
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, resDistFact=resDistFact, boxCox=True)
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD29 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD29.run()
detectionSize29 = POD29.computeDetectionSize(0.9, 0.95)
def test_29_a90():
    np.testing.assert_almost_equal(detectionSize29[0], 0.313117628628)
def test_29_a9095():
    np.testing.assert_almost_equal(detectionSize29[1], 0.324672953337)


# Test from the linear analysis with censored data
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=True)
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD30 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD30.run()
detectionSize30 = POD30.computeDetectionSize(0.9, 0.95)
def test_30_a90():
    np.testing.assert_almost_equal(detectionSize30[0], 0.311596807823)
def test_30_a9095():
    np.testing.assert_almost_equal(detectionSize30[1], 0.323809926781, decimal=6)


# Test without Box Cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD31 = otpod.UnivariateLinearModelPOD(defects, signals, detection, resDistFact=resDistFact, boxCox=False)
POD31.run()
detectionSize31 = POD31.computeDetectionSize(0.9, 0.95)
def test_31_a90():
    np.testing.assert_almost_equal(detectionSize31[0], 0.331990846766)
def test_31_a9095():
    np.testing.assert_almost_equal(detectionSize31[1], 0.349279588446)

# Test linear regression with no hypothesis on residuals, censored data
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD32 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=False)
POD32.run()
detectionSize32 = POD32.computeDetectionSize(0.9, 0.95)
def test_32_a90():
    np.testing.assert_almost_equal(detectionSize32[0], 0.327103582077)
def test_32_a9095():
    np.testing.assert_almost_equal(detectionSize32[1], 0.342861356319)


# Test linear regression with no hypothesis on residuals, low censored data
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD33 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, None, resDistFact=resDistFact, boxCox=False)
POD33.run()
detectionSize33 = POD33.computeDetectionSize(0.9, 0.95)
def test_33_a90():
    np.testing.assert_almost_equal(detectionSize33[0], 0.338759336615)
def test_33_a9095():
    np.testing.assert_almost_equal(detectionSize33[1], 0.355642257962)


# Test linear regression with no hypothesis on residuals, high censored data
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD34 = otpod.UnivariateLinearModelPOD(defects, signals, detection, None, saturationThres, resDistFact=resDistFact, boxCox=False)
POD34.run()
detectionSize34 = POD34.computeDetectionSize(0.9, 0.95)
def test_34_a90():
    np.testing.assert_almost_equal(detectionSize34[0], 0.319447937432)
def test_34_a9095():
    np.testing.assert_almost_equal(detectionSize34[1], 0.33529012435)


# Test from the linear analysis
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, resDistFact=resDistFact, boxCox=False)
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD35 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD35.run()
detectionSize35 = POD35.computeDetectionSize(0.9, 0.95)
def test_35_a90():
    np.testing.assert_almost_equal(detectionSize35[0], 0.331990846766)
def test_35_a9095():
    np.testing.assert_almost_equal(detectionSize35[1], 0.349279588446)


# Test from the linear analysis
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=False)
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD36 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD36.run()
detectionSize36 = POD36.computeDetectionSize(0.9, 0.95)
def test_36_a90():
    np.testing.assert_almost_equal(detectionSize36[0], 0.327103582077)
def test_36_a9095():
    np.testing.assert_almost_equal(detectionSize36[1], 0.342861356319)


######### Test with filterCensoredData #######################################
dataFiltered = otpod.DataHandling.filterCensoredData(defects, signals,
                                                    noiseThres, saturationThres)
# linear regression binomial
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD37 = otpod.UnivariateLinearModelPOD(dataFiltered[0], dataFiltered[3],
                            detection, resDistFact=None, boxCox=False)
POD37.run()
detectionSize37 = POD37.computeDetectionSize(0.9, 0.95)
def test_37_a90():
    np.testing.assert_almost_equal(detectionSize37[0], 0.301918226192)
def test_37_a9095():
    np.testing.assert_almost_equal(detectionSize37[1], 0.331276651014)

# linear regression binomial with box cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD38 = otpod.UnivariateLinearModelPOD(dataFiltered[0], dataFiltered[3],
                            detection, resDistFact=None, boxCox=True)
POD38.run()
detectionSize38 = POD38.computeDetectionSize(0.9, 0.95)
def test_38_a90():
    np.testing.assert_almost_equal(detectionSize38[0], 0.285933846089)
def test_38_a9095():
    np.testing.assert_almost_equal(detectionSize38[1], 0.301444232939)

# linear regression Gauss
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD39 = otpod.UnivariateLinearModelPOD(dataFiltered[0], dataFiltered[3],
                            detection, resDistFact=ot.NormalFactory(), boxCox=False)
POD39.run()
detectionSize39 = POD39.computeDetectionSize(0.9, 0.95)
def test_39_a90():
    np.testing.assert_almost_equal(detectionSize39[0], 0.315143231299)
def test_39_a9095():
    np.testing.assert_almost_equal(detectionSize39[1], 0.330683811473)

# linear regression Gauss with box cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD40 = otpod.UnivariateLinearModelPOD(dataFiltered[0], dataFiltered[3],
                            detection, resDistFact=ot.NormalFactory(), boxCox=True)
POD40.run()
detectionSize40 = POD40.computeDetectionSize(0.9, 0.95)
def test_40_a90():
    np.testing.assert_almost_equal(detectionSize40[0], 0.299146252531)
def test_40_a9095():
    np.testing.assert_almost_equal(detectionSize40[1], 0.312151336843)

# linear regression Kernel smoothing
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD41 = otpod.UnivariateLinearModelPOD(dataFiltered[0], dataFiltered[3],
                            detection, resDistFact=ot.KernelSmoothing(), boxCox=False)
POD41.setSimulationSize(100)
POD41.run()
detectionSize41 = POD41.computeDetectionSize(0.9, 0.95)
def test_41_a90():
    np.testing.assert_almost_equal(detectionSize41[0], 0.31607502563958656)
def test_41_a9095():
    np.testing.assert_almost_equal(detectionSize41[1], 0.3324353655255358)

# linear regression kernel smoothing with box cox
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
POD42 = otpod.UnivariateLinearModelPOD(dataFiltered[0], dataFiltered[3],
                            detection, resDistFact=ot.KernelSmoothing(), boxCox=True)
POD42.setSimulationSize(100)
POD42.run()
detectionSize42 = POD42.computeDetectionSize(0.9, 0.95)
def test_42_a90():
    np.testing.assert_almost_equal(detectionSize42[0], 0.29132498542758567)
def test_42_a9095():
    np.testing.assert_almost_equal(detectionSize42[1], 0.3074995096280702)
