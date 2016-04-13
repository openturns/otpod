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
    np.testing.assert_almost_equal(PODmodel([0.3])[0], 0.942528735632)
def test_2_PODModelCl():
    PODmodelCl = POD2.getPODCLModel()
    np.testing.assert_almost_equal(PODmodelCl([0.3])[0], 0.897873469605)
def test_2_PODModelCl09():
    PODmodelCl = POD2.getPODCLModel(confidenceLevel=0.9)
    np.testing.assert_almost_equal(PODmodelCl([0.3])[0], 0.910216675197)
def test_2_PODModelCensore():
    PODmodel = POD2.getPODModel('censored')
    np.testing.assert_almost_equal(PODmodel([0.3])[0], 0.908045977011)
def test_2_PODModelClCensore():
    PODmodelCl = POD2.getPODCLModel('censored')
    np.testing.assert_almost_equal(PODmodelCl([0.3])[0], 0.840198204355)
def test_2_PODModelCl09Censore():
    PODmodelCl = POD2.getPODCLModel('censored',confidenceLevel=0.9)
    np.testing.assert_almost_equal(PODmodelCl([0.3])[0], 0.854928609849)


############### Test on the detection size #####################################

######### Test with the Linear regression and binomial hypothesis ##############
detectionSize1 = POD1.computeDetectionSize(0.9, 0.95)
def test_1_a90():
    np.testing.assert_almost_equal(detectionSize1[0][0], 0.309875091327)
def test_1_a9095():
    np.testing.assert_almost_equal(detectionSize1[1][0], 0.331125479592)

detectionSize2 = POD2.computeDetectionSize(0.9, 0.95)
def test_2_a90():
    np.testing.assert_almost_equal(detectionSize2[0][0], 0.285936878479)
def test_2_a9095():
    np.testing.assert_almost_equal(detectionSize2[1][0], 0.301445237328)
def test_2_a90c():
    np.testing.assert_almost_equal(detectionSize2[2][0], 0.299650947187)
def test_2_a9095c():
    np.testing.assert_almost_equal(detectionSize2[3][0], 0.314205304121)


# Test linear regression with no hypothesis on residuals, low censored data and Box Cox
ot.RandomGenerator.SetSeed(0)
POD3 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, None, boxCox=True)
POD3.run()
detectionSize3 = POD3.computeDetectionSize(0.9, 0.95)
def test_3_a90():
    np.testing.assert_almost_equal(detectionSize3[0][0], 0.288547940226)
def test_3_a9095():
    np.testing.assert_almost_equal(detectionSize3[1][0], 0.301249691263)
def test_3_a90c():
    np.testing.assert_almost_equal(detectionSize3[2][0], 0.298335917914)
def test_3_a9095c():
    np.testing.assert_almost_equal(detectionSize3[3][0], 0.30669905666)


# Test linear regression with no hypothesis on residuals, high censored data and Box Cox
ot.RandomGenerator.SetSeed(0)
POD4 = otpod.UnivariateLinearModelPOD(defects, signals, detection, None, saturationThres, boxCox=True)
POD4.run()
detectionSize4 = POD4.computeDetectionSize(0.9, 0.95)
def test_4_a90():
    np.testing.assert_almost_equal(detectionSize4[0][0], 0.30523744115)
def test_4_a9095():
    np.testing.assert_almost_equal(detectionSize4[1][0], 0.326713779985)
def test_4_a90c():
    np.testing.assert_almost_equal(detectionSize4[2][0], 0.310853812252)
def test_4_a9095c():
    np.testing.assert_almost_equal(detectionSize4[3][0], 0.321207886529)


# # Test from the linear analysis
# ot.RandomGenerator.SetSeed(0)
# analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, boxCox=True)
# POD5 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
# POD5.run()
# detectionSize5 = POD5.computeDetectionSize(0.9, 0.95)
# def test_5_a90():
#     np.testing.assert_almost_equal(detectionSize5[0][0], 0.309875091327)
# def test_5_a9095():
#     np.testing.assert_almost_equal(detectionSize5[1][0], 0.331125479592)


# # Test from the linear analysis with censored data
# ot.RandomGenerator.SetSeed(0)
# analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres, saturationThres, boxCox=True)
# POD6 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
# POD6.run()
# detectionSize6 = POD6.computeDetectionSize(0.9, 0.95)
# def test_6_a90():
#     np.testing.assert_almost_equal(detectionSize6[0][0], 0.285936878479)
# def test_6_a9095():
#     np.testing.assert_almost_equal(detectionSize6[1][0], 0.301445237328)
# def test_6_a90c():
#     np.testing.assert_almost_equal(detectionSize6[2][0], 0.299650947187)
# def test_6_a9095c():
#     np.testing.assert_almost_equal(detectionSize6[3][0], 0.314205304121)


# Test without Box Cox
ot.RandomGenerator.SetSeed(0)
POD7 = otpod.UnivariateLinearModelPOD(defects, signals, detection, boxCox=False)
POD7.run()
detectionSize7 = POD7.computeDetectionSize(0.9, 0.95)
def test_7_a90():
    np.testing.assert_almost_equal(detectionSize7[0][0], 0.315363719618)
def test_7_a9095():
    np.testing.assert_almost_equal(detectionSize7[1][0], 0.336508146999)

# Test linear regression with no hypothesis on residuals, censored data
ot.RandomGenerator.SetSeed(0)
POD8 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, saturationThres, boxCox=False)
POD8.run()
detectionSize8 = POD8.computeDetectionSize(0.9, 0.95)
def test_8_a90():
    np.testing.assert_almost_equal(detectionSize8[0][0], 0.301912404146)
def test_8_a9095():
    np.testing.assert_almost_equal(detectionSize8[1][0], 0.331264210996)
def test_8_a90c():
    np.testing.assert_almost_equal(detectionSize8[2][0], 0.322675144764)
def test_8_a9095c():
    np.testing.assert_almost_equal(detectionSize8[3][0], 0.338746591291)


# Test linear regression with no hypothesis on residuals, low censored data
ot.RandomGenerator.SetSeed(0)
POD9 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, None, boxCox=False)
POD9.run()
detectionSize9 = POD9.computeDetectionSize(0.9, 0.95)
def test_9_a90():
    np.testing.assert_almost_equal(detectionSize9[0][0], 0.321848597143)
def test_9_a9095():
    np.testing.assert_almost_equal(detectionSize9[1][0], 0.34071089181)
def test_9_a90c():
    np.testing.assert_almost_equal(detectionSize9[2][0], 0.325755096299)
def test_9_a9095c():
    np.testing.assert_almost_equal(detectionSize9[3][0], 0.343380433369)


# Test linear regression with no hypothesis on residuals, high censored data
ot.RandomGenerator.SetSeed(0)
POD10 = otpod.UnivariateLinearModelPOD(defects, signals, detection, None, saturationThres, boxCox=False)
POD10.run()
detectionSize10 = POD10.computeDetectionSize(0.9, 0.95)
def test_10_a90():
    np.testing.assert_almost_equal(detectionSize10[0][0], 0.299976107854)
def test_10_a9095():
    np.testing.assert_almost_equal(detectionSize10[1][0], 0.321811449958)
def test_10_a90c():
    np.testing.assert_almost_equal(detectionSize10[2][0], 0.309059536753)
def test_10_a9095c():
    np.testing.assert_almost_equal(detectionSize10[3][0], 0.332040815487)


# # Test from the linear analysis
# ot.RandomGenerator.SetSeed(0)
# analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, boxCox=False)
# POD11 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
# POD11.run()
# detectionSize11 = POD11.computeDetectionSize(0.9, 0.95)
# def test_11_a90():
#     np.testing.assert_almost_equal(detectionSize11[0][0], 0.315363719618)
# def test_11_a9095():
#     np.testing.assert_almost_equal(detectionSize11[1][0], 0.336508146999)


# # Test from the linear analysis
# ot.RandomGenerator.SetSeed(0)
# analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres, saturationThres, boxCox=False)
# POD12 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
# POD12.run()
# detectionSize12 = POD12.computeDetectionSize(0.9, 0.95)
# def test_12_a90():
#     np.testing.assert_almost_equal(detectionSize12[0][0], 0.301912404146)
# def test_12_a9095():
#     np.testing.assert_almost_equal(detectionSize12[1][0], 0.331264210996)
# def test_12_a90c():
#     np.testing.assert_almost_equal(detectionSize12[2][0], 0.322675144764)
# def test_12_a9095c():
#     np.testing.assert_almost_equal(detectionSize12[3][0], 0.338746591291)



######### Test with the Linear regression and kernel smoothing #################
resDistFact = ot.KernelSmoothing()
# Test with Box Cox
ot.RandomGenerator.SetSeed(0)
POD13 = otpod.UnivariateLinearModelPOD(defects, signals, detection, resDistFact=resDistFact, boxCox=True)
POD13.setSimulationSize(100)
POD13.run()
detectionSize13 = POD13.computeDetectionSize(0.9, 0.95)
def test_13_a90():
    np.testing.assert_almost_equal(detectionSize13[0][0], 0.315383858389)
def test_13_a9095():
    np.testing.assert_almost_equal(detectionSize13[1][0], 0.331458994153)

# Test with censored data and box cox
ot.RandomGenerator.SetSeed(0)
POD14 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=True)
POD14.setSimulationSize(100)
POD14.run()
detectionSize14 = POD14.computeDetectionSize(0.9, 0.95)
def test_14_a90():
    np.testing.assert_almost_equal(detectionSize14[0][0], 0.291282399909)
def test_14_a9095():
    np.testing.assert_almost_equal(detectionSize14[1][0], 0.310228245177)
def test_14_a90c():
    np.testing.assert_almost_equal(detectionSize14[2][0], 0.306196744634)
def test_14_a9095c():
    np.testing.assert_almost_equal(detectionSize14[3][0], 0.323242622884, decimal=6)


# Test linear regression with no hypothesis on residuals, low censored data and Box Cox
ot.RandomGenerator.SetSeed(0)
POD15 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, None, resDistFact=resDistFact, boxCox=True)
POD15.setSimulationSize(100)
POD15.run()
detectionSize15 = POD15.computeDetectionSize(0.9, 0.95)
def test_15_a90():
    np.testing.assert_almost_equal(detectionSize15[0][0], 0.295735964553)
def test_15_a9095():
    np.testing.assert_almost_equal(detectionSize15[1][0], 0.31175928274)
def test_15_a90c():
    np.testing.assert_almost_equal(detectionSize15[2][0], 0.303907054908)
def test_15_a9095c():
    np.testing.assert_almost_equal(detectionSize15[3][0], 0.319609972955)


# Test linear regression with no hypothesis on residuals, high censored data and Box Cox
ot.RandomGenerator.SetSeed(0)
POD16 = otpod.UnivariateLinearModelPOD(defects, signals, detection, None, saturationThres, resDistFact=resDistFact, boxCox=True)
POD16.setSimulationSize(100)
POD16.run()
detectionSize16 = POD16.computeDetectionSize(0.9, 0.95)
def test_16_a90():
    np.testing.assert_almost_equal(detectionSize16[0][0], 0.311311864902)
def test_16_a9095():
    np.testing.assert_almost_equal(detectionSize16[1][0], 0.332974592594)
def test_16_a90c():
    np.testing.assert_almost_equal(detectionSize16[2][0], 0.314714793438)
def test_16_a9095c():
    np.testing.assert_almost_equal(detectionSize16[3][0], 0.330851193848)


# Test from the linear analysis
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, resDistFact=resDistFact, boxCox=True)
ot.RandomGenerator.SetSeed(0)
POD17 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD17.setSimulationSize(100)
POD17.run()
detectionSize17 = POD17.computeDetectionSize(0.9, 0.95)
def test_17_a90():
    np.testing.assert_almost_equal(detectionSize17[0][0], 0.315383858393)
def test_17_a9095():
    np.testing.assert_almost_equal(detectionSize17[1][0], 0.33145899415)


# Test from the linear analysis with censored data
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=True)
ot.RandomGenerator.SetSeed(0)
POD18 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD18.setSimulationSize(100)
POD18.run()
detectionSize18 = POD18.computeDetectionSize(0.9, 0.95)
def test_18_a90():
    np.testing.assert_almost_equal(detectionSize18[0][0], 0.291282399909)
def test_18_a9095():
    np.testing.assert_almost_equal(detectionSize18[1][0], 0.310228245177)
def test_18_a90c():
    np.testing.assert_almost_equal(detectionSize18[2][0], 0.306196744634)
def test_18_a9095c():
    np.testing.assert_almost_equal(detectionSize18[3][0], 0.323242622884, decimal=6)



# Test without Box Cox
ot.RandomGenerator.SetSeed(0)
POD19 = otpod.UnivariateLinearModelPOD(defects, signals, detection, resDistFact=resDistFact, boxCox=False)
POD19.setSimulationSize(100)
POD19.run()
detectionSize19 = POD19.computeDetectionSize(0.9, 0.95)
def test_19_a90():
    np.testing.assert_almost_equal(detectionSize19[0][0], 0.327600094017)
def test_19_a9095():
    np.testing.assert_almost_equal(detectionSize19[1][0], 0.339010405225)

# Test linear regression with no hypothesis on residuals, censored data
ot.RandomGenerator.SetSeed(0)
POD20 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=False)
POD20.setSimulationSize(100)
POD20.run()
detectionSize20 = POD20.computeDetectionSize(0.9, 0.95)
def test_20_a90():
    np.testing.assert_almost_equal(detectionSize20[0][0], 0.315997633325)
def test_20_a9095():
    np.testing.assert_almost_equal(detectionSize20[1][0], 0.330164428396)
def test_20_a90c():
    np.testing.assert_almost_equal(detectionSize20[2][0], 0.332304242822)
def test_20_a9095c():
    np.testing.assert_almost_equal(detectionSize20[3][0], 0.345249438995)


# Test linear regression with no hypothesis on residuals, low censored data
ot.RandomGenerator.SetSeed(0)
POD21 = otpod.UnivariateLinearModelPOD(defects, signals, detection, noiseThres, None, resDistFact=resDistFact, boxCox=False)
POD21.setSimulationSize(100)
POD21.run()
detectionSize21 = POD21.computeDetectionSize(0.9, 0.95)
def test_21_a90():
    np.testing.assert_almost_equal(detectionSize21[0][0], 0.332769943567)
def test_21_a9095():
    np.testing.assert_almost_equal(detectionSize21[1][0], 0.344666123986)
def test_21_a90c():
    np.testing.assert_almost_equal(detectionSize21[2][0], 0.334893306287)
def test_21_a9095c():
    np.testing.assert_almost_equal(detectionSize21[3][0], 0.346600799157)


# Test linear regression with no hypothesis on residuals, high censored data
ot.RandomGenerator.SetSeed(0)
POD22 = otpod.UnivariateLinearModelPOD(defects, signals, detection, None, saturationThres, resDistFact=resDistFact, boxCox=False)
POD22.setSimulationSize(100)
POD22.run()
detectionSize22 = POD22.computeDetectionSize(0.9, 0.95)
def test_22_a90():
    np.testing.assert_almost_equal(detectionSize22[0][0], 0.310728432327)
def test_22_a9095():
    np.testing.assert_almost_equal(detectionSize22[1][0], 0.325135515775)
def test_22_a90c():
    np.testing.assert_almost_equal(detectionSize22[2][0], 0.324956744187)
def test_22_a9095c():
    np.testing.assert_almost_equal(detectionSize22[3][0], 0.336622929806)


# Test from the linear analysis
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, resDistFact=resDistFact, boxCox=False)
ot.RandomGenerator.SetSeed(0)
POD23 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD23.setSimulationSize(100)
POD23.run()
detectionSize23 = POD23.computeDetectionSize(0.9, 0.95)
def test_23_a90():
    np.testing.assert_almost_equal(detectionSize23[0][0], 0.327600094017)
def test_23_a9095():
    np.testing.assert_almost_equal(detectionSize23[1][0], 0.339010405225)


# Test from the linear analysis
ot.RandomGenerator.SetSeed(0)
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres, saturationThres, resDistFact=resDistFact, boxCox=False)
ot.RandomGenerator.SetSeed(0)
POD24 = otpod.UnivariateLinearModelPOD(analysis=analysis, detection=detection)
POD24.setSimulationSize(100)
POD24.run()
detectionSize24 = POD24.computeDetectionSize(0.9, 0.95)
def test_24_a90():
    np.testing.assert_almost_equal(detectionSize24[0][0], 0.315997633325)
def test_24_a9095():
    np.testing.assert_almost_equal(detectionSize24[1][0], 0.330164428396)
def test_24_a90c():
    np.testing.assert_almost_equal(detectionSize24[2][0], 0.332304242822)
def test_24_a9095c():
    np.testing.assert_almost_equal(detectionSize24[3][0], 0.345249438995)

