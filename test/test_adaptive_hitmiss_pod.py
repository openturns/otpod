import numpy as np
import openturns as ot

ot.TBB.Disable()
import otpod

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

# detection threshold
detection = 38

# Select point as initial DOE
inputDOE = inputSample[:]
outputDOE = signals[:]

# simulate the true physical model
basis = ot.ConstantBasisFactory(4).build()
covarianceModel = ot.SquaredExponential([5.03148, 13.9442, 20, 20], [15.1697])
krigingModel = ot.KrigingAlgorithm(inputSample, signals, covarianceModel, basis)

ot.RandomGenerator.SetSeed(0)
np.random.seed(0)
krigingModel.run()
physicalModel = krigingModel.getResult().getMetaModel()


####### Test on the POD models ###################
# Test hitmiss without Box Cox with rf classifier
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
ot.RandomGenerator.SetState(ot.RandomGeneratorState(ot.Indices([0] * 768), 0))
POD1 = otpod.AdaptiveHitMissPOD(inputDOE, outputDOE, physicalModel, 20, detection)
POD1.run()
detectionSize1 = POD1.computeDetectionSize(0.9, 0.95)


def test_1_a90():
    np.testing.assert_allclose(detectionSize1[0], 4.703290234448557, rtol=1e-1)


def test_1_a95():
    np.testing.assert_allclose(detectionSize1[1], 5.160478340894737, rtol=1e-1)


def test_1_confusion_matrix():
    np.testing.assert_allclose(
        POD1.getConfusionMatrix(),
        [[0.875, 0.08833333], [0.125, 0.91166667]],
        rtol=5e-1,
    )


####### Test on the POD models ###################
# Test hitmiss without Box Cox with svc classifier
np.random.seed(0)
ot.RandomGenerator.SetSeed(0)
ot.RandomGenerator.SetState(ot.RandomGeneratorState(ot.Indices([0] * 768), 0))
POD2 = otpod.AdaptiveHitMissPOD(inputDOE, outputDOE, physicalModel, 10, detection)
POD2.setClassifierType("svc")
POD2.run()
detectionSize2 = POD2.computeDetectionSize(0.5, 0.95)
# the test is not reproducible
# def test_2_a90():
#     np.testing.assert_almost_equal(detectionSize2[0], 4.267675047394733, decimal=5)
# def test_2_a95():
#     np.testing.assert_almost_equal(detectionSize2[1], 4.473170408027023, decimal=5)
# def test_2_confusion_matrix():
#     np.testing.assert_almost_equal(POD2.getConfusionMatrix(), [[ 0.83333333,  0.24175824], [ 0.16666667,  0.75824176]], decimal=5)
