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

noiseThres = 60
saturationThres = 1700


# Test default analysis
ot.RandomGenerator.SetSeed(0)
analysis1 = otpod.UnivariateLinearModelAnalysis(defects, signals)
def test_1_residual_dist():
    dist = analysis1.getResidualsDistribution()[0]
    param = dist.getParametersCollection()[0]
    Nparam = dist.getParameterDimension()
    values = [param[i] for i in range(Nparam)]
    np.testing.assert_almost_equal(values, [8.412825991399587e-13, 267.52649350967033], decimal=5)
def test_1_intercept():
    np.testing.assert_almost_equal(analysis1.getIntercept()[0], -660.288442844, decimal=5)
def test_1_slope():
    np.testing.assert_almost_equal(analysis1.getSlope()[0], 3634.82388325, decimal=5)
def test_1_standard_error():
    np.testing.assert_almost_equal(analysis1.getStandardError()[0], 268.887960285, decimal=5)
def test_1_boxcox():
    assert analysis1.getBoxCoxParameter() == None
def test_1_R2():
    np.testing.assert_almost_equal(analysis1.getR2()[0], 0.7833036490272336, decimal=5)
def test_1_kolmogorov():
    np.testing.assert_almost_equal(analysis1.getKolmogorovPValue()[0], 0.566952771557, decimal=5)
def test_1_anderson():
    np.testing.assert_almost_equal(analysis1.getAndersonDarlingPValue()[0], 0.0180973077091, decimal=5)
def test_1_cramer():
    np.testing.assert_almost_equal(analysis1.getCramerVonMisesPValue()[0], 0.0546363022558, decimal=5)
def test_1_zeromean():
    np.testing.assert_almost_equal(analysis1.getZeroMeanPValue()[0], 1.0, decimal=5)
def test_1_breusch():
    np.testing.assert_almost_equal(analysis1.getBreuschPaganPValue()[0], 0.00166534636771, decimal=5)
def test_1_harrison():
    np.testing.assert_almost_equal(analysis1.getHarrisonMcCabePValue()[0], 0.323, decimal=5)
def test_1_durbin():
    np.testing.assert_almost_equal(analysis1.getDurbinWatsonPValue()[0], 0.616564347897, decimal=5)
def test_1_warnings():
    msg = analysis1._printWarnings()
    assert msg == ['Some hypothesis tests failed : you may consider to use the Box Cox transformation.', '', '']

# Test with box cox
ot.RandomGenerator.SetSeed(0)
analysis2 = otpod.UnivariateLinearModelAnalysis(defects, signals, boxCox=True)
def test_2_residual_dist():
    dist = analysis2.getResidualsDistribution()[0]
    param = dist.getParametersCollection()[0]
    Nparam = dist.getParameterDimension()
    values = [param[i] for i in range(Nparam)]
    np.testing.assert_almost_equal(values, [7.44293515708705e-15, 1.9277340138649486], decimal=5)
def test_2_intercept():
    np.testing.assert_almost_equal(analysis2.getIntercept()[0], 2.49892917786, decimal=5)
def test_2_slope():
    np.testing.assert_almost_equal(analysis2.getSlope()[0], 39.1864764933, decimal=4)
def test_2_standard_error():
    np.testing.assert_almost_equal(analysis2.getStandardError()[0], 1.93754442844, decimal=5)
def test_2_boxcox():
    np.testing.assert_almost_equal(analysis2.getBoxCoxParameter(), 0.282443618674, decimal=5)
def test_2_R2():
    np.testing.assert_almost_equal(analysis2.getR2()[0], 0.890005925784, decimal=5)
def test_2_kolmogorov():
    np.testing.assert_almost_equal(analysis2.getKolmogorovPValue()[0], 0.858370833532, decimal=5)
def test_2_anderson():
    np.testing.assert_almost_equal(analysis2.getAndersonDarlingPValue()[0], 0.479646586112, decimal=5)
def test_2_cramer():
    np.testing.assert_almost_equal(analysis2.getCramerVonMisesPValue()[0], 0.465610653203, decimal=5)
def test_2_zeromean():
    np.testing.assert_almost_equal(analysis2.getZeroMeanPValue()[0], 1.0, decimal=5)
def test_2_breusch():
    np.testing.assert_almost_equal(analysis2.getBreuschPaganPValue()[0], 0.756088416904, decimal=5)
def test_2_harrison():
    np.testing.assert_almost_equal(analysis2.getHarrisonMcCabePValue()[0], 0.032, decimal=5)
def test_2_durbin():
    np.testing.assert_almost_equal(analysis2.getDurbinWatsonPValue()[0], 0.565540794481, decimal=5)
def test_2_warnings():
    msg = analysis2._printWarnings()
    assert msg == ['', 'Some hypothesis tests failed : you may consider to use quantile regression or kriging (if input dimension > 1) to build POD.', '']


# Test with residual distribution as Weibull
ot.RandomGenerator.SetSeed(0)
analysis3 = otpod.UnivariateLinearModelAnalysis(defects, signals, resDistFact=ot.WeibullMinFactory())
def test_3_residual_dist():
    dist = analysis3.getResidualsDistribution()[0]
    param = dist.getParametersCollection()[0]
    Nparam = dist.getParameterDimension()
    values = [param[i] for i in range(Nparam)]
    np.testing.assert_almost_equal(values, [797.98712, 2.75994, -712.89819], decimal=5)
def test_3_intercept():
    np.testing.assert_almost_equal(analysis3.getIntercept()[0], -660.288442844, decimal=5)
def test_3_slope():
    np.testing.assert_almost_equal(analysis3.getSlope()[0], 3634.82388325, decimal=5)
def test_3_standard_error():
    np.testing.assert_almost_equal(analysis3.getStandardError()[0], 268.887960285, decimal=5)
def test_3_boxcox():
    assert analysis3.getBoxCoxParameter() == None
def test_3_R2():
    np.testing.assert_almost_equal(analysis3.getR2()[0], 0.783303649027, decimal=5)
def test_3_kolmogorov():
    np.testing.assert_almost_equal(analysis3.getKolmogorovPValue()[0], 0.55593, decimal=5)
def test_3_anderson():
    np.testing.assert_almost_equal(analysis3.getAndersonDarlingPValue()[0], 0.0180973077091, decimal=5)
def test_3_cramer():
    np.testing.assert_almost_equal(analysis3.getCramerVonMisesPValue()[0], 0.0546363022558, decimal=5)
def test_3_zeromean():
    np.testing.assert_almost_equal(analysis3.getZeroMeanPValue()[0], 1.0, decimal=5)
def test_3_breusch():
    np.testing.assert_almost_equal(analysis3.getBreuschPaganPValue()[0], 0.00166534636771, decimal=5)
def test_3_harrison():
    np.testing.assert_almost_equal(analysis3.getHarrisonMcCabePValue()[0], 0.323, decimal=5)
def test_3_durbin():
    np.testing.assert_almost_equal(analysis3.getDurbinWatsonPValue()[0], 0.616564347897, decimal=5)
def test_3_warnings():
    msg = analysis3._printWarnings()
    assert msg == ['Some hypothesis tests failed : you may consider to use the Box Cox transformation.', '', 'Confidence interval, Normality tests and zero residual mean test are given assuming the residuals follow a Normal distribution.']


# Test residual distribution as Weibull and box cox
ot.RandomGenerator.SetSeed(0)
analysis4 = otpod.UnivariateLinearModelAnalysis(defects, signals, resDistFact=ot.WeibullMinFactory(), boxCox=True)
def test_4_residual_dist():
    dist = analysis4.getResidualsDistribution()[0]
    param = dist.getParametersCollection()[0]
    Nparam = dist.getParameterDimension()
    values = [param[i] for i in range(Nparam)]
    np.testing.assert_almost_equal(values, [7.22875, 3.71098, -6.54064], decimal=5)
def test_4_intercept():
    np.testing.assert_almost_equal(analysis4.getIntercept()[0], 2.49892917613, decimal=5)
def test_4_slope():
    np.testing.assert_almost_equal(analysis4.getSlope()[0], 39.1864765171, decimal=4)
def test_4_standard_error():
    np.testing.assert_almost_equal(analysis4.getStandardError()[0], 1.93754442961, decimal=5)
def test_4_boxcox():
    np.testing.assert_almost_equal(analysis4.getBoxCoxParameter(), 0.282443618774, decimal=5)
def test_4_R2():
    np.testing.assert_almost_equal(analysis4.getR2()[0], 0.890005925785, decimal=5)
def test_4_kolmogorov():
    np.testing.assert_almost_equal(analysis4.getKolmogorovPValue()[0], 0.678058, decimal=5)
def test_4_anderson():
    np.testing.assert_almost_equal(analysis4.getAndersonDarlingPValue()[0], 0.47964658675, decimal=5)
def test_4_cramer():
    np.testing.assert_almost_equal(analysis4.getCramerVonMisesPValue()[0], 0.46561065371, decimal=5)
def test_4_zeromean():
    np.testing.assert_almost_equal(analysis4.getZeroMeanPValue()[0], 1.0, decimal=5)
def test_4_breusch():
    np.testing.assert_almost_equal(analysis4.getBreuschPaganPValue()[0], 0.756088417911, decimal=5)
def test_4_harrison():
    np.testing.assert_almost_equal(analysis4.getHarrisonMcCabePValue()[0], 0.032, decimal=5)
def test_4_durbin():
    np.testing.assert_almost_equal(analysis4.getDurbinWatsonPValue()[0], 0.565540794446, decimal=5)
def test_4_warnings():
    msg = analysis4._printWarnings()
    assert msg == ['', 'Some hypothesis tests failed : you may consider to use quantile regression or kriging (if input dimension > 1) to build POD.', 'Confidence interval, Normality tests and zero residual mean test are given assuming the residuals follow a Normal distribution.']

# Test with censored data
ot.RandomGenerator.SetSeed(0)
analysis5 = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres, saturationThres)
def test_5_residual_dist():
    dist = analysis5.getResidualsDistribution()[0]
    param = dist.getParametersCollection()[0]
    Nparam = dist.getParameterDimension()
    values = [param[i] for i in range(Nparam)]
    np.testing.assert_almost_equal(values, [-4.044376583314961e-13, 193.4290569841005], decimal=5)
def test_5_intercept():
    np.testing.assert_almost_equal(analysis5.getIntercept()[0], -548.273499069, decimal=5)
def test_5_slope():
    np.testing.assert_almost_equal(analysis5.getSlope()[0], 3170.9853813, decimal=5)
def test_5_standard_error():
    np.testing.assert_almost_equal(analysis5.getStandardError()[0], 194.563547985, decimal=5)
def test_5_boxcox():
    assert analysis5.getBoxCoxParameter() == None
def test_5_R2():
    np.testing.assert_almost_equal(analysis5.getR2()[0], 0.813100745002, decimal=5)
def test_5_kolmogorov():
    np.testing.assert_almost_equal(analysis5.getKolmogorovPValue()[0], 0.85666607103, decimal=5)
def test_5_anderson():
    np.testing.assert_almost_equal(analysis5.getAndersonDarlingPValue()[0], 0.459142308323, decimal=5)
def test_5_cramer():
    np.testing.assert_almost_equal(analysis5.getCramerVonMisesPValue()[0], 0.355438037073, decimal=5)
def test_5_zeromean():
    np.testing.assert_almost_equal(analysis5.getZeroMeanPValue()[0], 1.0, decimal=5)
def test_5_breusch():
    np.testing.assert_almost_equal(analysis5.getBreuschPaganPValue()[0], 0.0141575095912, decimal=5)
def test_5_harrison():
    np.testing.assert_almost_equal(analysis5.getHarrisonMcCabePValue()[0], 0.153, decimal=5)
def test_5_durbin():
    np.testing.assert_almost_equal(analysis5.getDurbinWatsonPValue()[0], 0.462419719618, decimal=5)
def test_5_warnings():
    msg = analysis5._printWarnings()
    assert msg == ['Some hypothesis tests failed : you may consider to use the Box Cox transformation.', '', '']


# Test with censored data and box cox
ot.RandomGenerator.SetSeed(0)
analysis6 = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres, saturationThres, boxCox=True)
def test_6_residual_dist():
    dist = analysis6.getResidualsDistribution()[0]
    param = dist.getParametersCollection()[0]
    Nparam = dist.getParameterDimension()
    values = [param[i] for i in range(Nparam)]
    np.testing.assert_almost_equal(values, [5.155518413201876e-15, 1.3889338794567685], decimal=5)
def test_6_intercept():
    np.testing.assert_almost_equal(analysis6.getIntercept()[0], 4.28145128492, decimal=5)
def test_6_slope():
    np.testing.assert_almost_equal(analysis6.getSlope()[0], 28.4503453525, decimal=5)
def test_6_standard_error():
    np.testing.assert_almost_equal(analysis6.getStandardError()[0], 1.39708018907, decimal=5)
def test_6_boxcox():
    np.testing.assert_almost_equal(analysis6.getBoxCoxParameter(), 0.248476219177, decimal=5)
def test_6_R2():
    np.testing.assert_almost_equal(analysis6.getR2()[0], 0.871664790582, decimal=5)
def test_6_kolmogorov():
    np.testing.assert_almost_equal(analysis6.getKolmogorovPValue()[0], 0.414252477686, decimal=5)
def test_6_anderson():
    np.testing.assert_almost_equal(analysis6.getAndersonDarlingPValue()[0], 0.0439405361179, decimal=5)
def test_6_cramer():
    np.testing.assert_almost_equal(analysis6.getCramerVonMisesPValue()[0], 0.0481033088591, decimal=5)
def test_6_zeromean():
    np.testing.assert_almost_equal(analysis6.getZeroMeanPValue()[0], 1.0, decimal=5)
def test_6_breusch():
    np.testing.assert_almost_equal(analysis6.getBreuschPaganPValue()[0], 0.859123679656, decimal=5)
def test_6_harrison():
    np.testing.assert_almost_equal(analysis6.getHarrisonMcCabePValue()[0], 0.061, decimal=5)
def test_6_durbin():
    np.testing.assert_almost_equal(analysis6.getDurbinWatsonPValue()[0], 0.771991711693, decimal=5)
def test_6_warnings():
    msg = analysis6._printWarnings()
    assert msg == ['', 'Some hypothesis tests failed : you may consider to use quantile regression or kriging (if input dimension > 1) to build POD.', '']

# Test with censored data and Weibull distribution
ot.RandomGenerator.SetSeed(0)
analysis7 = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres, saturationThres, resDistFact=ot.WeibullMinFactory(), boxCox=False)
def test_7_residual_dist():
    dist = analysis7.getResidualsDistribution()[0]
    param = dist.getParametersCollection()[0]
    Nparam = dist.getParameterDimension()
    values = [param[i] for i in range(Nparam)]
    np.testing.assert_almost_equal(values, [727.52686, 3.76709, -657.67541], decimal=5)
def test_7_intercept():
    np.testing.assert_almost_equal(analysis7.getIntercept()[0], -548.273499069, decimal=5)
def test_7_slope():
    np.testing.assert_almost_equal(analysis7.getSlope()[0], 3170.9853813, decimal=5)
def test_7_standard_error():
    np.testing.assert_almost_equal(analysis7.getStandardError()[0], 194.563547985, decimal=5)
def test_7_boxcox():
    assert analysis7.getBoxCoxParameter() == None
def test_7_R2():
    np.testing.assert_almost_equal(analysis7.getR2()[0], 0.813100745002, decimal=5)
def test_7_kolmogorov():
    np.testing.assert_almost_equal(analysis7.getKolmogorovPValue()[0], 0.828167, decimal=5)
def test_7_anderson():
    np.testing.assert_almost_equal(analysis7.getAndersonDarlingPValue()[0], 0.459142308323, decimal=5)
def test_7_cramer():
    np.testing.assert_almost_equal(analysis7.getCramerVonMisesPValue()[0], 0.355438037073, decimal=5)
def test_7_zeromean():
    np.testing.assert_almost_equal(analysis7.getZeroMeanPValue()[0], 1.0, decimal=5)
def test_7_breusch():
    np.testing.assert_almost_equal(analysis7.getBreuschPaganPValue()[0], 0.0141575095912, decimal=5)
def test_7_harrison():
    np.testing.assert_almost_equal(analysis7.getHarrisonMcCabePValue()[0], 0.153, decimal=5)
def test_7_durbin():
    np.testing.assert_almost_equal(analysis7.getDurbinWatsonPValue()[0], 0.462419719618, decimal=5)
def test_7_warnings():
    msg = analysis7._printWarnings()
    assert msg == ['Some hypothesis tests failed : you may consider to use the Box Cox transformation.', '', 'Confidence interval, Normality tests and zero residual mean test are given assuming the residuals follow a Normal distribution.']


# Test with censored data, Weibull distribution and box cox
ot.RandomGenerator.SetSeed(0)
analysis8 = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres, saturationThres, resDistFact=ot.WeibullMinFactory(), boxCox=True)
def test_8_residual_dist():
    dist = analysis8.getResidualsDistribution()[0]
    param = dist.getParametersCollection()[0]
    Nparam = dist.getParameterDimension()
    values = [param[i] for i in range(Nparam)]
    np.testing.assert_almost_equal(values, [5.37449, 3.78189, -4.87536], decimal=5)
def test_8_intercept():
    np.testing.assert_almost_equal(analysis8.getIntercept()[0], 4.28145128492, decimal=5)
def test_8_slope():
    np.testing.assert_almost_equal(analysis8.getSlope()[0], 28.4503453525, decimal=5)
def test_8_standard_error():
    np.testing.assert_almost_equal(analysis8.getStandardError()[0], 1.39708018907, decimal=5)
def test_8_boxcox():
    np.testing.assert_almost_equal(analysis8.getBoxCoxParameter(), 0.248476219177, decimal=5)
def test_8_R2():
    np.testing.assert_almost_equal(analysis8.getR2()[0], 0.871664790582, decimal=5)
def test_8_kolmogorov():
    np.testing.assert_almost_equal(analysis8.getKolmogorovPValue()[0], 0.4006136, decimal=5)
def test_8_anderson():
    np.testing.assert_almost_equal(analysis8.getAndersonDarlingPValue()[0], 0.0439405361179, decimal=5)
def test_8_cramer():
    np.testing.assert_almost_equal(analysis8.getCramerVonMisesPValue()[0], 0.0481033088591, decimal=5)
def test_8_zeromean():
    np.testing.assert_almost_equal(analysis8.getZeroMeanPValue()[0], 1.0, decimal=5)
def test_8_breusch():
    np.testing.assert_almost_equal(analysis8.getBreuschPaganPValue()[0], 0.859123679656, decimal=5)
def test_8_harrison():
    np.testing.assert_almost_equal(analysis8.getHarrisonMcCabePValue()[0], 0.061, decimal=5)
def test_8_durbin():
    np.testing.assert_almost_equal(analysis8.getDurbinWatsonPValue()[0], 0.771991711693, decimal=5)
def test_8_warnings():
    msg = analysis8._printWarnings()
    assert msg == ['', 'Some hypothesis tests failed : you may consider to use quantile regression or kriging (if input dimension > 1) to build POD.', 'Confidence interval, Normality tests and zero residual mean test are given assuming the residuals follow a Normal distribution.']


# Test with low censored data
ot.RandomGenerator.SetSeed(0)
analysis9 = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres)
def test_9_residual_dist():
    dist = analysis9.getResidualsDistribution()[0]
    param = dist.getParametersCollection()[0]
    Nparam = dist.getParameterDimension()
    values = [param[i] for i in range(Nparam)]
    np.testing.assert_almost_equal(values, [-1.1126796883392207e-13, 272.05069780052406], decimal=5)
def test_9_intercept():
    np.testing.assert_almost_equal(analysis9.getIntercept()[0], -715.154543834, decimal=5)
def test_9_slope():
    np.testing.assert_almost_equal(analysis9.getSlope()[0], 3755.95821498, decimal=5)
def test_9_standard_error():
    np.testing.assert_almost_equal(analysis9.getStandardError()[0], 273.525238128, decimal=5)
def test_9_boxcox():
    assert analysis9.getBoxCoxParameter() == None
def test_9_R2():
    np.testing.assert_almost_equal(analysis9.getR2()[0], 0.770763274557, decimal=5)
def test_9_kolmogorov():
    np.testing.assert_almost_equal(analysis9.getKolmogorovPValue()[0], 0.466989666954, decimal=5)
def test_9_anderson():
    np.testing.assert_almost_equal(analysis9.getAndersonDarlingPValue()[0], 0.0216220791687, decimal=5)
def test_9_cramer():
    np.testing.assert_almost_equal(analysis9.getCramerVonMisesPValue()[0], 0.0461963923033, decimal=5)
def test_9_zeromean():
    np.testing.assert_almost_equal(analysis9.getZeroMeanPValue()[0], 1.0, decimal=5)
def test_9_breusch():
    np.testing.assert_almost_equal(analysis9.getBreuschPaganPValue()[0], 0.00448746848369, decimal=5)
def test_9_harrison():
    np.testing.assert_almost_equal(analysis9.getHarrisonMcCabePValue()[0], 0.358, decimal=5)
def test_9_durbin():
    np.testing.assert_almost_equal(analysis9.getDurbinWatsonPValue()[0], 0.623449231786, decimal=5)
def test_9_warnings():
    msg = analysis9._printWarnings()
    assert msg == ['Some hypothesis tests failed : you may consider to use the Box Cox transformation.', '', '']


# Test with low censored data and box cox
ot.RandomGenerator.SetSeed(0)
analysis10 = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres, None, boxCox=True)
def test_10_residual_dist():
    dist = analysis10.getResidualsDistribution()[0]
    param = dist.getParametersCollection()[0]
    Nparam = dist.getParameterDimension()
    values = [param[i] for i in range(Nparam)]
    np.testing.assert_almost_equal(values, [-7.369991142192528e-16, 1.045892341894865], decimal=5)
def test_10_intercept():
    np.testing.assert_almost_equal(analysis10.getIntercept()[0], 4.12213956419, decimal=5)
def test_10_slope():
    np.testing.assert_almost_equal(analysis10.getSlope()[0], 21.6589228719, decimal=5)
def test_10_standard_error():
    np.testing.assert_almost_equal(analysis10.getStandardError()[0], 1.0515611766, decimal=5)
def test_10_boxcox():
    np.testing.assert_almost_equal(analysis10.getBoxCoxParameter(), 0.195379066467, decimal=5)
def test_10_R2():
    np.testing.assert_almost_equal(analysis10.getR2()[0], 0.883242398756, decimal=5)
def test_10_kolmogorov():
    np.testing.assert_almost_equal(analysis10.getKolmogorovPValue()[0], 0.69014192754, decimal=5)
def test_10_anderson():
    np.testing.assert_almost_equal(analysis10.getAndersonDarlingPValue()[0], 0.0851157758707, decimal=5)
def test_10_cramer():
    np.testing.assert_almost_equal(analysis10.getCramerVonMisesPValue()[0], 0.106869822661, decimal=5)
def test_10_zeromean():
    np.testing.assert_almost_equal(analysis10.getZeroMeanPValue()[0], 1.0, decimal=5)
def test_10_breusch():
    np.testing.assert_almost_equal(analysis10.getBreuschPaganPValue()[0], 0.86092186831, decimal=5)
def test_10_harrison():
    np.testing.assert_almost_equal(analysis10.getHarrisonMcCabePValue()[0], 0.081, decimal=5)
def test_10_durbin():
    np.testing.assert_almost_equal(analysis10.getDurbinWatsonPValue()[0], 0.961978522957, decimal=5)
def test_10_warnings():
    msg = analysis10._printWarnings()
    assert msg == ['', '', '']


# Test with low censored data and Weibull distribution
ot.RandomGenerator.SetSeed(0)
analysis11 = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres, None, resDistFact=ot.WeibullMinFactory(), boxCox=False)
def test_11_residual_dist():
    dist = analysis11.getResidualsDistribution()[0]
    param = dist.getParametersCollection()[0]
    Nparam = dist.getParameterDimension()
    values = [param[i] for i in range(Nparam)]
    np.testing.assert_almost_equal(values, [811.9707, 2.76702, -725.31732], decimal=5)
def test_11_intercept():
    np.testing.assert_almost_equal(analysis11.getIntercept()[0], -715.154543834, decimal=5)
def test_11_slope():
    np.testing.assert_almost_equal(analysis11.getSlope()[0], 3755.95821498, decimal=5)
def test_11_standard_error():
    np.testing.assert_almost_equal(analysis11.getStandardError()[0], 273.525238128, decimal=5)
def test_11_boxcox():
    assert analysis11.getBoxCoxParameter() == None
def test_11_R2():
    np.testing.assert_almost_equal(analysis11.getR2()[0], 0.770763274557, decimal=5)
def test_11_kolmogorov():
    np.testing.assert_almost_equal(analysis11.getKolmogorovPValue()[0], 0.6662212, decimal=5)
def test_11_anderson():
    np.testing.assert_almost_equal(analysis11.getAndersonDarlingPValue()[0], 0.0216220791687, decimal=5)
def test_11_cramer():
    np.testing.assert_almost_equal(analysis11.getCramerVonMisesPValue()[0], 0.0461963923033, decimal=5)
def test_11_zeromean():
    np.testing.assert_almost_equal(analysis11.getZeroMeanPValue()[0], 1.0, decimal=5)
def test_11_breusch():
    np.testing.assert_almost_equal(analysis11.getBreuschPaganPValue()[0], 0.00448746848369, decimal=5)
def test_11_harrison():
    np.testing.assert_almost_equal(analysis11.getHarrisonMcCabePValue()[0], 0.358, decimal=5)
def test_11_durbin():
    np.testing.assert_almost_equal(analysis11.getDurbinWatsonPValue()[0], 0.623449231786, decimal=5)
def test_11_warnings():
    msg = analysis11._printWarnings()
    assert msg == ['Some hypothesis tests failed : you may consider to use the Box Cox transformation.', '', 'Confidence interval, Normality tests and zero residual mean test are given assuming the residuals follow a Normal distribution.']


# Test with low censored data, Weibull distribution and box cox
ot.RandomGenerator.SetSeed(0)
analysis12 = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres, None, resDistFact=ot.WeibullMinFactory(), boxCox=True)
def test_12_residual_dist():
    dist = analysis12.getResidualsDistribution()[0]
    param = dist.getParametersCollection()[0]
    Nparam = dist.getParameterDimension()
    values = [param[i] for i in range(Nparam)]
    np.testing.assert_almost_equal(values, [4.02804, 3.76232, -3.65199], decimal=5)
def test_12_intercept():
    np.testing.assert_almost_equal(analysis12.getIntercept()[0], 4.12213956422, decimal=5)
def test_12_slope():
    np.testing.assert_almost_equal(analysis12.getSlope()[0], 21.658922871, decimal=5)
def test_12_standard_error():
    np.testing.assert_almost_equal(analysis12.getStandardError()[0], 1.05156117655, decimal=5)
def test_12_boxcox():
    np.testing.assert_almost_equal(analysis12.getBoxCoxParameter(), 0.19537906646, decimal=5)
def test_12_R2():
    np.testing.assert_almost_equal(analysis12.getR2()[0], 0.883242398757, decimal=5)
def test_12_kolmogorov():
    np.testing.assert_almost_equal(analysis12.getKolmogorovPValue()[0], 0.5241805, decimal=5)
def test_12_anderson():
    np.testing.assert_almost_equal(analysis12.getAndersonDarlingPValue()[0], 0.0851157758678, decimal=5)
def test_12_cramer():
    np.testing.assert_almost_equal(analysis12.getCramerVonMisesPValue()[0], 0.106869822657, decimal=5)
def test_12_zeromean():
    np.testing.assert_almost_equal(analysis12.getZeroMeanPValue()[0], 1.0, decimal=5)
def test_12_breusch():
    np.testing.assert_almost_equal(analysis12.getBreuschPaganPValue()[0], 0.860921868345, decimal=5)
def test_12_harrison():
    np.testing.assert_almost_equal(analysis12.getHarrisonMcCabePValue()[0], 0.081, decimal=5)
def test_12_durbin():
    np.testing.assert_almost_equal(analysis12.getDurbinWatsonPValue()[0], 0.96197852296, decimal=5)
#def test_12_warnings():
    #msg = analysis12._printWarnings()
    #assert msg == ['', 'Some hypothesis tests failed : you may consider to use quantile regression or kriging (if input dimension > 1) to build POD.', 'Confidence interval, Normality tests and zero residual mean test are given assuming the residuals follow a Normal distribution.']


# Test with high censored data
ot.RandomGenerator.SetSeed(0)
analysis13 = otpod.UnivariateLinearModelAnalysis(defects, signals, None, saturationThres)
def test_13_residual_dist():
    dist = analysis13.getResidualsDistribution()[0]
    param = dist.getParametersCollection()[0]
    Nparam = dist.getParameterDimension()
    values = [param[i] for i in range(Nparam)]
    np.testing.assert_almost_equal(values, [-1.1001952037575745e-13, 189.58500581594078], decimal=5)
def test_13_intercept():
    np.testing.assert_almost_equal(analysis13.getIntercept()[0], -515.170723345, decimal=5)
def test_13_slope():
    np.testing.assert_almost_equal(analysis13.getSlope()[0], 3094.49716693, decimal=5)
def test_13_standard_error():
    np.testing.assert_almost_equal(analysis13.getStandardError()[0], 190.623835542, decimal=5)
def test_13_boxcox():
    assert analysis13.getBoxCoxParameter() == None
def test_13_R2():
    np.testing.assert_almost_equal(analysis13.getR2()[0], 0.82718696356, decimal=5)
def test_13_kolmogorov():
    np.testing.assert_almost_equal(analysis13.getKolmogorovPValue()[0], 0.789486311952, decimal=5)
def test_13_anderson():
    np.testing.assert_almost_equal(analysis13.getAndersonDarlingPValue()[0], 0.504387479088, decimal=5)
def test_13_cramer():
    np.testing.assert_almost_equal(analysis13.getCramerVonMisesPValue()[0], 0.402894000718, decimal=5)
def test_13_zeromean():
    np.testing.assert_almost_equal(analysis13.getZeroMeanPValue()[0], 1.0, decimal=5)
def test_13_breusch():
    np.testing.assert_almost_equal(analysis13.getBreuschPaganPValue()[0], 0.00260710624553, decimal=5)
def test_13_harrison():
    np.testing.assert_almost_equal(analysis13.getHarrisonMcCabePValue()[0], 0.128, decimal=5)
def test_13_durbin():
    np.testing.assert_almost_equal(analysis13.getDurbinWatsonPValue()[0], 0.610205622901, decimal=5)
def test_13_warnings():
    msg = analysis13._printWarnings()
    assert msg == ['Some hypothesis tests failed : you may consider to use the Box Cox transformation.', '', '']


# Test with high censored data and box cox
ot.RandomGenerator.SetSeed(0)
analysis14 = otpod.UnivariateLinearModelAnalysis(defects, signals, None, saturationThres, boxCox=True)
def test_14_residual_dist():
    dist = analysis14.getResidualsDistribution()[0]
    param = dist.getParametersCollection()[0]
    Nparam = dist.getParameterDimension()
    values = [param[i] for i in range(Nparam)]
    np.testing.assert_almost_equal(values, [-1.2606403376388875e-15, 2.4508852295357273], decimal=5)
def test_14_intercept():
    np.testing.assert_almost_equal(analysis14.getIntercept()[0], 2.13686409675, decimal=5)
def test_14_slope():
    np.testing.assert_almost_equal(analysis14.getSlope()[0], 50.0190600361, decimal=5)
def test_14_standard_error():
    np.testing.assert_almost_equal(analysis14.getStandardError()[0], 2.46431483817, decimal=5)
def test_14_boxcox():
    np.testing.assert_almost_equal(analysis14.getBoxCoxParameter(), 0.331569838524, decimal=5)
def test_14_R2():
    np.testing.assert_almost_equal(analysis14.getR2()[0], 0.882118212744, decimal=5)
def test_14_kolmogorov():
    np.testing.assert_almost_equal(analysis14.getKolmogorovPValue()[0], 0.657004524302, decimal=5)
def test_14_anderson():
    np.testing.assert_almost_equal(analysis14.getAndersonDarlingPValue()[0], 0.372627831653, decimal=5)
def test_14_cramer():
    np.testing.assert_almost_equal(analysis14.getCramerVonMisesPValue()[0], 0.369904297054, decimal=5)
def test_14_zeromean():
    np.testing.assert_almost_equal(analysis14.getZeroMeanPValue()[0], 1.0, decimal=5)
def test_14_breusch():
    np.testing.assert_almost_equal(analysis14.getBreuschPaganPValue()[0], 0.496055842, decimal=5)
def test_14_harrison():
    np.testing.assert_almost_equal(analysis14.getHarrisonMcCabePValue()[0], 0.015, decimal=5)
def test_14_durbin():
    np.testing.assert_almost_equal(analysis14.getDurbinWatsonPValue()[0], 0.816864678987, decimal=5)
def test_14_warnings():
    msg = analysis14._printWarnings()
    assert msg == ['', 'Some hypothesis tests failed : you may consider to use quantile regression or kriging (if input dimension > 1) to build POD.', '']


# Test with high censored data and Weibull distribution
ot.RandomGenerator.SetSeed(0)
analysis15 = otpod.UnivariateLinearModelAnalysis(defects, signals, None, saturationThres, resDistFact=ot.WeibullMinFactory(), boxCox=False)
def test_15_residual_dist():
    dist = analysis15.getResidualsDistribution()[0]
    param = dist.getParametersCollection()[0]
    Nparam = dist.getParameterDimension()
    values = [param[i] for i in range(Nparam)]
    np.testing.assert_almost_equal(values, [713.96296, 3.76608, -645.5258], decimal=5)
def test_15_intercept():
    np.testing.assert_almost_equal(analysis15.getIntercept()[0], -515.170723345, decimal=5)
def test_15_slope():
    np.testing.assert_almost_equal(analysis15.getSlope()[0], 3094.49716693, decimal=5)
def test_15_standard_error():
    np.testing.assert_almost_equal(analysis15.getStandardError()[0], 190.623835542, decimal=5)
def test_15_boxcox():
    assert analysis15.getBoxCoxParameter() == None
def test_15_R2():
    np.testing.assert_almost_equal(analysis15.getR2()[0], 0.82718696356, decimal=5)
def test_15_kolmogorov():
    np.testing.assert_almost_equal(analysis15.getKolmogorovPValue()[0], 0.7887058, decimal=5)
def test_15_anderson():
    np.testing.assert_almost_equal(analysis15.getAndersonDarlingPValue()[0], 0.504387479088, decimal=5)
def test_15_cramer():
    np.testing.assert_almost_equal(analysis15.getCramerVonMisesPValue()[0], 0.402894000718, decimal=5)
def test_15_zeromean():
    np.testing.assert_almost_equal(analysis15.getZeroMeanPValue()[0], 1.0, decimal=5)
def test_15_breusch():
    np.testing.assert_almost_equal(analysis15.getBreuschPaganPValue()[0], 0.00260710624553, decimal=5)
def test_15_harrison():
    np.testing.assert_almost_equal(analysis15.getHarrisonMcCabePValue()[0], 0.128, decimal=5)
def test_15_durbin():
    np.testing.assert_almost_equal(analysis15.getDurbinWatsonPValue()[0], 0.610205622901, decimal=5)
def test_15_warnings():
    msg = analysis15._printWarnings()
    assert msg == ['Some hypothesis tests failed : you may consider to use the Box Cox transformation.', '', 'Confidence interval, Normality tests and zero residual mean test are given assuming the residuals follow a Normal distribution.']


# Test with high censored data, Weibull distribution and box cox
ot.RandomGenerator.SetSeed(0)
analysis16 = otpod.UnivariateLinearModelAnalysis(defects, signals, None, saturationThres, resDistFact=ot.WeibullMinFactory(), boxCox=True)
def test_16_residual_dist():
    dist = analysis16.getResidualsDistribution()[0]
    param = dist.getParametersCollection()[0]
    Nparam = dist.getParameterDimension()
    values = [param[i] for i in range(Nparam)]
    np.testing.assert_almost_equal(values, [9.23258,  3.72967, -8.35842], decimal=5)
def test_16_intercept():
    np.testing.assert_almost_equal(analysis16.getIntercept()[0], 2.13686409675, decimal=5)
def test_16_slope():
    np.testing.assert_almost_equal(analysis16.getSlope()[0], 50.0190600361, decimal=5)
def test_16_standard_error():
    np.testing.assert_almost_equal(analysis16.getStandardError()[0], 2.46431483817, decimal=5)
def test_16_boxcox():
    np.testing.assert_almost_equal(analysis16.getBoxCoxParameter(), 0.331569838524, decimal=5)
def test_16_R2():
    np.testing.assert_almost_equal(analysis16.getR2()[0], 0.882118212744, decimal=5)
def test_16_kolmogorov():
    np.testing.assert_almost_equal(analysis16.getKolmogorovPValue()[0], 0.50687167, decimal=5)
def test_16_anderson():
    np.testing.assert_almost_equal(analysis16.getAndersonDarlingPValue()[0], 0.372627831653, decimal=5)
def test_16_cramer():
    np.testing.assert_almost_equal(analysis16.getCramerVonMisesPValue()[0], 0.369904297054, decimal=5)
def test_16_zeromean():
    np.testing.assert_almost_equal(analysis16.getZeroMeanPValue()[0], 1.0, decimal=5)
def test_16_breusch():
    np.testing.assert_almost_equal(analysis16.getBreuschPaganPValue()[0], 0.496055842, decimal=5)
def test_16_harrison():
    np.testing.assert_almost_equal(analysis16.getHarrisonMcCabePValue()[0], 0.015, decimal=5)
def test_16_durbin():
    np.testing.assert_almost_equal(analysis16.getDurbinWatsonPValue()[0], 0.816864678987, decimal=5)
def test_16_warnings():
    msg = analysis16._printWarnings()
    assert msg == ['', 'Some hypothesis tests failed : you may consider to use quantile regression or kriging (if input dimension > 1) to build POD.', 'Confidence interval, Normality tests and zero residual mean test are given assuming the residuals follow a Normal distribution.']

