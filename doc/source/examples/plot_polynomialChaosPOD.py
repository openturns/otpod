#!/usr/bin/env python
# coding: utf-8

# # Polynomial chaos POD

# In[1]:


# import relevant module
import openturns as ot
import otpod
# enable display figure in notebook
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    pass


# ## Generate 1D data

# In[2]:


N = 100
ot.RandomGenerator.SetSeed(123456)
defectDist = ot.Uniform(0.1, 0.6)
# normal epsilon distribution
epsilon = ot.Normal(0, 1.9)
defects = defectDist.getSample(N)
signalsInvBoxCox = defects * 43. + epsilon.getSample(N) + 2.5
# Inverse Box Cox transformation
invBoxCox = ot.InverseBoxCoxTransform(0.3)
signals = invBoxCox(signalsInvBoxCox)


# ## Build POD with polynomial chaos model

# In[3]:


# signal detection threshold
detection = 200.
# The POD with censored data actually builds a POD only on filtered data.
# A warning is diplayed in this case.
POD = otpod.PolynomialChaosPOD(defects, signals, detection,
                               noiseThres=200., saturationThres=1700.,
                               boxCox=True)


# ### User-defined defect sizes
# 
# The user-defined defect sizes must range between the minimum and
# maximum of the defect values after filtering. An error is raised if 
# it is not the case. The available range is then returned to the user.

# In[4]:


# Default defect sizes
print('Default defect sizes : ')
print(POD.getDefectSizes())

# Wrong range
try:
    POD.setDefectSizes([0.12, 0.3, 0.5, 0.57])
except ValueError as e:
    print('')
    print('Range of the defect sizes is too large, it returns a value error : ')
    print(e)


# In[5]:


# Good range
POD.setDefectSizes([0.1929, 0.3, 0.4, 0.5, 0.5979])
print('User-defined defect size : ')
print(POD.getDefectSizes())


# ### Running the polynomial chaos based POD
# 
# The computing time can be reduced by setting the simulation size attribute to 
# another value. However the confidence interval is less accurate.
# 
# The sampling size is the number of the samples used to compute the POD
# with the Monte Carlo simulation for each defect sizes.
# 
# A progress is displayed, which can be disabled with the method *setVerbose*.

# In[6]:


# Computing the confidence interval in the run takes few minutes.
POD = otpod.PolynomialChaosPOD(defects, signals, detection,
                                  boxCox=True)
# we can change the sample size of the Monte Carlo simulation
POD.setSamplingSize(2000) # default is 5000
# we can also change the size of the simulation to compute the confidence interval
POD.setSimulationSize(500) # default is 1000
# we can change the degree of the polynomial chaos, default is 3.
POD.setDegree(3)
POD.run()


# ## Compute detection size

# In[7]:


# Detection size at probability level 0.9
# and confidence level 0.95
print(POD.computeDetectionSize(0.9, 0.95))

# probability level 0.95 with confidence level 0.99
print(POD.computeDetectionSize(0.95, 0.99))


# ## get POD Function

# In[8]:


# get the POD model
PODmodel = POD.getPODModel()
# get the POD model at the given confidence level
PODmodelCl95 = POD.getPODCLModel(0.95)

# compute the probability of detection for a given defect value
print('POD : {:0.3f}'.format(PODmodel([0.3])[0]))
print('POD at level 0.95 : {:0.3f}'.format(PODmodelCl95([0.3])[0]))


# ## Compute the R2 and the Q2
# Enable to check the quality of the model.

# In[9]:


print('R2 : {:0.4f}'.format(POD.getR2()))
print('Q2 : {:0.4f}'.format(POD.getQ2()))


# ## Show POD graphs
# ### Mean POD and POD at confidence level with the detection size for a given probability level

# In[10]:


fig, ax = POD.drawPOD(probabilityLevel=0.9, confidenceLevel=0.95,
                      name='figure/PODPolyChaos.png')
# The figure is saved in PODPolyChaos.png
fig.show()


# ### Show the polynomial chaos model (only available if the input dimension is 1) 

# In[11]:


fig, ax = POD.drawPolynomialChaosModel()
fig.show()


# ## Advanced user mode
# 
# The user can defined one or all parameters of the polynomial chaos algorithm : 
# - the distribution of the input parameters
# - the adaptive strategy
# - the projection strategy

# In[12]:


# new POD study
PODnew = otpod.PolynomialChaosPOD(defects, signals, detection,
                               boxCox=True)


# In[13]:


# define the input parameter distribution
distribution = ot.ComposedDistribution([ot.Normal(0.3, 0.1)])
PODnew.setDistribution(distribution)


# In[14]:


# define the adaptive strategy
polyCol = [ot.HermiteFactory()]
enumerateFunction = ot.LinearEnumerateFunction(1)
multivariateBasis = ot.OrthogonalProductPolynomialFactory(polyCol, enumerateFunction)
# degree 1
p = 1
indexMax = enumerateFunction.getStrataCumulatedCardinal(p)
adaptiveStrategy = ot.FixedStrategy(multivariateBasis, indexMax)

PODnew.setAdaptiveStrategy(adaptiveStrategy)


# In[15]:


# define the projection strategy
projectionStrategy = ot.LeastSquaresStrategy()
PODnew.setProjectionStrategy(projectionStrategy)


# In[16]:


POD.setSamplingSize(2000)
POD.setSimulationSize(500)
PODnew.run()


# In[17]:


print(PODnew.computeDetectionSize(0.9, 0.95))
print('R2 : {:0.4f}'.format(POD.getR2()))
print('Q2 : {:0.4f}'.format(POD.getQ2()))


# In[18]:


fig, ax = PODnew.drawPOD(probabilityLevel=0.9, confidenceLevel=0.95)
fig.show()


# In[19]:


fig, ax = PODnew.drawPolynomialChaosModel()
fig.show()


# In[ ]:




