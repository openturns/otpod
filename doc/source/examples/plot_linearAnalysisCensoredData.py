#!/usr/bin/env python
# coding: utf-8

# # Linear model analysis with censored data

# In[1]:


# import relevant module
import openturns as ot
import otpod
# enable display figure in notebook
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    pass


# ## Generate data

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


# ## Run analysis with Box Cox

# In[3]:


noiseThres = 60.
saturationThres = 1700.
analysis = otpod.UnivariateLinearModelAnalysis(defects, signals, noiseThres,
                                               saturationThres, boxCox=True)


# ## Get some particular results
# Result values are given for both analysis performed on filtered data (uncensored case) and on censored data.

# In[4]:


print(analysis.getIntercept())
print(analysis.getR2())
print(analysis.getKolmogorovPValue())


# ## Print all results of the linear regression and all tests on the residuals

# In[5]:


# Results are displayed for both case
print(analysis.getResults())


# ## Save all results in a csv file

# In[6]:


analysis.saveResults('results.csv')


# ## Show graphs
# ### The linear regression model with data for the uncensored case (default case)

# In[7]:


# draw the figure for the uncensored case and save it as png file
fig, ax = analysis.drawLinearModel(name='figure/linearModelUncensored.png')
fig.show()


# ### The linear regression model with data for the censored case

# In[8]:


# draw the figure for the censored case and save it as png file
fig, ax = analysis.drawLinearModel(model='censored', name='figure/linearModelCensored.png')
fig.show()


# In[ ]:





# In[ ]:




