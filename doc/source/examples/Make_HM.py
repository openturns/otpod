from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble.forest import RandomForestRegressor
import numpy as np
import openturns as ot
import os

base_dir = os.getcwd()
while base_dir.split('/')[-1] != 'otpod':
    base_dir = os.path.dirname(base_dir)

base_dir = base_dir + os.sep + 'doc' + os.sep + 'source' + os.sep + 'examples' + os.sep


d = 4
inputs = ot.NumericalSample.ImportFromTextFile(base_dir+'inputs.txt', '\t')
X = np.array(inputs)
X = X[:, :d]
(i1,i2)=(0,1)

x_min = np.min(np.array(X),axis=0)
x_max = np.max(np.array(X),axis=0)
outputs = ot.NumericalSample.ImportFromTextFile(base_dir+'outputs.txt', '\t')
y = np.array(outputs).reshape((1,len(outputs)))[0]


x_min = np.min(X,axis=0)
x_max = np.max(X,axis=0)

n_train = 5000

X_train = np.array(X)[:n_train,:d]
y_train = y[:n_train]

X_test = np.array(X)[n_train:,:d]
y_true = y[n_train:]


reg =AdaBoostRegressor(RandomForestRegressor(),n_estimators=50) #, random_state=rng)
fit_train = reg.fit(X_train,y_train)

#plt.plot(y_true,fit_train.predict(X_test),'.')
#plt.plot(y_true,y_true,color="red",lw=2)

reg =AdaBoostRegressor(RandomForestRegressor(),n_estimators=20) #, random_state=rng)
fit_all = reg.fit(X,y)
s = 33
global fit_all, s

#plt.plot(y,fit_all.predict(X),'.')
#plt.plot(y,y,color="red",lw=2)

def MyHM_py(X):
    import numpy as np
    return(np.array(fit_all.predict(X)> s,dtype='int'))


def fit_all_py(X):
    import numpy as np
    return np.atleast_2d(fit_all.predict(X)).T

MyHM = ot.PythonFunction(d, 1, MyHM_py)
MyFit = ot.PythonFunction(d, 1, fit_all_py)

print("-----------------------------------")
print("The function 'MyHM' has been loaded")
print("MyHM inputs dimension : %i" %d)
print("MyHM output dimension : ")
print("1 if signal > "+ str(s))
print("0 if signal < "+ str(s))