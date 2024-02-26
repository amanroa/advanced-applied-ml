---
layout: page
title: "Project-2"
permalink: /project_2/
---

# Project 2 - Aashni Manroa

In this project, I will be experimenting with Gradient Boosting. I will be using the concrete csv that we used in class. First, we have to import various packages and load the data. 

```c
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from scipy.spatial import Delaunay
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error as mse
from scipy import linalg
from scipy.interpolate import interp1d, LinearNDInterpolator, NearestNDInterpolator
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split as tts, KFold, GridSearchCV
```

```c
data = pd.read_csv('/concrete.csv')
data
```
<img width="676" alt="Screenshot 2024-02-25 at 7 01 39 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/10af192b-0923-4982-9d22-9dae9305a94c">

This is what my data looks like. I decided to have the x variables be age and cement, and the y variable be age. I also split them into training and testing sets, with 70% for training and 30% for testing.

```c
x = data.loc[:,'cement':'age'].values
y = data['strength'].values
xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.3,shuffle=True,random_state=123)
```

There are also four kernels that I needed to define:

```c
# Gaussian Kernel
def Gaussian(x):
  return np.where(np.abs(x)>4,0,1/(np.sqrt(2*np.pi))*np.exp(-1/2*x**2))
# this is the correct vectorized version
def Tricubic(x):
  return np.where(np.abs(x)>1,0,(1-np.abs(x)**3)**3)
# Epanechnikov Kernel
def Epanechnikov(x):
  return np.where(np.abs(x)>1,0,3/4*(1-np.abs(x)**2))
# Quartic Kernel
def Quartic(x):
  return np.where(np.abs(x)>1,0,15/16*(1-np.abs(x)**2)**2)
```

## Part 1: Lowess/Gradient Boosting

I created a Lowess class containing the methods `fit`, `predict`, and `is_fitted`.

```c
class Lowess:
    def __init__(self, kernel = Gaussian, tau=0.05):
        self.kernel = kernel
        self.tau = tau

    def fit(self, x, y):
        kernel = self.kernel
        tau = self.tau
        self.xtrain_ = x
        self.yhat_ = y

    def predict(self, x_new):
        self.is_fitted()
        x = self.xtrain_
        y = self.yhat_
        lm = linear_model.Ridge(alpha=0.001)
        w = weight_function(x,x_new,self.kernel,self.tau)

        if np.isscalar(x_new):
          lm.fit(np.diag(w)@(x.reshape(-1,1)),np.diag(w)@(y.reshape(-1,1)))
          yest = lm.predict([[x_new]])[0][0]
        else:

          n = len(x_new)
          yest_test = []
          for i in range(n):
            lm.fit(np.diag(w[:,i])@x,np.diag(w[:,i])@y)
            yest_test.append(lm.predict(x_new[i].reshape(1,-1)))
        return np.array(yest_test).reshape(-1,1)

    def is_fitted(self):
      if self.tau is None or self.kernel is None:
        raise ValueError("Scaler has not been fitted yet. Please call 'fit' with the appropriate values.")
```

I also defined a scaler, which was the MinMaxScaler (chosen randomly), and a weight function. This is an example of what the x data would look like after running it through the weight function using a Tricubic kernel. 

```c
scale = MinMaxScaler()
def weight_function(u,v,kern=Gaussian,tau=0.5):
    return kern(cdist(u, v, metric='euclidean')/(2*tau))
W = weight_function(scale.fit_transform(x),scale.fit_transform(x),kern=Tricubic,tau=0.3)
W
```
<img width="624" alt="Screenshot 2024-02-25 at 7 34 06 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/4c81940c-23c0-4c95-9928-3f6f25e47e16">

Let's try creating and running this model. 

```c
model = Lowess(kernel=Epanechnikov,tau=0.05)
xscaled = scale.fit_transform(x)
model.fit(xscaled,y)
yhat = model.predict(xscaled)
yhat
```
<img width="192" alt="Screenshot 2024-02-25 at 7 44 56 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/1703d1a8-228b-4f80-90cd-8f7f9ebf8797">

```c
mse(yhat, y)
```
<img width="158" alt="Screenshot 2024-02-25 at 7 45 23 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/a2c46739-70fd-4470-a025-80263cc54cdd">







