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

## Lowess/Gradient Boosting

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

I also defined a scaler, which was the StandardScaler, and a weight function. This is an example of what the x data would look like after running it through the weight function using a Tricubic kernel. 

```c
scale = StandardScaler()
def weight_function(u,v,kern=Gaussian,tau=0.5):
    return kern(cdist(u, v, metric='euclidean')/(2*tau))
W = weight_function(scale.fit_transform(x),scale.fit_transform(x),kern=Tricubic,tau=0.3)
W
```
<img width="628" alt="Screenshot 2024-02-25 at 7 49 44 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/0566f1ce-dfed-44d4-bf14-5b17c4a9f6f4">

Let's try creating and running this model. 

```c
model = Lowess(kernel=Epanechnikov,tau=0.05)
xscaled = scale.fit_transform(x)
model.fit(xscaled,y)
yhat = model.predict(xscaled)
yhat
```
<img width="198" alt="Screenshot 2024-02-25 at 7 49 59 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/207ae41c-815b-44eb-a47d-2624d343ca96">

```c
mse(yhat, y)
```
<img width="163" alt="Screenshot 2024-02-25 at 7 50 15 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/1bb6de6c-af5d-491e-9cd0-dd6982898893">

That's a pretty good MSE for this model. Let's try with the other kernels and see what we get before we move into 10-fold Validation. 

```c
model = Lowess(kernel=Tricubic,tau=0.05)
xscaled = scale.fit_transform(x)
model.fit(xscaled,y)
yhat = model.predict(xscaled)
mse(yhat,y)
```
<img width="172" alt="Screenshot 2024-02-25 at 7 51 20 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/d4f05c22-dc30-443a-a82a-ba8be7a1a18e">

```c
model = Lowess(kernel=Quartic,tau=0.05)
xscaled = scale.fit_transform(x)
model.fit(xscaled,y)
yhat = model.predict(xscaled)
mse(yhat,y)
```
<img width="167" alt="Screenshot 2024-02-25 at 7 52 22 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/80d59737-95e5-480c-a6fc-77a723e1b503">

```c
model = Lowess(kernel=Gaussian,tau=0.05)
xscaled = scale.fit_transform(x)
model.fit(xscaled,y)
yhat = model.predict(xscaled)
mse(yhat,y)
```
<img width="153" alt="Screenshot 2024-02-25 at 7 52 36 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/b5f102bc-dea2-4679-86ee-93d9bcb67129">

It seems like the Quartic kernel is the best performing kernel (has the lowest MSE). Because of this, I will use the Quartic kernel in the rest of this question. 

## 10-Fold Validation 

Next, using the Quartic kernel, I tried to do 10 fold validation using three different scalers. 

### MinMax Scaler

```c
scale = MinMaxScaler()
mse_lwr = []
mse_rf = []
kf = KFold(n_splits=10,shuffle=True,random_state=1234)
model_lw = Lowess(kernel= Quartic,tau=0.14)

for idxtrain, idxtest in kf.split(x):
  xtrain = x[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  model_lw.fit(xtrain,ytrain)
  yhat_lw = model_lw.predict(xtest)

  mse_lwr.append(mse(ytest,yhat_lw))
print('The Cross-validated Mean Squared Error for Locally Weighted Regression is : '+str(np.mean(mse_lwr)))
```
<img width="795" alt="Screenshot 2024-02-25 at 9 40 20 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/bb2321e6-a16e-4f54-97ec-c680fcd752d2">

### Standard Scaler








