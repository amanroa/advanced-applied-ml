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
<img width="798" alt="Screenshot 2024-02-25 at 10 03 31 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/81572b33-6a07-4d37-8aba-7adc8d3800f6">


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

It seems like the Quartic kernel is the best performing kernel (has the lowest MSE). Next, I will move on to 10-Fold validation.

## 10-Fold Validation 

I tried to do 10 fold validation using three different scalers. Within each scaler, I tested all of the kernels. I did this in order to find the perfect combination of scaler and kernel that yields the lowest Cross validated MSE. To reduce redundancy, I will not paste my full code for most of the kernels, as it is the same each time - the only thing changing is the kernel name.

### MinMax Scaler

#### Quartic Kernel 

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

#### Gaussian Kernel
```c
model_lw = Lowess(kernel= Gaussian,tau=0.14)
```
<img width="807" alt="Screenshot 2024-02-25 at 9 54 32 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/3873081a-9b2a-44fb-b45f-ba48d9161d92">

#### Tricubic Kernel
```c
model_lw = Lowess(kernel= Tricubic,tau=0.14)
```
<img width="802" alt="Screenshot 2024-02-25 at 10 11 35 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/beb2ec3c-bab9-4d18-a2c5-89a8aa15a429">


#### Epanechnikov Kernel
```c
model_lw = Lowess(kernel= Epanechnikov,tau=0.14)
```
<img width="792" alt="Screenshot 2024-02-25 at 9 57 13 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/7ef7ba6a-d38e-4d26-a24b-0d6c4b6c69c7">

### Standard Scaler

#### Quartic Kernel 
```c
scale = StandardScaler()
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
<img width="800" alt="Screenshot 2024-02-25 at 9 58 10 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/024ab7bd-a781-46ee-9a13-40b76cce61e4">


#### Gaussian Kernel
```c
model_lw = Lowess(kernel= Gaussian,tau=0.14)
```
<img width="824" alt="Screenshot 2024-02-25 at 9 58 49 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/9fe4e03a-0171-4137-b8a4-081fd828cfee">


#### Tricubic Kernel
```c
model_lw = Lowess(kernel= Tricubic,tau=0.14)
```
<img width="797" alt="Screenshot 2024-02-25 at 9 59 00 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/5c8c0182-f441-4c89-afdb-cf92d950436c">


#### Epanechnikov Kernel
```c
model_lw = Lowess(kernel= Epanechnikov,tau=0.14)
```
<img width="809" alt="Screenshot 2024-02-25 at 9 59 15 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/fe9a91f4-bbe7-4cba-bfc8-2c7ee1f58546">

### Quantile Scaler

#### Quartic Kernel

```c
scale = QuantileTransformer()
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
<img width="809" alt="Screenshot 2024-02-25 at 10 05 38 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/b83da444-d2ef-42f9-ba35-0cd6f70c6fa1">

#### Gaussian Kernel
```c
model_lw = Lowess(kernel= Gaussian,tau=0.14)
```
<img width="798" alt="Screenshot 2024-02-25 at 10 07 01 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/6c5fe9fa-45d7-408e-b7a6-b3377db1bf15">

#### Tricubic Kernel
```c
model_lw = Lowess(kernel= Tricubic,tau=0.14)
```
<img width="810" alt="Screenshot 2024-02-25 at 10 07 23 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/65b6f379-a9bd-4c2d-8329-4e448d4d8eea">

#### Epanechnikov Kernel
```c
model_lw = Lowess(kernel= Epanechnikov,tau=0.14)
```
<img width="796" alt="Screenshot 2024-02-25 at 10 07 46 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/646845e1-a909-4511-993a-67b6315edb85">

The results of all of this testing: The Quantile Scaler with the Gaussian kernel was the best performing (had the lowest MSE) with a cross validated MSE of 21.99. The overall worst performing Scaler was the Standard Scaler. Interestingly, the Quartile Scaler did not perform well, with the other kernels returning MSEs > 100 except for the Gaussian kernel. The MinMax Scaler's MSE was consistently in the 46 - 83 range. 

## XGBoost vs my model

In this section, I will be comparing my best model to the XGBoost model. In order to do this, I need to import the xgboost library and create a model. I will use my best model (Quantile Scaler with Gaussian kernel), and compare the two. 

```c
import xgboost
model_xgboost = xgboost.XGBRFRegressor(n_estimators=200,max_depth=7)
model_xgboost.fit(xtrain,ytrain)
mse(ytest,model_xgboost.predict(xtest))
```
<img width="167" alt="Screenshot 2024-02-25 at 10 25 47 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/8c2ee91b-7b78-4d08-8779-b1f31f44fe87">

```c
scale = QuantileTransformer()
model = Lowess(kernel=Gaussian,tau=0.05)
xscaled = scale.fit_transform(x)
model.fit(xscaled,y)
yhat = model.predict(xscaled)
mse(yhat,y)
```
<img width="165" alt="Screenshot 2024-02-25 at 10 27 14 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/c6022723-7ec8-40e6-b5ea-84b01239534b">

By tuning the hyperparameters of the scaler and the kernel, I was able to get an MSE that was much smaller than the one given by XGBoost. However, if I wanted to get the MSE even lower, I could decrease the tau value. For example, if I changed `tau = 0.01`, the MSE value decreased to 1.2911046017204297. 

## USearch 

For this section, I needed to create a KNN for each observation in the test set by measuring distances with the observations in the train set. I wasn't too sure how to do this, so I tried a couple different ways. But first, I had to import the USearch library.

```c
!pip install usearch
from usearch.index import search, MetricKind, Matches, BatchMatches
```

And just to remind us, this is how I split the test and train data:

```c
xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.3,shuffle=True,random_state=123)
```

So I first tried creating a Matches object with the xtrain and xtest data, and searching for the distances between them. 

```c
one_in_many: Matches = search(xtrain, xtest, 50, MetricKind.L2sq, exact=True)
one_in_many.to_list()
```
<img width="143" alt="Screenshot 2024-02-25 at 11 23 28 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/d22c21a8-9091-42d0-a996-c2110a2aad77">

Interestingly, I also found that using BatchMatches produces the same output. So, I printed out the distances found by the search function:

```c
one_in_many.distances
```

<img width="595" alt="Screenshot 2024-02-25 at 11 28 36 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/4e0b0ceb-5183-49a0-a526-52b0e97d521f">

If I were to create a class for this, it would look something like this. 

```c
class Distances:
   
  def __init__(self, vector1, vector2):
    self.vector1 = vector1
    self.vector2 = vector2
  
  def calc_dist(self):
    one_in_many: Matches = search(self.vector1, self.vector2, 50, MetricKind.L2sq, exact=True)
    self.one_in_many = one_in_many
    return one_in_many

  def show_list(self):
    print(self.one_in_many.to_list())
```
Here is an example of how to implement my code, as well as the output.

```c
d = Distances(xtrain, xtest)
output = d.calc_dist()
output.distances
```
<img width="588" alt="Screenshot 2024-02-25 at 11 38 20 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/4fc9ff50-7de3-414d-b679-82bffa05277d">

This is how I was able to create a class that uses USearch to calculate KNN!
