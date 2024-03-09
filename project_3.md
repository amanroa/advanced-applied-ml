---
layout: page
title: "Project-3"
permalink: /project_3/
---
# Project 3

In this project, I will investigate Smoothly Clipped Absolute Deviations (SCAD). The first part of this project involves creating the SCAD class and trying it on a dataset. The second part of the project involves comparing SCAD to SqrtLasso and ElasticNet to see which produces the best approximation of a solution. 

## Part 1: Creating a Class for SCAD Regularization for Linear Models 

I utilized the link given to us in the assignment to get started on my SCAD class. The most important part of the link were these two methods: 

```c
def scad_penalty(beta_hat, lambda_val, a_val):
    is_linear = (np.abs(beta_hat) <= lambda_val)
    is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
    is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
    linear_part = lambda_val * np.abs(beta_hat) * is_linear
    quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
    constant_part = (lambda_val**2 * (a + 1)) / 2 * is_constant
    return linear_part + quadratic_part + constant_part
    
def scad_derivative(beta_hat, lambda_val, a_val):
    return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))
```

I made sure to include these methods into my LinearSCAD class. However, I wasn't too sure if I needed to use `scad_derivative` because I am more familiar with built in optimization methods such as Adam. So, I ended up using that instead of the `scad_derivative` method. 

To get started with this project, we had to import many classes. Here are the import statements: 

```c
import torch 
import torch.nn as nn
from ignite.contrib.metrics.regression import R2Score
import torch.optim as optim
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.datasets import make_spd_matrix
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, r2_score as R2
from sklearn.model_selection import train_test_split as tts, KFold, GridSearchCV
```

I also chose to run my program on cpu, with a data type of float64. This is what that looked like in my code.

```c
device = torch.device("cpu")
dtype = torch.float64
```

Here is my LinearSCAD class, which uses SCAD regularization and variable selection for linear models. 

```c
class LinearSCAD(nn.Module):
    def __init__(self, input_dim, lambda_val=0.01, a_val=3.7):
        super(LinearSCAD, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  
        self.lambda_val = lambda_val
        self.a_val = a_val
    
    def scad_penalty(self, beta_hat, lambda_val, a_val):
      abs_beta = torch.abs(beta_hat)
      is_linear = (abs_beta <= lambda_val)
      is_quadratic = (lambda_val < abs_beta) & (abs_beta <= a_val * lambda_val)
      is_constant = (a_val * lambda_val) < abs_beta
      
      linear_part = lambda_val * abs_beta * is_linear.float()
      quadratic_part = (2 * a_val * lambda_val * abs_beta - beta_hat.pow(2) - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic.float()
      constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant.float()
      
      return linear_part + quadratic_part + constant_part
      
    def scad_derivative(self, beta_hat, lambda_val, a_val):
        return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))
    
    def forward(self, x):
        return self.linear(x)
    
    def scad_loss(self, x, y):
        mse_loss = nn.MSELoss()(self.forward(x), y)
        scad_penalty_value = self.scad_penalty(self.linear.weight, self.lambda_val, self.a_val).sum()
        return mse_loss + scad_penalty_value

```

To test this class, I used the concrete dataset that we have used previously. This dataset contains eight columns of features, ranging from the age to the amount of cement, and one y column for the strength of concrete. 

I chose to split the x and y data into training and testing sets. I also chose a 80-20 split for training and testing. Additionally, I had to convert the data (which was in a numpy array data type) into torch tensors. 

```c
x = data.loc[:,'cement':'age'].values
y = data['strength'].values

xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.2,shuffle=True,random_state=123)

# Converting to torch tensors
xtrain_torch = torch.from_numpy(xtrain).to(dtype = torch.float64)
xtest_torch = torch.from_numpy(xtest).to(dtype = torch.float64)
ytrain_torch = torch.from_numpy(ytrain).to(dtype = torch.float64)
ytest_torch = torch.from_numpy(ytest).to(dtype = torch.float64)
```
