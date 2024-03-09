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

I made sure to include these methods into my LinearSCAD class. However, I wasn't too sure if I needed to use scad_derivative because ___. 

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
