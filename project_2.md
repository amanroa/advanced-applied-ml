---
layout: page
title: "Project-2"
permalink: /project_2.md
---

# Project 2 - Aashni Manroa

In this project, I will be experimenting with Gradient Boosting. I will be using the concrete csv that we used in class. First, we have to import various packages and load the data. 

```
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

```
data = pd.read_csv('/concrete.csv')
data
```
<img width="676" alt="Screenshot 2024-02-25 at 7 01 39 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/10af192b-0923-4982-9d22-9dae9305a94c">
