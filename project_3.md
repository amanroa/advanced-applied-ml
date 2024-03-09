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

Next, I wrote some code that created the SCAD model and fit the data to it over 100 epochs. In hindsight, I realize that I could have made a `fit()` method within the SCAD class, and this would have been more streamlined. 

```c
model = LinearSCAD(input_dim=8)
model.to(dtype = torch.float64)

optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100

ytrain_torch_unsqueezed = ytrain_torch.unsqueeze(1)
ytest_torch_unsqueezed = ytest_torch.unsqueeze(1)

for e in range(num_epochs):
  optimizer.zero_grad()
  loss = model.scad_loss(xtrain_torch, ytrain_torch_unsqueezed)
  loss.backward()
  optimizer.step()

  if e % 10 == 0:
    print("Loss at", e, "epoch:", loss.item())
```
When running this code, my loss decreased greatly from around 21,000 to 223. 

<img width="318" alt="Screenshot 2024-03-08 at 10 19 30 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/00187954-c54f-477e-b589-830ecd02fa12">

But to get a real sense of how accurate this model is, I calculated the R^2 value using the sklearn library.

```c
y_pred = model(xtest_torch).detach().numpy()
true_values = ytest_torch_unsqueezed.detach().numpy()
r2 = R2(true_values, y_pred)
```
The R^2 value was quite low, at 0.21975052241838167. This means that this model is not great at predicting the strength of the concrete given all of the possible features. Some of these features may be less helpful to the model. To figure this out, I looked at the coefficients of each of the features, and graphed the absolute value of them on a bar chart. 

```c
coefficients = model.linear.weight.detach().numpy()
features = ["cement", "slag", "ash", "water", "superplastic", "coarseagg", "fineagg", "age"]
plt.bar(features, abs(coefficients.flatten()))  # Use absolute value to consider magnitude
plt.xticks(rotation=45)
plt.ylabel('Importance')
plt.title('Feature Importance according to SCAD')
plt.show()
```
The coefficients themselves were: [[ 0.07156901  0.01445613 -0.00761952  0.22866092  0.60831127 -0.03531219   0.0012781   0.06330673]], and the graph of importance looked like this. It's important to note that features with coefficients closer to 0 are less important. So the water and superplastic seem to be the most important, with cement, age, and coarseagg potentially being features that we could add to the model if need be. 

<img width="565" alt="Screenshot 2024-03-08 at 10 40 14 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/03b21c55-a7d8-4c12-aae8-8b622e3cdd4e">

Interestingly, when I modified my model to include only those features, my R^2 became negative, with it being -0.9558600377736646 after adding all of the features whos absolute values were greater than 0.02 (all of the features mentioned above). I'm not too sure as to why this happened - maybe my model was overfitted or underfitted. 

This is how I created my SCAD model. Now, I will compare it to the ElasticNet and SqrtLasso models. 

## Part 2: ElasticNet, SqrtLasso and SCAD Comparison

The ElasticNet and SqrtLasso classes that I used were the same classes defined in the notebook titled 'Variable_Selection_and_Regularization_Introducation'. For redundancy and to make this notebook easier to read, I will not paste them here.

From that same notebook, I used the `make_correlated_features` method to make the 500 datasets with a correlation of 0.9. 

```c
def make_correlated_features(num_samples,p,rho):
  vcor = []
  for i in range(p):
    vcor.append(rho**i)
  r = toeplitz(vcor)
  mu = np.repeat(0,p)
  x = np.random.multivariate_normal(mu, r, size=num_samples)
  return x

rho =0.9
p = 20
n = 500
vcor = []
for i in range(p):
  vcor.append(rho**i)

x = make_correlated_features(n,p,rho)
```

I did not change the amount of features (20) from the notebook that we used in class, as I feel like having more features could reduce the reliance on one or two specific features (as I suspect is happening with the concrete dataset which only has 8 features). So with that made, I created the betastar array and the noise array. After making the x arrays, the betastar array, and the noise array, I performed the calculation of y = x * betastar + noise. 

```c
beta =np.array([-1,2,3,0,0,0,0,2,-1,4])
beta = beta.reshape(-1,1)
betastar = np.concatenate([beta,np.repeat(0,p-len(beta)).reshape(-1,1)],axis=0)

noise_std = 0.5 
noise = np.random.normal(0, noise_std, size=(500, 1))
y = np.dot(x, betastar) + noise
```

Once I had both x and y created, I split them into training and testing sets, the same 80-20 split as before. I also converted them to tensors for pytorch. Finally, I was having some issues with the ytrain and ytest, so I added a few lines to make sure they were the correct shape. 

```c
xtrain, xtest, ytrain, ytest = tts(x, y, test_size=0.2,shuffle=True,random_state=123)
xtrain_torch = torch.from_numpy(xtrain).to(dtype = torch.float64)
xtest_torch = torch.from_numpy(xtest).to(dtype = torch.float64)
ytrain_torch = torch.from_numpy(ytrain).to(dtype = torch.float64)
ytest_torch = torch.from_numpy(ytest).to(dtype = torch.float64)

ytrain_torch_unsqueezed = ytrain_torch.unsqueeze(1)
ytest_torch_unsqueezed = ytrain_torch.unsqueeze(1)
```

### SqrtLasso

First, I created the SqrtLasso model and fit the xtrain and ytrain data to it. Next, I got the predicted y values. Finally, I printed out the MSE. I got an MSE of 1.9689211475189672. I'll compare it with the other two models' MSEs to see how they match.

```c
sqrt_model = sqrtLasso(input_size=20)
sqrt_model.fit(xtrain_torch, ytrain_torch)
y_pred = sqrt_model.predict(xtest_torch)
mse_loss = nn.MSELoss()(y_pred, ytest_torch)
print("Test MSE:", mse_loss.item())
```
<img width="251" alt="Screenshot 2024-03-08 at 11 45 27 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/7ee78abe-e92c-4fa9-ba2d-74b0f1569ae3">

But more importantly, we should see if the betastar values match the original values. 

```c
betastar_torch = torch.from_numpy(betastar).to(dtype = torch.float64)
sqrt_betastar = sqrt_model.get_coefficients().detach() 
print("Sqrt Lasso betastar:", sqrt_betastar)
```
<img width="801" alt="Screenshot 2024-03-09 at 12 47 30 AM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/24d91900-bbce-42e2-9690-2d44804a3fc4">

I decided to create a graph of actual betastar values as compared to the model's betastar values.

<img width="997" alt="Screenshot 2024-03-09 at 12 48 15 AM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/ce8be07e-a825-42bb-9493-46714a666353">

This graph shows that SqrtLasso models have the trend of betastar values down well but it isn't exact. For example, the latter indicies don't have any betastar values, and the model predicted values that were close to 0 for those. And in the areas where the betastar values were higher, the model predicted higher values there too. But the values that it predicted for those coefficients that were less than 0 were not close to the actual values at all.

### ElasticNet

Next, I looked at ElasticNet. I defined a model, fitted it to the training data, predicted values, and calculated the MSE.

```c
elastic_model = ElasticNet(input_size=20).double()
elastic_model.fit(xtrain_torch, ytrain_torch)
y_pred = elastic_model.predict(xtest_torch)
mse_loss = nn.MSELoss()(y_pred, ytest_torch)
print("Test MSE:", mse_loss.item())
```
<img width="257" alt="Screenshot 2024-03-09 at 12 56 15 AM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/17819e26-e3ef-404a-9845-d652bcb8a140">

This MSE was slightly less than the SqrtLasso MSE, but not by a lot. Once again, I created the betastar values as seen in the class notebook, and tested them with this model.

```c
betastar_torch = torch.from_numpy(betastar).to(dtype = torch.float64)
elastic_betastar = elastic_model.get_coefficients().detach() 
print("elastic betastar:", elastic_betastar)
```
<img width="772" alt="Screenshot 2024-03-09 at 1 14 23 AM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/08d95372-c32b-44af-9c9d-ce2281584c68">

Now let's look at the graph.

<img width="995" alt="Screenshot 2024-03-09 at 1 14 50 AM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/c08428c2-cd32-4d78-b57d-5594f4277c13">

Interestingly, this model was also not able to get any betastar values that were negative. However, it performed slightly better in being more specific with it's predicted values. For example, indices 8 and 9 in the Sqrt Lasso model were very similar, despite their huge difference. But in this model, index 9 is higher than index 8. Also, the indices on the right side of the chart were closer to 0 (their original value) than their counterparts in the SqrtLasso model. 

### SCAD




