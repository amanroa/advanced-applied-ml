---
layout: page
title: "Project-5"
permalink: /project_5/
---
# Project 5 - Two Class Classification with KNN, SMOTE, ADASYN and FastKDE

To create an imbalanced dataset in class, I followed our in-class code to generate 10,000 datapoints, with 99% of them being 0 and 1% of them being 1. 

```c
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
counter = Counter(y)
print(counter)
```
<img width="228" alt="Screenshot 2024-04-21 at 8 10 07 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/15422dba-e908-4397-945b-c4fdb8be9264">

I also created a graph showing the distribution of the datapoints, differentiated by class label.

<img width="565" alt="Screenshot 2024-04-21 at 8 11 16 PM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/1d01d4ea-6987-4b5f-a8a3-508597530582">

Next, I created a KNN function using USearch. 

```c
class KNN:
  def __init__(self, vector1, vector2):
    self.vector1 = vector1
    self.vector2 = vector2

  def calc_dist(self):
    one_in_many: Matches = search(self.vector1, self.vector2, 1, MetricKind.L2sq, exact=True)
    self.one_in_many = one_in_many
    return one_in_many

  def show_list(self):
    return self.one_in_many.to_list()
  
  def predict(self, y_train):
    predictions = []
    neighbors = self.show_list()
    for n in neighbors:
      (index, distance) = n
      neighbor_indices = [index]  
      neighbor_labels = [y_train[index]]
      vote_counts = Counter(neighbor_labels)
      most_common_label, _ = vote_counts.most_common(1)[0]
      predictions.append(most_common_label)
    return(predictions)
```

There are a few things I want to note with my KNN implementation. First, I used some of my submission from Project 2. Second, I am not 100% familiar with the USearch library. I wasn't too sure how to implement the prediction method of KNN. Specifically, where to add the 'k'. This is because of the format of `one_in_many` and the `search()` function. In my other homework, I used 50 as an input to the `search()` function, so I repeated that here. It looks like it makes a 2D array that has a shape of (3000, 50). This is interesting, but I wasn't able to use this array to check my y_pred against my y_test, as they had different dimensions (y_test had a shape of (3000, 1)). In order to simplify things, I decided to use 1 as an input in the `search()` function. This might have impacted my ability to check for 'k' nearest neighbors. 

Even though my KNN implementation doesn't take a 'k' value, I wrote cross validation code because the question asked for it. 

```c
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, f1_score

k_values = range(1, 20)
k_scores = []

for k in k_values:
    knn = KNN(xtrain, xtest, k)
    scores = cross_val_score(knn, xtrain, ytrain, cv=5, scoring=make_scorer(accuracy_score))
    k_scores.append(scores.mean())

best_k = k_values[np.argmax(k_scores)]
print("Best k:", best_k, "with Cross-Validated Accuracy:", max(k_scores))
```

Here is how I ran my KNN method without the k value. 

```c
xtrain, xtest, ytrain, ytest = tts(X,y,test_size=0.3,shuffle=True,random_state=123)
d = KNN(xtrain, xtest)
output = d.calc_dist()
y_pred = d.predict(ytrain)
np.mean(y_pred == ytest)
```

This gives me an output of:

<img width="168" alt="Screenshot 2024-04-22 at 12 55 46 AM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/ff010757-183e-478f-a87a-9c8fab5eb2f3">

This is a very high accuracy. It might have to do with my model overfitting, or if I didn't implement KNN correctly. However, I decided to keep going and try to implement SMOTE, ADASYN and FastKDE. 

## SMOTE

This is the code where I implemented SMOTE, and a graph of the changed values. We can see that there are an equal amount of 1's as compared to 0's.

```c
oversample = SMOTE()
Xs, ys = oversample.fit_resample(X, y)
counter = Counter(ys)
print(counter)
for label, _ in counter.items():
	row_ix = np.where(ys == label)[0]
	plt.scatter(Xs[row_ix, 0], Xs[row_ix, 1], label=str(label),alpha=0.4)
plt.legend()
plt.show()
```
<img width="578" alt="Screenshot 2024-04-22 at 12 57 54 AM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/82fe7994-d18f-4262-9b19-f0c657b9666e">

I tried 2 ways to implement SMOTE. The first method involved resampling the entire x and y data, as shown above. And the second method involved just resampling the x_train and y_train data. I chose these two ways because I wasn't sure which one was the best option. 

First method: 

```c
oversample = SMOTE()
Xs, ys = oversample.fit_resample(X, y)
xtrain, xtest, ytrain, ytest = tts(Xs,ys,test_size=0.3,shuffle=True,random_state=123)
d = KNN(xtrain, xtest)
output = d.calc_dist()
y_pred = d.predict(ytrain)
np.mean(y_pred == ytest)
```
<img width="170" alt="Screenshot 2024-04-22 at 1 03 32 AM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/5914102c-e940-4c78-8062-1bd3170ee9e4">

Second method:

```c
xtrain, xtest, ytrain, ytest = tts(X,y,test_size=0.3,shuffle=True,random_state=123)
oversample = SMOTE()
X_res, y_res = oversample.fit_resample(xtrain, ytrain)
d = KNN(X_res, xtest)
output = d.calc_dist()
y_pred = d.predict(y_res)
np.mean(y_pred == ytest)
```
<img width="173" alt="Screenshot 2024-04-22 at 1 04 48 AM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/4a2e4dfc-10ed-43ea-9dae-5d0884abba1a">

Clearly, the first method came up with a higher accuracy. However, that accuracy is still less than the original accuracy produced by the KNN method. 

## ADASYN

I followed the same procedure for ADASYN. First, I made a plot to show the distribution, and how the amount of 1's was almost equal to the amount of 0's.

```c
ada = ADASYN(random_state=42)
Xs, ys = ada.fit_resample(X, y)
counter = Counter(ys)
print(counter)
for label, _ in counter.items():
	row_ix = np.where(ys == label)[0]
	plt.scatter(Xs[row_ix, 0], Xs[row_ix, 1], label=str(label),alpha=0.4)
plt.legend()
plt.show()
```
<img width="566" alt="Screenshot 2024-04-22 at 1 07 11 AM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/ea4977ae-da25-4ef2-a98c-c793558c2e55">

I then repeated the two methods I did with SMOTE. 

First method:

```c
oversample = ADASYN(random_state=42)
Xs, ys = oversample.fit_resample(X, y)
xtrain, xtest, ytrain, ytest = tts(Xs,ys,test_size=0.3,shuffle=True,random_state=123)
d = KNN(xtrain, xtest)
output = d.calc_dist()
y_pred = d.predict(ytrain)
np.mean(y_pred == ytest)
```
<img width="164" alt="Screenshot 2024-04-22 at 1 09 22 AM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/b88b530d-a7fc-4b5b-8bb7-047ce377b795">

Second method:
```c
xtrain, xtest, ytrain, ytest = tts(X,y,test_size=0.3,shuffle=True,random_state=123)
oversample = ADASYN(random_state=42)
X_res, y_res = oversample.fit_resample(xtrain, ytrain)
d = KNN(X_res, xtest)
output = d.calc_dist()
y_pred = d.predict(y_res)
np.mean(y_pred == ytest)
```
<img width="159" alt="Screenshot 2024-04-22 at 1 10 40 AM" src="https://github.com/amanroa/advanced-applied-ml/assets/26678552/cde8bdfe-5e93-4c0e-bf89-534f7712f3b8">

Interestingly, ADASYN gave us very similar results to SMOTE. The first method looks to have an identical accuracy, but that could be due to my misimplementation of the KNN class.

## FastKDE


## Question 2 - Final Project Information

My final project will be titled: 'Identifying Bird Species Based on Their Song'. I will use this dataset: [https://www.kaggle.com/datasets/vinayshanbhag/bird-song-data-set]. I chose this dataset because it only contains five bird species, which is a manageable amount. If possible, I will also try to use this dataset: [https://www.kaggle.com/competitions/birdclef-2024/data?select=train_metadata.csv], but that may be a bit aspirational as there are about 200 bird species in this dataset. 

In this project, I will aim to utilize sound recordings to identify which calls belong to which bird species. My main goal is to be able to correctly identify the bird species from their calls with an accuracy that is > 50%. I hope that this will be acheivable using Time Series models or other similar models. To map the audio into a format that the computer can understand it, I am also considering using spectograms. These are essentially maps that track noise, and show darker patterns where a noise is heard. I hope to create a model that can recognize these visual patterns and match them to a bird species! 

Example of a spectogram:

![image](https://github.com/amanroa/advanced-applied-ml/assets/26678552/be811836-09ba-46da-88e9-540bfdce9ded)












