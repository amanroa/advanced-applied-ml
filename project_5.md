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
