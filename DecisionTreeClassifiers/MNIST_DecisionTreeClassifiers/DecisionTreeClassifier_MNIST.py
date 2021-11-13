#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import os


# In[ ]:


X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X = X / 255

# (60K: Train) and (10K: Test) 
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

X_train_data, X_valid_data = X_train[:50000], X_train[50000:]
Y_train_data, Y_valid_data = y_train[:50000], y_train[50000:]


# In[ ]:


train_features_df = X_train_data
train_class_df = Y_train_data
valid_features_df = X_valid_data
valid_class_df = Y_valid_data
test_features_df = X_test
test_class_df = y_test

criterion = ['gini', 'entropy']
splitter = ['best', 'random']
max_depth = [500,1000]
best_acc = 0

for c in criterion:
    for s in splitter:
        for d in max_depth:
            clf = DecisionTreeClassifier(criterion= c, splitter= s, max_depth=d)
            clf.fit(train_features_df, train_class_df)
            y_pred = clf.predict(valid_features_df)
            acc = metrics.accuracy_score(valid_class_df, y_pred)
            print("     Criterion: ",c," splitter:",s," depth: ",d," Accuracy: ",acc)
            if(acc>best_acc):
                best_acc = acc
                best_c = c
                best_s = s
                best_d = d

new_train_features = np.concatenate((train_features_df, valid_features_df), axis=0)
new_train_class = np.concatenate((train_class_df, valid_class_df), axis=0)

clf = DecisionTreeClassifier(criterion=best_c, splitter=best_s, max_depth=best_d)
clf.fit(new_train_features, new_train_class)
test_pred = clf.predict(test_features_df)
model_acc = metrics.accuracy_score(test_class_df, test_pred)
print("     best_criterion: ",best_c," best_splitter:",best_s," best_max_depth: ",best_d)
print("     Model_accuracy: ", model_acc)


# In[ ]:




