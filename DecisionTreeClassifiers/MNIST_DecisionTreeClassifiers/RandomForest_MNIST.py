#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

max_features = ['sqrt', 'log2']
criterion = ['gini', 'entropy']
n_estimators = [100,500,1000]
max_depth = [50,100]
best_acc = 0

for f in max_features:
    for c in criterion:
        for n in n_estimators:
            for d in max_depth:
                clf = RandomForestClassifier(n_estimators = n, max_features=f, criterion=c, max_depth=d)
                clf.fit(train_features_df, train_class_df)
                y_pred = clf.predict(valid_features_df)
                acc = metrics.accuracy_score(valid_class_df, y_pred)
                print("  n_estimators ",n," criterion: ",c," max_features: ",f," max_depth: ",d, "Accuracy: ",acc)
                if(acc>best_acc):
                    best_acc = acc
                    best_n = n
                    best_f = f
                    best_c = c
                    best_d = d

new_train_features = np.concatenate([train_features_df, valid_features_df], axis=0)
new_train_class = np.concatenate([train_class_df, valid_class_df], axis=0)

clf = RandomForestClassifier(n_estimators = best_n, max_features=best_f, criterion=best_c, max_depth=best_d)
clf.fit(new_train_features, new_train_class)
test_pred = clf.predict(test_features_df)
model_acc = metrics.accuracy_score(test_class_df, test_pred)
print("     Best_max_features: ", best_f, " best_criterion: ", best_c," best_n_estimators: ", best_n, " best_max_depth: ", best_d)
print("     Model_accuracy: ", model_acc)

