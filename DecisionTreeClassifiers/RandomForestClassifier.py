#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###RandomForest###
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os


# In[ ]:


for filename in os.listdir("../all_data/"):
    if(filename.startswith('train')):
        train_filename = filename;
        valid_filename = filename.replace('train', 'valid')
        test_filename = filename.replace('train', 'test')
        print('Set: ',filename.split("_", 1)[1])
        RandomForest(train_filename, valid_filename, test_filename)


# In[ ]:


def RandomForest(train_filename, valid_filename, test_filename):

  #Reading Train data
    train_df = pd.read_csv("../all_data/"+train_filename, header=None)
    valid_df = pd.read_csv("../all_data/"+valid_filename, header=None)
    test_df = pd.read_csv("../all_data/"+test_filename, header=None)

    train_features_df = train_df.iloc[:,:-1]
    train_class_df = train_df.iloc[:,-1]
    valid_features_df = valid_df.iloc[:,:-1]
    valid_class_df = valid_df.iloc[:,-1]
    test_features_df = test_df.iloc[:,:-1]
    test_class_df = test_df.iloc[:,-1]

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

    new_train_features = pd.concat([train_features_df, valid_features_df], axis=0)
    new_train_class = pd.concat([train_class_df, valid_class_df], axis=0)

    clf = RandomForestClassifier(n_estimators = best_n, max_features=best_f, criterion=best_c, max_depth=best_d)
    clf.fit(new_train_features, new_train_class)
    test_pred = clf.predict(test_features_df)
    model_acc = metrics.accuracy_score(test_class_df, test_pred)
    model_f1_score = metrics.f1_score(test_class_df, test_pred, pos_label=1)
    print("     Best_max_features: ", best_f, " best_criterion: ", best_c," best_n_estimators: ", best_n, " best_max_depth: ", best_d)
    print("     Model_accuracy: ", model_acc, 'F1 score: ',model_f1_score)


# In[ ]:




