#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os


# In[6]:


for filename in os.listdir("../all_data/"):
    if(filename.startswith('train')):
        train_filename = filename;
        valid_filename = filename.replace('train', 'valid')
        test_filename = filename.replace('train', 'test')
        print('Set: ',filename.split("_", 1)[1])
        DecisionTree(train_filename, valid_filename, test_filename)


# In[5]:


def DecisionTree(train_filename, valid_filename, test_filename):

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

    criterion = ['gini', 'entropy']
    splitter = ['best', 'random']
    max_depth = [500,1000]
    max_features = ['sqrt', 'log2']
    best_acc = 0

    for c in criterion:
        for s in splitter:
            for d in max_depth:
                for f in max_features:
                    clf = DecisionTreeClassifier(criterion= c, splitter= s, max_depth=d, max_features=f)
                    clf.fit(train_features_df, train_class_df)
                    y_pred = clf.predict(valid_features_df)
                    acc = metrics.accuracy_score(valid_class_df, y_pred)
                    print(" Criterion: ",c," splitter:",s," depth: ",d," max_features: ",f," Accuracy: ",acc)
                    if(acc>best_acc):
                        best_acc = acc
                        best_c = c
                        best_s = s
                        best_d = d
                        best_f = f

    new_train_features = pd.concat([train_features_df, valid_features_df], axis=0)
    new_train_class = pd.concat([train_class_df, valid_class_df], axis=0)

    clf = DecisionTreeClassifier(criterion=best_c, splitter=best_s, max_depth=best_d, max_features=best_f)
    clf.fit(new_train_features, new_train_class)
    test_pred = clf.predict(test_features_df)
    model_acc = metrics.accuracy_score(test_class_df, test_pred)
    model_f1_score = metrics.f1_score(test_class_df, test_pred, pos_label=1)
    print(" best_criterion: ",best_c," best_splitter:",best_s," best_depth: ",best_d," best_max_features: ",best_f)
    print("     Model_accuracy: ", model_acc, 'F1 score: ',model_f1_score)


# In[ ]:




