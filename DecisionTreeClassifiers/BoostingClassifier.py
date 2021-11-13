#!/usr/bin/env python
# coding: utf-8

# In[1]:


###Boosting###
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os


# In[3]:


for filename in os.listdir("../all_data/"):
    if(filename.startswith('train')):
        train_filename = filename;
        valid_filename = filename.replace('train', 'valid')
        test_filename = filename.replace('train', 'test')
        print('Set: ',filename.split("_", 1)[1])
        Boosting(train_filename, valid_filename, test_filename)


# In[2]:


def Boosting(train_filename, valid_filename, test_filename):

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

    loss = ['deviance', 'exponential']
    criterion = ['friedman_mse', 'mse']
    n_estimators = [100,200]
    learning_rate = [0.1, 0.5]

    best_acc = 0

    for l in loss:
        for n in n_estimators:
            for c in criterion:
                for lr in learning_rate:
                    clf = GradientBoostingClassifier(loss=l, n_estimators=n, criterion=c, learning_rate=lr)
                    clf.fit(train_features_df, train_class_df)
                    y_pred = clf.predict(valid_features_df)
                    acc = metrics.accuracy_score(valid_class_df, y_pred)
                    print(" loss: ",l," n_estimators: ",n," criterion: ",c," learning_rate: ",lr," Accuracy: ",acc)
                    if(acc>best_acc):
                        best_acc = acc
                        best_l = l
                        best_n = n
                        best_c = c
                        best_lr = lr

    new_train_features = pd.concat([train_features_df, valid_features_df], axis=0)
    new_train_class = pd.concat([train_class_df, valid_class_df], axis=0)

    clf = GradientBoostingClassifier(loss=best_l, n_estimators=best_n, criterion=best_c, learning_rate=best_lr)
    clf.fit(new_train_features, new_train_class)
    test_pred = clf.predict(test_features_df)
    model_acc = metrics.accuracy_score(test_class_df, test_pred)
    model_f1_score = metrics.f1_score(test_class_df, test_pred, pos_label=1)
    print("     best_loss: ",best_l,"best_n_estimators: ",best_n," best_criterion: ",best_c," best_learning_rate: ",best_lr)
    print("     Model_accuracy: ", model_acc, 'F1 score: ',model_f1_score)
    print('')


# In[ ]:




