# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 01:03:32 2017

@author: Dell
"""

from sklearn import datasets
from sklearn.model_selection import LeaveOneOut
#importing the logistic regression module from logisticRegression.py
from LogisticRegression import LogisticRegression 

import numpy as np

irisdata = datasets.load_iris()
data = np.delete(irisdata['data'][50:], [0, 1], axis=1)

lv = LeaveOneOut()
lv.get_n_splits(data)

labels = irisdata.target[50:].ravel()

error = []


for train_index, test_index in lv.split(data):
    
    #creating object for logistic regression
    model = LogisticRegression()
    
    train_data = data[train_index]
    train_labels = labels[train_index]
    test_data = data[test_index]
    test_labels = labels[test_index]
  #training the logistic regression object  
    model.train(train_data, train_labels)
    
    error.append(1 - (test_labels == model.test(test_data, test_labels)[0]))
    
print(np.mean(error))