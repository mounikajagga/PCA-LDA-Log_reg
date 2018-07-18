# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 02:52:35 2017

@author: Dell
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import LeaveOneOut

class NN:
    
    syn0=[]
    syn1=[]
    alpha = 1;
    
    def train(self, X, y, iter=100, alpha=1):
        self.syn0 = (2*np.random.random((2,2)) - 1) * 0.00001
        self.syn1 = (2*np.random.random((2,1)) - 1) * 0.0001
        for j in xrange(iter):
            l1 = 1/(1+np.exp(-(np.dot(X,self.syn0))))
            l2 = 1/(1+np.exp(-(np.dot(l1,self.syn1))))
            l2_delta = (y.reshape([1, -1]).T - l2)*(l2.T.dot(1-l2))
            l1_delta = np.matmul(l2_delta, self.syn1.T).dot(l1.T.dot(1-l1))
            self.syn1 += self.alpha * l1.T.dot(l2_delta)
            self.syn0 += self.alpha * X.T.dot(l1_delta)
            
    def predict(self, X):
        l1 = 1/(1+np.exp(-(np.dot(X,self.syn0))))
        l2 = 1/(1+np.exp(-(np.dot(l1,self.syn1))))
        return (l2 >= 0.45) * 1

irisdata = datasets.load_iris()
data = np.delete(irisdata['data'][50:], [0, 1], axis=1)

lv = LeaveOneOut()
lv.get_n_splits(data)

labels = (irisdata.target[50:] - 1).ravel()

classifier = NN();

error = 0

#for i in range(data.shape[0]):
for train_index, test_index in lv.split(data):
    
    
    
    train_data = data[train_index]
    train_labels = labels[train_index]
    test_data = data[test_index]
    test_labels = labels[test_index]
    
    classifier.train(train_data, train_labels, iter=1000, alpha=1)
    
    error += 1 - (test_labels == classifier.predict(test_data))
    
print(float(error) / data.shape[0])