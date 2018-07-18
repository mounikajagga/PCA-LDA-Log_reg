# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 02:04:01 2017

@author: Dell
"""

import numpy as np, copy, math

class LogisticRegression:
    'Model class for Logistic Regression Model'
    
    def __init__(self):
        self.weights = None
        self.targets = None
        self.cost = None
        self.scaling = None

    def train(self, data, targets, iter = 100, step = 1, lamda = 0):
        assert (data.shape[0] == targets.shape[0]), "Number of samples in data and labels should match"
        data = copy.deepcopy(data)
        if self.scaling is None:
            self.scaling = np.zeros(data.shape[1])
        m, n = data.shape
        self.__scale_features(data.reshape(m*n), m, n)
        data = np.pad(data, ((0, 0), (1, 0)), mode='constant', constant_values=1)
        if self.targets is None:
            self.targets = np.unique(targets)
        if self.weights is None:
            self.weights = np.random.random((self.targets.size, data.shape[1]))
        if self.cost is None:
            self.cost = np.zeros((iter, self.targets.size + 1))
        for i in range(iter):
            predictions = data.dot(self.weights.transpose())
            predictions = np.array([self.__sigmoid(k) for k in predictions.reshape(predictions.size)]).reshape((data.shape[0], self.weights.shape[0]))
            self.cost[i][1:] = self.__gradientDescent(data, targets, predictions.transpose(), step, lamda)
            self.cost[i][0] = i+1
        self.cost = self.cost[0:i+1]
        
    def test(self, data, targets):
        assert (data.shape[0] == targets.shape[0]), "Number of samples in data and labels should match"
        data = copy.deepcopy(data)
        m, n = data.shape
        data = self.__scale_features(data.reshape(data.size), m, n)
        data = np.pad(data, ((0, 0), (1, 0)), mode='constant', constant_values=1)
        predictions = data.dot(self.weights.transpose())
        predictions = np.array([self.__sigmoid(k) for k in predictions.reshape(predictions.size)]).reshape((data.shape[0], self.weights.shape[0]))
        labels = [self.__classify(k, True) for k in predictions]
        return (labels, predictions)
    
    def __scale_features(self, data, m, n):
        for i in range(n):
            tmp = data[list(range(i, data.size, n))]
            if self.scaling[i] == 0:
                self.scaling[i] = max(tmp) - min(tmp)
            data[list(range(i, data.size, n))] = tmp / self.scaling[i]
        return data.reshape((m, n))
    
    def __sigmoid(self, val):
        return 1 / (1 + math.exp(-val))
    
    def __classify(self, targets, return_class = False):
        index = np.argmax(targets)
        if return_class:
            index = self.targets[index]
        return index
    
    def __gradientDescent(self, data, targets, predictions, step, lamda):
        target_y = np.zeros((targets.shape[0], 1))
        cost = np.zeros((1, self.targets.size))
        for i in range(self.targets.size):
            index = np.where(targets == self.targets[i])[0]
            target_y[index] = 1
            diff = predictions[i] - target_y.transpose();
            gradients = diff.dot(data)
            self.weights[i] = self.weights[i] * (1 - step * (lamda / target_y.size)) - (step * gradients / target_y.size)
            term1 = np.array([math.log(k) for k in predictions[i]])
            term2 = np.array([math.log(1-k) for k in predictions[i]])
            cost[0][i] = (target_y.transpose().dot(term1) + (1 - target_y).transpose().dot(term2)[0] + (lamda * sum(self.weights[i]**2))) / target_y.size
            target_y[index] = 0
        return cost
