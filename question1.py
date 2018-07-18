# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:11:30 2017

@author: Dell
"""
import numpy as np
import do_PCA as pca
import pandas as pd
import LDA as lda 
import matplotlib.pyplot as plt


#importing data

data=pd.read_csv('dataset_1.csv')
data.head()
X=data.iloc[:,[0,1]]
X=np.array(X)
y=data.iloc[:,2]
pca_output=pca.do_PCA(X)

#plot the data

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(2, 1, 1)
ax.plot(X[0:30:,0], X[0:30,1], linestyle='None', marker='o', markersize=5, color='green')
ax.plot(X[30:,0],X[30:,1], linestyle='None', marker='o', markersize=7, color='blue')
ax.set_xlabel('V1')
ax.set_ylabel('V2')
ax.set_title('V1 vs V2')
fig.show()
fig.savefig('plot of features.jpg')


X_pca_projected=X.dot(pca_output['loadings'][0])

#plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(2, 1, 1)
ax.plot(X_pca_projected[0:30], np.zeros(30), linestyle='None', marker='o', markersize=5, color='orange')
ax.plot(X_pca_projected[30:],np.zeros(30), linestyle='None', marker='o', markersize=7, color='blue')
ax.set_xlabel('PC1')
ax.set_ylabel('')
ax.set_title('Projected of X onto PC1')
fig.show()
fig.savefig('Projected of X onto PC1')

X.shape[0]

y=np.array(y)
W=lda.LDA(X,y)
set(y)


X_Wproj=X.dot(W)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(2, 1, 1)
ax.plot(X_Wproj[0:30], np.zeros(30), linestyle='None', marker='o', markersize=5, color='orange')
ax.plot(X_Wproj[30:],np.zeros(30), linestyle='None', marker='o', markersize=7, color='blue')
ax.set_xlabel('PC1')
ax.set_ylabel('')
ax.set_title('Projected of X onto W')
fig.show()
fig.savefig('Projected of X onto W')

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(2, 1, 1)
ax.plot(X[0:30,0], X[0:30,1], linestyle='None', marker='o', markersize=5, color='orange')
ax.plot(X[30:,0], X[30:,1], linestyle='None', marker='o', markersize=7, color='blue')
#ax.plot(pca_output['loadings'][0,0],pca_output['loadings'][1,0])
k=30
ax.plot([0, (-1)*k*pca_output['loadings'][0,0]], [0, (-1)*k*pca_output['loadings'][1,0]],color='red', linewidth=3)
ax.plot([0, (1)*k*W[0]],[0,(1)*k*W[1]],color='green', linewidth=3)
ax.set_xlabel('V1')
ax.set_ylabel('V2')
ax.set_title('Projected of X onto PC1')
fig.show()
fig.savefig('Projected of X onto PC1')


#variance of X onto W
np.var(X_Wproj)
# 0.1007877130352357
# variance of x onto PCA  axes

var1=np.var(X_pca_projected)
var1
# 158.35671176837866

var2=np.var(X.dot(pca_output['loadings'][1]))
# 5.039862489564209




