#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def LDA(X,y):
    Vector_m=[]
    fea=X.shape[1]
    targets=set(y)
    for target in targets:
        Vector_m.append(np.mean(X[y==target],axis=0))
    print Vector_m
    
    ws=np.zeros((fea,fea))
    for target, vm in zip (targets, Vector_m):
        scatter=np.zeros((fea,fea))
        for row in X[y==target]:
            row, vm=row.reshape(fea,1), vm.reshape(fea,1)
            scatter+=(row-vm).dot((row-vm).T)
        ws+=scatter
        
    mean1=Vector_m[0].reshape(fea,1)
    mean2=Vector_m[1].reshape(fea,1)
    mean_d=mean1-mean2
    LDA=np.linalg.inv(ws).dot(mean_d)
    return LDA