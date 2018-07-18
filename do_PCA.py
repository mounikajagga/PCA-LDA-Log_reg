
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
#from sklearn.decomposition import PCA
from matplotlib.mlab import PCA

#PCA function
def do_PCA(data_frame):
    
    #calculating varinace between X and Y
    cov_mat = np.cov(data_frame.T)
    #print('Covariance Matrix  between X & Y\n',cov_mat)
    #calculating the Eigen values and Eigen vectors 
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    #print('eigen values and eigen vectors\n',eig_val,eig_vec)
    idx=eig_val.argsort()[::-1]
    eigen_values = eig_val[idx]
    eigen_Vectors = eig_vec[:,idx]
    pca_scores = np.matmul(data_frame, eigen_Vectors)
    #print('PCA Scores\n',pca_scores)
    #defining the pca results for plotting
    #splitting into a 2-dimentional matrix
    
    #matrix_w = np.hstack((eig_pairs[0][1].reshape(2,1)))
    #Calculating new Eigen Spaces
    
    #eig_space = matrix_w.dot(data_frame.T)
    #print('Transformed matrix\n',eig_space)
    results = {
                   'PC_variance': eigen_values,
                   'scores': pca_scores, 
                   'loadings': eigen_Vectors}
    return results