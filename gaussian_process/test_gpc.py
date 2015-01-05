
import numpy as np
import scipy as sp
import pylab as pl

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import sys

from gp_classification_np import *


#data
def sample_data1(ax, numPoints):
    
    X = np.random.rand(3,numPoints);
    xtrain =X[:, :numPoints * 2/3];
    xt = X[:, numPoints *2/3 :];
    xMean = np.mean(xtrain, axis=1);

    xtrain.sort(axis=1);
    xt.sort(axis=1);
    x1 = xtrain[:,:(xtrain.shape[1]/2) ];
    x2 = xtrain[:,(xtrain.shape[1]/2) :];

    x = np.concatenate((x1, x2), axis=1);

    #visualizing the toy data
    ax.plot(x1[0,:], x1[1,:], x1[2,:], 'o', label='class1 train data');
    ax.plot(x2[0,:], x2[1,:], x2[2,:], 'o', label='class2 train data');
    ax.plot([xMean[0]], [xMean[1]], [xMean[2]], 'o', label='mean');

def sample_data_exp(ax, k, numPoints):

    # sampling points from standard exponential distribution
    X1 = np.random.standard_exponential((3, numPoints));
    X2 = k- np.random.standard_exponential((3, numPoints));
    Y1 = np.ones((numPoints,), dtype='int8')
    Y2 = -1*np.ones((numPoints,), dtype='int8')
    X  = np.concatenate((X1,X2), axis=1);
    Y  = np.concatenate((Y1,Y2), axis=1);

    x1  = X1[:,:X1.shape[1] * 2/3];
    xt1 = X1[:, X1.shape[1] * 2/3 :];
    x2  = X2[:,:X2.shape[1] * 2/3];
    xt2 = X2[:, X2.shape[1] * 2/3 :];
    y1  = Y1[:X1.shape[1] * 2/3];
    yt1 = Y1[X1.shape[1] * 2/3 :];
    y2  = Y2[:X2.shape[1] * 2/3];
    yt2 = Y2[X2.shape[1] * 2/3 :];

    x = np.concatenate((x1,x2), axis =1);
    Xt = np.concatenate((xt1, xt2), axis=1);
    y = np.concatenate((y1,y2), axis =1);
    yt = np.concatenate((yt1, yt2), axis=1);
    xMean = np.mean(X, axis=1);

    #visualizing the toy data
    #if (vis1 ==1):
    ax.plot(x1[0,:], x1[1,:], x1[2,:], 'o', label='class1 train data');
    ax.plot(x2[0,:], x2[1,:], x2[2,:], 'o', label='class2 train data');
    ax.plot([xMean[0]], [xMean[1]], [xMean[2]], 'o', label='mean');

    return x.T, y, Xt.T, yt, X 


def main(Xtrain,ytrain,Xtest,ytest,alpha, beta,noise,epsilon):

    gpc = Gaussian_Process_Classifiation(alpha, beta,noise,epsilon)
    K   = gpc.compute_cov(Xtrain,Xtrain)

    '''GPC using laplace approximation'''
    #likelihood_fn='logistic'
    likelihood_fn='logistic'
    f, est_log_marg_like, bookkeeping = gpc.infer_laplace(Xtrain, ytrain, K, likelihood_fn=likelihood_fn)
    pred     = gpc.prediction_laplace(Xtrain, ytrain, Xtest, f, K, bookkeeping, likelihood_fn=likelihood_fn)
    err_rate = gpc.get_classification_err_rate(pred,ytest)
    print '***GPC using Laplace approximator Error Rate %f\n' % err_rate


    '''GPC using expectation propagation'''
    likelihood_fn='erf'

    print '... Start estimating hyperparameter'
    gpc.hyperparam_estimation(Xtrain, ytrain, 6)

    print '... Start infering the posterior'
    K   = gpc.compute_cov(Xtrain,Xtrain)
    mu, sigma, bookkeeping = gpc.infer_expectation_propagation(Xtrain, ytrain, K, likelihood_fn=likelihood_fn)

    print '... Predicting'
    pred = gpc.prediction_expecation_propagation(Xtrain, ytrain, Xtest, K, bookkeeping, likelihood_fn=likelihood_fn)
    err_rate = gpc.get_classification_err_rate(pred,ytest)   
    print '***GPC using Expectation Propagation Error Rate %f' % err_rate


if __name__ == '__main__':


    #Hyper-parameter
    alpha = 1.0 
    beta  = 0.5
    noise = 0.02
    epsilon = 0.0002
    num_examples = 150

    #initializing
    fig = plt.figure(1)
    fig.clf()
    ax = Axes3D(fig);

    Xtrain, ytrain, Xtest, ytest, tmp = sample_data_exp(ax, 5, num_examples)
    main(Xtrain,ytrain,Xtest,ytest,alpha,beta,noise,epsilon) 


    #pl.show()



