''' Version 1.000

 Code provided by Daniel Jiwoong Im 
 www.uoguelph.ca/~imj

 Permission is granted for anyone to copy, use, modify, or distribute this
 program and accompanying programs and documents for any purpose, provided
 this copyright notice is retained and prominently displayed, along with
 a note saying that the original programs are available from our
 web page.
 The programs and documents are distributed without any warranty, express or
 implied.  As the programs were written for research purposes only, they have
 not been tested to the degree that would be advisable in any important
 application.  All use of these programs is entirely at the user's own risk.'''





import numpy as np
import scipy as sp
import pylab as pl

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import sys

from gp_classification_np import *

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../util/'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
import load_data as ld


DATAPATH = '/mnt/data/datasets/oil/oil_flow_3classes.pickle'


def load_oil():
    data = ld.unpickle( DATAPATH )
    train_set = [data['DataTrn'], data['DataTrnLbls']]
    valid_set = [data['DataVdn'], data['DataVdnLbls']]
    test_set  = [data['DataTst'], data['DataTstLbls']]

    return train_set, valid_set, test_set

def binary_oil_data():
    train_set, valid_set, test_set = load_oil()
    
    
    #Eliminate third class
    N = train_set[1].shape[0]
    class3 = np.asarray([0., 0., 1.])
    index = np.sum(train_set[1] * np.kron(class3, np.ones((N,1))), axis=1) == 0
    train_set[1] = train_set[1][index]
    train_set[0] = train_set[0][index]
    new_label = np.zeros((train_set[0].shape[0],))
    new_label[train_set[1][:,0] == 1] = -1
    new_label[train_set[1][:,1] == 1] = 1
    train_set[1] = new_label


    index = np.sum(valid_set[1] * np.kron(class3, np.ones((N,1))), axis=1) == 0
    valid_set[0] = valid_set[0][index]
    valid_set[1] = valid_set[1][index]  
    new_label = np.zeros((valid_set[0].shape[0],))
    new_label[valid_set[1][:,0] == 1] = -1
    new_label[valid_set[1][:,1] == 1] = 1
    valid_set[1] = new_label

 
    index = np.sum( test_set[1] * np.kron(class3, np.ones((N,1))), axis=1) == 0
    test_set[0]  =  test_set[0][index]
    test_set[1]  =  test_set[1][index]     
    new_label = np.zeros((test_set[0].shape[0],))
    new_label[test_set[1][:,0] == 1] = -1
    new_label[test_set[1][:,1] == 1] = 1
    test_set[1] = new_label

    return train_set, valid_set, test_set


def main(Xtrain,ytrain,Xtest,ytest,alpha, beta,noise,epsilon):

    gpc = Gaussian_Process_Classifiation(alpha, beta,noise,epsilon)
    #gpc.hyperparma_estimation(Xtrain, ytrain, 100)
    K   = gpc.compute_cov(Xtrain,Xtrain)

    '''GPC using laplace approximation'''
    #likelihood_fn = 'logistic'
    #f, est_log_marg_like, bookkeeping = gpc.infer_laplace(Xtrain, ytrain, K, likelihood_fn=likelihood_fn)
    #pred     = gpc.prediction_laplace(Xtrain, ytrain, Xtest, f, K, bookkeeping, likelihood_fn=likelihood_fn)
    #err_rate = gpc.get_classification_err_rate(pred,ytest)
    #print 'GPC using Laplace Approx. Error Rate %f' % err_rate

    '''GPC using expectation propagation'''
    likelihood_fn = 'erf'

    #print '... Start infering the posterior'
    K   = gpc.compute_cov(Xtrain,Xtrain)
    mu, sigma, bookkeeping = gpc.infer_expectation_propagation(Xtrain, ytrain, K, likelihood_fn=likelihood_fn)

    print '... Predicting'
    pred = gpc.prediction_expecation_propagation(Xtrain, ytrain, Xtest, K, bookkeeping, likelihood_fn=likelihood_fn)
    err_rate = gpc.get_classification_err_rate(pred,ytest)   
    print 'GPC using Expectation Propagation Error Rate %f' % err_rate



    print '... Start estimating hyperparameter'
    gpc.hyperparam_estimation(Xtrain, ytrain, 6)

    print '... Start infering the posterior'
    K   = gpc.compute_cov(Xtrain,Xtrain)
    mu, sigma, bookkeeping = gpc.infer_expectation_propagation(Xtrain, ytrain, K, likelihood_fn=likelihood_fn)

    print '... Predicting'
    pred = gpc.prediction_expecation_propagation(Xtrain, ytrain, Xtest, K, bookkeeping, likelihood_fn=likelihood_fn)
    err_rate = gpc.get_classification_err_rate(pred,ytest)   
    print 'GPC using Expectation Propagation Error Rate %f' % err_rate
    
    return err_rate, pred

if __name__ == '__main__':

    #Hyper-parameter
    alpha = 1.0
    beta  = 0.5 
    noise = 0.02
    epsilon = 0.00005

    train_set, valid_set, test_set = binary_oil_data()
    train_set[0] = np.concatenate((train_set[0], valid_set[0]), axis=0)
    train_set[1] = np.concatenate((train_set[1], valid_set[1]), axis=0)
    N = train_set[0].shape[0]
    perm = np.random.permutation(N)
    train_set[0] = train_set[0][perm,:]
    train_set[1] = train_set[1][perm]
    main(train_set[0],train_set[1],test_set[0],test_set[1],alpha,beta,noise,epsilon) 




