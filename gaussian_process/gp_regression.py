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

'''Any questions/commments email: imj@uoguelph.ca'''


import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.linalg.ops import Cholesky, A_Xinv_b

import numpy as np
import scipy as sp

import pylab as pl
import os
import sys

from kernel_fn import *
from utils import *


'''Gaussian Process Regression (GPU version + Theano module required)
alpha, beta, noise - the hyper-parameters
N_train and N_test - the number of training and test examples
epsilon            - learning rates for finding optimal hyper-paramters 
                     (though we do not tune the hyper-parameters in this version
                     but gp_regression_cg.py does'''
class GPR():

    def __init__(self, alpha, beta, noise, N_train, N_test, epsilon=None):

        self.alpha = alpha
        self.beta  = beta 
        self.noise = noise
        self.epsilon = epsilon
        self.N = N_train
        self.M = N_test


    '''Computing Covariance Matrix of composition of two set (training and test 
        set) with a kernal. Default kernel is set to squared exponential kernel
        X1, X2 - training and test data 
        N , M  - number of training and test data points'''
    def compute_cov(self, X1, X2, N, M, kernel_name='squared_exp'):


        def cov(i, Cov, X1, X2):
            return T.set_subtensor(Cov[i], kernel_fns(X1, X2[i], kernel_name,\
                                                        self.alpha, self.beta)) 

        results, updates = theano.scan(fn=cov, outputs_info=T.zeros((M,N)), \
                sequences=[T.arange(M)], non_sequences=[X1, X2])

        return results[-1]

    '''Computing Covariance Matrix of a set (training/test set) with a kernal
    default kernel is set to squared exponential kernel
    Default kernel is set to squared exponential kernel
        X1 - dataset 
        N  - number of data points'''
    def compute_cov_s(self, X1, N, kernel_name='squared_exp'):

        def cov(i, Cov, X1):
            return T.set_subtensor(Cov[i], kernel_fns(X1, X1[i], kernel_name, \
                                                        self.alpha, self.beta)) 

        results, updates = theano.scan(fn=cov, outputs_info=T.zeros((N,N)), \
                sequences=[T.arange(N)], non_sequences=[X1])

        return results[-1]

    '''Returns mean and the diagonal portions of covariance matrix
    after predicting the curve''' 
    def predict(self, X1, y1, X2):
   
        cov_train = self.compute_cov_s(X1,self.N)
        cov_test  = self.compute_cov_s(X2,self.M)
        cov_te_tr = self.compute_cov(X1,X2,self.N,self.M)     
        cov_tr_te = cov_te_tr.T

        arg0  = T.inv(cov_train+self.noise**2 *T.identity_like(cov_train))
        #arg0  = T.inv(cov_train)
        arg1  = T.dot(cov_te_tr, arg0)
        mu    = T.dot(arg1,y1)
        sigma = cov_test - T.dot(arg1, cov_tr_te) 

        return mu,T.diag(sigma)


    
