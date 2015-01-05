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



import numpy as np
import scipy as sp
from scipy.optimize import fmin_cg, fmin_bfgs

import pylab as pl
import os
import sys

from kernel_fn import *
from utils import *
#path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../util/'))
#if not path in sys.path:
#    sys.path.insert(1, path)
#del path
#import load_data as ld
#import multi_gauss_pdf as mgp
#from plot import *

'''Gaussian Process Regression (CPU version)
alpha, beta, noise - the hyper-parameters
N_train and N_test - the number of training and test examples
epsilon            - learning rates for finding optimal hyper-paramters'''
class Gaussian_Process_Reg():

    def __init__(self, alpha, beta, noise, epsilon):

        self.alpha = alpha
        self.beta  = beta 
        self.noise = noise
        self.epsilon = epsilon
 
    '''Computing Covariance Matrix of composition of two set (training and test 
        set) with a kernal. Default kernel is set to squared exponential kernel
        X1, X2 - training and test data 
        N , M  - number of training and test data points'''
    def compute_cov(self, X1, X2, kernel_name=None, flag=False, params=None):

        alpha, beta, noise = [self.alpha,self.beta,self.noise] 
        if  not params is None:
            alpha,beta,noise = list(params)

        K = X2.shape[0]
        Cov = np.zeros((X1.shape[0], K))

        for i in xrange(K):
            Cov[:,i] = kernel_sq_exp(X1, X2[i,:], alpha, beta) 

            if flag and np.array_equal(X1,X2):
                Cov[i,i] = Cov[i,i] + noise

        return Cov

    '''Returns mean and the diagonal portions of covariance matrix
    after predicting the curve''' 
    def predict(self, X1, y1, X2):
  
        cov_train = self.compute_cov(X1,X1, flag=True)
        cov_test  = self.compute_cov(X2,X2)       
        cov_tr_te = self.compute_cov(X1,X2)       
        cov_te_tr = cov_tr_te.T
    
        arg0  = np.linalg.inv(cov_train)
        arg1  = np.dot(cov_te_tr, arg0)
        mu    = np.dot(arg1,y1)
        #sigma = cov_test - np.dot(arg1, cov_tr_te) \
        #                + self.noise*np.eye(cov_test.shape[0])
        sigma = cov_test - np.dot(arg1, cov_tr_te) 

        return mu,sigma

    '''Optimizing hyper-parameter based. It uses conjugate gradient to
    optimize. (But can be extended to BFGS)'''
    def hyperparam_estimation(self, X, y, max_iter):

        params = np.asarray([self.alpha,self.beta,self.noise])
        arg = (X,y)
        print '...Optimizing Using CG'
        params = list(fmin_cg(self.get_loglikelihood, params, \
                    self._kernel_sq_exp_derivative, arg, maxiter=4))

        self.alpha,self.beta,self.noise = params   

        print 'Finale hyper parameters are alpha %f beta %f noise %f' % \
                                            (self.alpha,self.beta,self.noise)


    '''Computes derivative of squared exponential kernel function.
    This function returns the derivative of three hyper-parameters in
    a vector form [grad_alpha, grad_beta, grad_noise]'''
    def _kernel_sq_exp_derivative(self, params, X,y):

        alpha,beta,noise = params   
        N = X.shape[0]
        K = self.compute_cov(X,X, flag=True, params=params)
        L = np.linalg.cholesky(K)
        inv_K   = np.linalg.inv(K) 
        #inv_KY  = np.linalg.solve(K,y) #np.divide(K,y)
        inv_KY  = np.linalg.solve(L.T, np.linalg.solve(L,y)) #np.divide(K,y)
        inv_KYsq = np.outer(inv_KY, inv_KY) #TODO CHECK
        
        dK_dalpha = np.zeros((N,N))
        dK_dbeta  = np.zeros((N,N))
        dK_dnoise = np.zeros((N,N))
        for i in xrange(N):
            for j in xrange(N):
           
                dij = np.sqrt(np.sum((X[i,:] - X[j,:]) **2))
    
                #Derivative with respect to alpha
                dK_dalpha[i,j] = 2*np.sqrt(alpha)*np.exp(-0.5*(dij/beta)**2)
    
                #Derivative with respect to beta
                dK_dbeta[i,j]  = alpha * np.exp(-0.5*(dij/beta)**2)*dij**2/ beta**3
            
            dK_dnoise[i,i] = 2*np.sqrt(noise)
           
        dalpha = 0.5 * np.trace(np.dot((inv_KYsq - inv_K), dK_dalpha))
        dbeta  = 0.5 * np.trace(np.dot((inv_KYsq - inv_K), dK_dbeta))
        #dnoise = 0.5 * np.trace(np.dot((inv_KYsq - inv_K), dK_dnoise))
        dnoise = 0 
    
        #loglikelihood  = 0.5*(-np.dot(inv_KY, y.T) -  np.sum(np.diag(np.log(L))) \
        #                                            - N*np.log(2*np.pi)) 
    
        return -np.asarray([dalpha, dbeta, dnoise])
   

    '''Computes Log likelihood, so that we can optimize the hyper-parameters'''    
    def get_loglikelihood(self, params, X,y):
    
        alpha,beta,noise = params
        print params
        N = X.shape[0]
        K = self.compute_cov(X,X, flag=True, params=params)
        L = np.linalg.cholesky(K)
        inv_K   = np.linalg.inv(K) 
        #inv_KY  = np.linalg.solve(K,y) #np.divide(K,y)
        inv_KY  = np.linalg.solve(L.T, np.linalg.solve(L,y)) #np.divide(K,y)
        inv_KYsq = np.outer(inv_KY, inv_KY) #TODO CHECK
        loglikelihood  = -0.5*(-np.dot(inv_KY, y.T) -  np.sum(np.diag(np.log(L))) \
                                                    - N*np.log(2*np.pi)) 
    
        return loglikelihood 
    
        

