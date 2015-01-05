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
import os
import sys

import theano
import theano.tensor as T


def kernel_fns(X,x2,name, alpha, beta):

    if name == 'squared_exp':
        return _kernel_sq_exp(X,x2, alpha, beta)


"""Squared exponential kernel in THEANO"""
def _kernel_sq_exp(X,x2, alpha, beta):
    arg = T.sqrt(T.sum((X - x2)**2, axis=1))
    return alpha * T.exp(-0.5*(arg/ beta)**2)

"""Squared exponential kernel in numpy"""   
def kernel_sq_exp(X, x2, alpha, beta):

    arg = np.sqrt(np.sum((X - x2)**2, axis=1))
    return alpha * np.exp(-0.5*(arg / beta)**2)

"""Computes derivative of squared exponential kernel function.
    This function returns log likelihood and the derivative of 
    three hyper-parameters in a vector form 
    [grad_alpha, grad_beta, grad_noise]"""
def kernel_sq_exp_derivative(params, model, X, y):

    alpha, beta, noise = params
    N = X.shape[0]
    K = model.compute_cov(X,X, flag=True)
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
    dnoise = 0.5 * np.trace(np.dot((inv_KYsq - inv_K), dK_dnoise))
    dnoise = 0 

    loglikelihood  = 0.5*(-np.dot(inv_KY, y.T) -  np.sum(np.diag(np.log(L))) \
                                                - N*np.log(2*np.pi)) 

    return loglikelihood, np.asarray([dalpha, dbeta, dnoise])

