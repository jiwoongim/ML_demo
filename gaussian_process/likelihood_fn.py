''' Version 1.000

 Code provided by Daniel Jiwoong Im 

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
from scipy.stats import norm

import pylab as pl
import os
import sys

'''Returns the value of sigmoid function, gradient of sigmoid function, 
    and hessian of sigmoid function'''
def logistic(x,y=None):
 
    y_g_x  = 1.0 /(1+np.exp(-x))

    if y==None:
        return y_g_x

    log_grad_y = y-y_g_x
    if isinstance(x, np.ndarray):
        log_hess_y = np.diag(- y_g_x * (1-y_g_x))
    else:
        log_hess_y = - y_g_x * (1-y_g_x)
    return y_g_x, log_grad_y, log_hess_y


'''Returns the value of probit function, gradient of probit function, 
    and hessian of probit function'''
def probit(x,y=None):

    norm_x  = norm.pdf(x)
    y_g_x   = norm.cdf(x)

    if y==None:
        return y_g_x

    if isinstance(x, np.ndarray):
        log_grad_y  = y*norm_x / y_g_x if y_g_x > 0 else np.zeros(x.shape[0])
        log_ggrad_y = np.diag(- norm_x**2 / y_g_x**2 - y*x*norm_x/ y_g_x  if y_g_x > 0 else 0.0)
    else:
        log_grad_y  = y*norm_x / y_g_x if y_g_x > 0 else 0.0
        log_ggrad_y = - norm_x**2 / y_g_x**2 - y*x*norm_x/ y_g_x  if y_g_x > 0 else 0.0

    if np.isnan(log_ggrad_y) or np.isinf(log_ggrad_y):
        log_ggrad_y = 0 
    
    return y_g_x, log_grad_y, log_ggrad_y


'''Returns the value of erf function, gradient of erf function, 
    and hessian of erf function'''
def erf(x,y=None, s2=None):

    y_g_x = 0.5* (1 + sp.special.erf(x/np.sqrt(2)))

    if y==None:
        return y_g_x

    norm_x  = norm.pdf(x)
    log_grad_y  = y*norm_x/s2 
    log_ggrad_y = - norm_x * (x+norm_x) / s2

    assert not np.isnan(log_grad_y) , 'log_erf_grad is nan'
    assert not np.isnan(log_ggrad_y), 'log_erf_ggrad is nan'
    assert not np.isinf(log_grad_y) , 'log_erf_grad is inf'
    assert not np.isinf(log_ggrad_y), 'log_erf_ggrad is inf'   
    return y_g_x, log_grad_y, log_ggrad_y


