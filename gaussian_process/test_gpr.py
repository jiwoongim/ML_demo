import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import matplotlib as mp
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp

import pylab as pl
import os
import sys


from gp_regression_cg import *
from gp_regression import *

theano.config.cmodule.warn_no_version = True
theano.config.DebugMode.warn_input_not_reused = False


'''Today data'''
def sample_poly3(numPoints):
    
    #INIT poly3 coef
    a = 1;
    b = -7;

    numPoints = numPoints/10;
    xL = range(1, 10*numPoints, 1) + [numPoints*10]
    yL = [0]*len(xL);
    tL = [0]*len(xL);
    xA = np.array(xL)/float(numPoints);

    for i in range(len(xL)):
        xd = xA[i];
        x = xd-3;
        epsilon = numpy.random.randn(1);
        tL[i] = 0.1*(a* pow(x,3) + b*x*x + x);
        yL[i] = tL[i] + epsilon[0];

       
    return [np.asarray([list(xA)]).T, np.asarray(yL), tL];

'''Today data'''
def a_cosb(numPoints):
    X = np.asarray([range(1,numPoints)])*0.01
    y = 3 + np.cos(3*X)
    return X.T,y.flatten()


'''Testing Gaussian Process Regression in Numpy Version''' 
def test_GPR_np(Xtrain,ytrain,Xtest,ytest,alpha, beta,noise,epsilon):

    gpr = Gaussian_Process_Reg(alpha, beta, noise, epsilon)
    gpr.hyperparam_estimation(Xtrain, ytrain, 100)
    mu, sigma = gpr.predict(Xtrain,ytrain, Xtest)
    pl.plot(Xtrain, ytrain, 'bo');
    pl.plot(Xtest , mu, 'ro');
    plt.show(block=True)

'''Testing Gaussian Process Regression in GPU (Theano) Version''' 
def test_GPR_theano(Xtrain,ytrain,Xtest,ytest,alpha, beta,noise,epsilon):
 
    X = T.fmatrix('X'); Y = T.fvector('Y'); Xt = T.fmatrix('Xt');
    NN = T.iscalar('NN'); MM = T.iscalar('MM')

    XtrainT = theano.shared(Xtrain)
    YtrainT = theano.shared(ytrain)
    XtestT  = theano.shared(Xtest)

    N = Xtrain.shape[0]; M = Xtest.shape[0]
    gpr_theano = GPR(alpha, beta, noise, N, M, epsilon)

    import pdb; pdb.set_trace()
    print '...GPR Prediction'   
    mu, sigma = gpr_theano.predict(X,Y, Xt)
    get_mean = theano.function([], mu, givens={
            X:XtrainT, Y:YtrainT, Xt:XtestT})
    mu = get_mean()    
        
    print '...Plotting'
    pl.plot(Xtrain, ytrain, 'bo');
    pl.plot(Xtest , mu, 'ro');
    plt.show(block=True)
 
def test_covariance_matrice(Xtrain,ytrain,Xtest,ytest,alpha, beta,noise,epsilon):
    
    ## THEANO VERSION ##
    X = T.fmatrix('X'); Y = T.fvector('Y'); Xt = T.fmatrix('Xt');
    NN = T.iscalar('NN'); MM = T.iscalar('MM')
    XtrainT = theano.shared(Xtrain)
    YtrainT = theano.shared(ytrain)
    XtestT  = theano.shared(Xtest)

    N = Xtrain.shape[0]; M = Xtest.shape[0]
    gpr_theano = GPR(alpha, beta, noise, epsilon, N, M)

    print '...Computing Covariance Matrix in theano'   
    cov = theano.function([NN,MM], gpr_theano.compute_cov(X, Xt, NN, MM),
                        givens={X:XtrainT, Xt:XtestT}) 
    K = cov(N,M) 
    cov = theano.function([NN,MM], gpr_theano.compute_cov(X, Xt, NN, MM),
                        givens={X:XtrainT, Xt:XtrainT}) 
    KK = cov(N,N)+ gpr_theano.noise**2 *np.eye(N)
    
    ## NUMPY VERSION ##
    print '...Computing Covariance Matrix in numpy'   
    gpr = Gaussian_Process_Reg(alpha, beta, noise, epsilon)
    K_np= gpr.compute_cov(Xtrain,Xtest)
    KK_np= gpr.compute_cov(Xtrain,Xtrain)

def test_inverse_of_covariance_on_p_y_f(Xtrain,ytrain,Xtest,ytest,alpha, beta,noise,epsilon):
    
    ## THEANO VERSION ##
    X = T.fmatrix('X'); Y = T.fvector('Y'); Xt = T.fmatrix('Xt');
    NN = T.iscalar('NN'); MM = T.iscalar('MM')
    XtrainT = theano.shared(Xtrain)
    YtrainT = theano.shared(ytrain)
    XtestT  = theano.shared(Xtest)

    N = Xtrain.shape[0]; M = Xtest.shape[0]
    gpr_theano = GPR(alpha, beta, noise, epsilon, N, M)

    cov_train = gpr_theano.compute_cov_s(XtrainT,N)              
    cov_test  = gpr_theano.compute_cov_s(XtestT,M)              
    cov_te_tr = gpr_theano.compute_cov(XtrainT,XtestT,N,M)      
    cov_tr_te = cov_te_tr.T                                

    arg0  = T.inv(cov_train+gpr_theano.noise**2 *T.identity_like(cov_train))

    ## NUMPY VERSION ##
    print '...Computing Covariance Matrix in numpy'   
    gpr = Gaussian_Process_Reg(alpha, beta, noise, epsilon)
    ncov_train = gpr.compute_cov(Xtrain,Xtrain)
    ncov_test  = gpr.compute_cov(Xtest,Xtest)       
    ncov_tr_te = gpr.compute_cov(Xtrain,Xtest)       
    ncov_te_tr = cov_tr_te.T

    arg00  = np.linalg.inv(ncov_train)

    import pdb; pdb.set_trace()

def main(Xtrain,ytrain,Xtest,ytest,alpha, beta,noise,epsilon):

    print "...Testing Gaussian Process Regression in Numpy Version"
    test_GPR_np(Xtrain,ytrain,Xtest,ytest,alpha, beta,noise,epsilon)
 
 
    print "...Testing Covariance Matrices in Theano Version"
    #test_covariance_matrice(Xtrain,ytrain,Xtest,ytest,alpha, beta,noise,epsilon)   

    #print "...Testing inverse of Covariance on P_y|f in Theano Version"
    #test_inverse_of_covariance_on_p_y_f(Xtrain,ytrain,Xtest,ytest,alpha, beta,noise,epsilon)

    #print "...Testing Gaussian Process Regression in GPU (Theano) Version"   
    #test_GPR_theano(Xtrain,ytrain,Xtest,ytest,alpha, beta,noise,epsilon)



if __name__ == '__main__':


    #Hyper-parameter
    alpha = 1.0
    beta  = 0.5 
    noise = 0.02
    epsilon = 0.0001
    num_examples = 300

    #X,y,z = sample_poly3(num_examples)
    X,y = a_cosb(num_examples)
    Xtrain = X[0:200,:].astype('float32')
    ytrain = y[0:200].astype('float32')
    Xtest  = X[200:300,:].astype('float32')
    ytest  = y[200:300].astype('float32')

    main(Xtrain,ytrain,Xtest,ytest,alpha,beta,noise,epsilon) 


