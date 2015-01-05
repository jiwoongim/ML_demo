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
from scipy.stats import norm

import pylab as pl
import os
import sys

from kernel_fn import *
from likelihood_fn import *


TINY = 1e-5
'''Gaussian Process Classification (CPU version)
alpha, beta, noise - the hyper-parameters
epsilon            - learning rates for finding optimal hyper-paramters 
                     (though we do not tune the hyper-parameters in this version
                     but gp_regression_cg.py does'''

class Gaussian_Process_Classifiation():

    def __init__(self, alpha, beta, noise, epsilon):

        self.alpha = alpha
        self.beta  = beta 
        self.noise = noise
        self.epsilon = epsilon

    '''Computing Covariance Matrix of composition of two set (training and test 
        set) with a kernal. Default kernel is set to squared exponential kernel
        X1, X2 - training and test data 
        N , M  - number of training and test data points'''
    def compute_cov(self, X1, X2, kernel_name=None):

        K = X2.shape[0]
        Cov = np.zeros((X1.shape[0], K))

        for i in xrange(K):
            Cov[:,i] = kernel_sq_exp(X1, X2[i,:], self.alpha, self.beta) 

            if np.array_equal(X1,X2):
                Cov[i,i] = Cov[i,i] + self.noise**2

        return Cov

    '''Infer the posterior distribution p(f|x,y) using expectation propagation algorithm
    X,y             - Data and labels
    K               - Covariance matrix of data set
    likelihood_fn   - likelihood function such as 'logistic', 'probit', and 'erf'
                     Default is set as logstic function
    max_iter        - Maximum number of iteration, default is 10'''
    def infer_expectation_propagation(self, X, y, K, likelihood_fn='logistic', max_iter=10):

        #Initialize parameters
        N = X.shape[0]
        tilde_tau   = np.zeros((N,)) 
        tilde_v     = np.zeros((N,))
        m           = np.mean(X, axis=1) #np.zeros((N,)) 
        mu          = np.zeros((N,)) 
        sigma       = K
        hat_mu      = np.zeros((N,)) 
        hat_sigma   = np.zeros((N,)) 
        delta_tilde_tau = 0

        def compute_marginal_likelihood(tilde_tau, tilde_v, m, L, sigma, mu, y_g_x):
            #Compute the log marginal likelihood
            tau_min = 1./np.diag(sigma) - tilde_tau
            v_min   = 1./np.diag(sigma) - tilde_v
            p = tilde_v - m * tilde_tau
            qq = v_min  - m * tau_min 
           
            #Decomposition of log marginal likelihood
            A = np.sum(np.diag(L)) - np.log(y_g_x+TINY)  #First & Third term
            B = - np.dot(np.dot(p.T,sigma),p) * 0.5 
            C = np.dot(np.diag(sigma).T,p**2) *0.5 
            D = 0.5 * np.sum(np.nan_to_num(np.log(1 + tilde_tau / tau_min))) #Fourth term
            E = - np.dot(qq.T, (tilde_tau/tau_min * qq - 2*p)* np.diag(sigma)) * 0.5

            assert not np.isnan(A +B +C+D+E), 'log marginal likelihood is nan'
            if np.isnan(A+B+C+D+E):
                import pdb; pdb.set_trace()
            return -(A+B+C+D+E)

        count = 0 
        est_log_margin = 0
        old_est_log_margin = est_log_margin+1
        while abs(est_log_margin - old_est_log_margin) > TINY and count < max_iter:
            perms = np.random.permutation(N)
            for i in perms: #Iterate EP updates (in random order) over examples

                #Compute approximate cavity patameters v_-i and tau_-i
                #using equation q_-i(f_i) = N(f_i|u_-i, simga^2_-i)
                tau_min_i = 1/sigma[i,i] - tilde_tau[i]         
                v_min_i   = mu[i] /sigma[i,i] - tilde_v[i]

                #Compute the marginal moments hat_u_i and hat_sigma_i
                #which is p(y_i|f_i)
                if likelihood_fn=='probit':
                    z_i = (y[i] * v_min_i / tau_min_i) / np.sqrt(1+1/tau_min_i) 
                    #if tau_min_i > 0 else (y[i] * v_min_i / tau_min_i) # Numerical stability
                    y_g_x, log_grad_y, log_ggrad_y = probit(z_i,y[i])
                elif likelihood_fn=='logistic':
                    y_g_x, log_grad_y, log_ggrad_y = logistic(y[i] * v_min_i / tau_min_i, y[i])
                elif likelihood_fn=='erf':
                    s2 = np.sqrt(1+1/tau_min_i) \
                        if tau_min_i > 0 else 1 # Numerical stability
                    if np.isnan(s2):
                        import pdb; pdb.set_trace()
                    z_i = (y[i] * v_min_i / tau_min_i) / s2 
                    y_g_x, log_grad_y, log_ggrad_y = erf(z_i, y[i], s2)

                #Old site parameter
                old_delta = delta_tilde_tau
                old_tilde_tau= tilde_tau[i]

                #Update site parameters from tilde_f_i(x)
                tilde_tau[i] = - log_ggrad_y / (1+log_ggrad_y/tau_min_i)
                tilde_tau[i] = max(0.0, tilde_tau[i]) 
                old_tilde_v  = tilde_v[i]
                tilde_v[i]   = (log_grad_y - v_min_i/tau_min_i *log_ggrad_y) \
                                    / (1+log_ggrad_y/tau_min_i)

                #Update sigma and mu from q(f|X,y)
                s_is_iT = np.outer(sigma[:,i], sigma[:,i]) #Check if s_is_iT is squarematrix
                assert s_is_iT.shape == (N,N), 'Needs to be square matrix'
                old_sig = sigma
                dtt     = tilde_tau[i] - old_tilde_tau #Derivatives
                dtv     = (tilde_v[i] - old_tilde_v)
                sigma   = sigma - (dtt / (1 + dtt * sigma[i,i])) *s_is_iT
                mu      = mu - ((dtt / (1 + dtt * sigma[i,i])) * \
                            (mu[i] + sigma[i,i]* dtv)- dtv) * sigma[:,i]

            #Re-compute the approximate posterior parameters sigma and mu using q(f|X,y) 
            tilde_S_sqrt = np.sqrt(np.diag(tilde_tau))
            B = np.eye(N) + np.dot(np.dot(tilde_S_sqrt, K), tilde_S_sqrt) #B is positive definite matrix
            L = np.linalg.cholesky(B)
            V = np.linalg.solve(L.T, (np.diag(tilde_S_sqrt)*K))
            old_sigma = sigma; old_mu = mu
            sigma = K - np.dot(V.T, V)
            mu    = np.dot(sigma,tilde_v) 

            #Compute the log marginal likelihood               
            old_est_log_margin = est_log_margin
            est_log_margin = compute_marginal_likelihood\
                            (tilde_tau, tilde_v, m, L, sigma, mu, y_g_x)
            print 'Estimated Log Marginal Likelihood %f, Diff Sigma %f, Diff Mean %f'\
                    %  (est_log_margin, np.sum(abs(sigma - old_sigma)),np.sum(abs(mu - old_mu)))

            count += 1
            bookkeeping = [tilde_S_sqrt, tilde_v]

        return [mu, sigma, bookkeeping]


    '''Infering during the predicting y using expectation propagation
    X,y, Xtest      - Data, labels, and test data
    K               - Covariance matrix of data set
    bookkeeping     - variables computed from training inference procedure
    likelihood_fn   - likelihood function such as 'logistic', 'probit', and 'erf'
                     Default is set as logstic function'''
    def prediction_expecation_propagation(self, X, y, Xtest, K, bookkeeping, likelihood_fn='logistic'):

        N = X.shape[0]
        tilde_S_sqrt, tilde_v = bookkeeping
        cov_test  = self.compute_cov(Xtest,Xtest, self.noise)       
        cov_tr_te = self.compute_cov(X,Xtest, self.noise)       
        cov_te_tr = cov_tr_te.T

        #Initialization
        B = np.eye(N) + np.dot(np.dot(tilde_S_sqrt, K), tilde_S_sqrt) #B is positive definite matrix
        L = np.linalg.cholesky(B)
        A = np.linalg.solve(L, np.dot(np.dot(tilde_S_sqrt,K),  tilde_v))

        #E_q [f* | X,y, x*] 
        z = np.dot(tilde_S_sqrt, np.linalg.solve(L.T, A))
        f_te = np.dot(cov_te_tr, (tilde_v - z))
    
        #V_q [f* | X,y, x*]
        v = np.linalg.solve(L.T, np.dot(tilde_S_sqrt,cov_tr_te))
        V = cov_test - np.dot(v.T,v)
        V[V < 0] = 0


        if likelihood_fn=='logistic':
            pred = logistic(f_te)   
        elif likelihood_fn=='probit':
            pred = probit(f_te / np.sqrt(1+np.diag(V)))          
        elif likelihood_fn=='erf':
            pred = erf(f_te / np.sqrt(1+np.diag(V)))          

        return pred

    '''Inference on p(f|x,y)
            Inputs:
                K - covariance matrix
                y - target (1|-1)
                likelihood function -'''
    def infer_laplace(self, X, y, K, likelihood_fn='logistic'):

        #0. Initialize f 
        N = X.shape[0]
        f = np.zeros((N,))
        old_f = np.zeros((N,)) + TINY + 1

        #Repeat 
        while np.sum(abs(f-old_f)) > TINY:

            # Compute hessian of y given f
            if likelihood_fn == 'logistic':
                y_g_f,gy_g_f,W = logistic(-y * f, y) # W is a diagonal matrix
            elif likelihood_fn == 'erf':
                y_g_f,gy_g_f,W = erf(-y * f, y,1) # W is a diagonal matrix
            elif likelihood_fn == 'probit':
                y_g_f,gy_g_f,W = probit(-y * f, y) # W is a diagonal matrix

            # Do cholesky factorization
            W_half = np.sqrt(-W) 
            B = np.eye(N) + np.dot(np.dot(W_half, K), W_half) #B is positive definite matrix
            L = np.linalg.cholesky(B)

            # Compute new f
            b = np.dot(W, f) + gy_g_f
            term = np.linalg.solve(L, np.dot(np.dot(W_half, K), b))
            a = b - np.dot(W_half, np.linalg.solve(L.T, term))
            old_f = f
            f = np.dot(K,a)
            
            print 'Difference in the previous f and new f: %f' % np.sum(abs(f-old_f))

        # Compute the log marginal likelihood
        est_log_marginal_likelihood = - 0.5 * np.dot(a.T, f) + np.sum(np.log(y_g_f)) - np.sum(np.diag(np.log(L))) #np.log??
        print 'Estimated Log Marginal Likelihood %f ' %  est_log_marginal_likelihood

        bookkeeping = [W,gy_g_f, L]
        return f, est_log_marginal_likelihood, bookkeeping

    '''Computing GP Classifier, p(y*|X,y,x*) 
       Inference on E(f* | X,y,x*)
        Inputs:
            X     - training data
            y     - training target (1 | -1)
            Xtest - test data
            K     - covariance matrix
            f_mu  - mean of f
            bookkeeping - [W,gy_g_f, L] where
              
                W     - hessian  of log P(y|f)
                gy_g_f- gradiant of log P(y|f)
                L     - lower triangular matrix of I + W1/2 K W1/2'''
    def prediction_laplace(self, X, y, Xtest, f_mu, K, bookkeeping, likelihood_fn='logistic'):

        [W,gy_g_f, L] = bookkeeping
        cov_test  = self.compute_cov(Xtest,Xtest, self.noise)       
        cov_tr_te = self.compute_cov(X,Xtest, self.noise)       
        cov_te_tr = cov_tr_te.T

        # Compute the Average f on test data
        f_mu_test = np.dot(cov_te_tr, gy_g_f)

        # Compute the variance of f on test data
        W_half = np.sqrt(-W)
        v = np.linalg.solve(L, np.dot(W_half, cov_tr_te))
        V = cov_test - np.dot(v.T, v)

        # Prediction 
        #np.exp(mgp.multi_gauss_var(f, 0, self.noise*np.eye(self.D)))

        if likelihood_fn == 'logistic':
            pred = logistic(f_mu_test)   
        elif likelihood_fn == 'probit':
            pred = probit(f_te / np.sqrt(1+np.diag(V)))
        return pred


    def get_classification_err_rate(self, pred,ytest):
        tmp = np.zeros(ytest.shape)
        tmp[pred>0.5] =  1
        tmp[pred<0.5] = -1

        wrong_example = np.sum([tmp != ytest])
        error_rate = wrong_example / float(ytest.shape[0])
        return error_rate

    '''Laplace approximation based classification'''
    def classify(self, infer_method='laplace', likelihood_fn='logistic'):
       
        if infer_method=='laplace':
            f, est_log_marg_like, bookkeeping = self.infer_laplace(Xtrain, ytrain, K, likelihood_fn='logistic')
            pred     = gpc.prediction_laplace(Xtrain, ytrain, Xtest, f, K, bookkeeping, likelihood_fn='logistic')
            err_rate = gpc.get_classification_err_rate(pred,ytest)

        return err_rate

    '''Computes derivative of squared exponential kernel function 
    for expectation propagation algorithm.
    This function returns the derivative of three hyper-parameters in
    a vector form [grad_alpha, grad_beta, grad_noise]'''
    def _kernel_sq_exp_derivative_EP(self, X, y):

        #Initialization
        N = X.shape[0]
        K = self.compute_cov(X,X)
        dK_dalpha = np.zeros((N,N))
        dK_dbeta  = np.zeros((N,N))
        dK_dnoise = np.zeros((N,N))


        inv_K   = np.linalg.inv(K) 
        inv_KY  = np.linalg.solve(K,y) #np.divide(K,y)
        inv_KYsq = np.outer(inv_KY, inv_KY) #TODO CHECK
    
        # Inference using expectation propagation algo.
        [mu, sigma, bookkeeping] = self.infer_expectation_propagation\
                            (X, y, K, likelihood_fn='erf', max_iter=10)    
        tilde_S_sqrt, tilde_v = bookkeeping

        # Compute 
        B = np.eye(N) + np.dot(np.dot(tilde_S_sqrt, K), tilde_S_sqrt) #B is positive definite matrix
        L = np.linalg.cholesky(B)
    
        # Compute the derivative of the log mariginal likelihood with respect to K
        A = np.linalg.solve(L.T, np.dot(np.dot(tilde_S_sqrt,K),  tilde_v)) 
        z = np.dot(tilde_S_sqrt, np.linalg.solve(L, A))
        b = tilde_v - z
        arg =  np.outer(b,b.T) 
        assert arg.shape == (N,N), 'NEED TO BE A SQUARE MATRIX'
        R= arg - np.dot(tilde_S_sqrt, np.linalg.solve(L.T, np.linalg.solve(L, tilde_S_sqrt)))

        # Copmute the derivative of K with respect to hyperparameters
        for i in xrange(N):
            for j in xrange(N):
           
                dij = np.sqrt(np.sum((X[i,:] - X[j,:]) **2))

                #Derivative with respect to alpha
                dK_dalpha[i,j] = 2*np.sqrt(self.alpha)*np.exp(-0.5*(dij/self.beta)**2)

                #Derivative with respect to beta
                dK_dbeta[i,j]  = self.alpha * np.exp(-0.5*(dij/self.beta)**2)*dij**2/ self.beta**3
            
            dK_dnoise[i,i] = 2*np.sqrt(self.noise)
          
        dalpha = 0.5*np.trace(np.dot(R,dK_dalpha))
        dbeta  = 0.5*np.trace(np.dot(R,dK_dbeta))
        dnoise = 0#0.5*np.trace(np.dot(R,dK_dnoise))
 
        return dalpha, dbeta, dnoise
    
    '''Optimizing hyper-parameter by gradient descent'''
    def hyperparam_estimation(self, X, y, max_iter=50, infer_name='EP'):

        dalpha = 0; dbeta = 0; dnoise = 0
        for i in xrange(max_iter):
            if infer_name == 'EP':
                dalpha, dbeta, dnoise = \
                                self._kernel_sq_exp_derivative_EP(X, y)   

            self.alpha -= self.epsilon * dalpha
            self.beta  -= self.epsilon * dbeta
            self.noise -= self.epsilon * dnoise

            print '... %dth iteration: Optimal alpha, beta, and noise are %f, %f, %f' \
                                        % (i, self.alpha, self.beta, self.noise)       

        print '***Finale hyper parameters are alpha %f beta %f noise %f\n' % \
                                            (self.alpha,self.beta,self.noise)

