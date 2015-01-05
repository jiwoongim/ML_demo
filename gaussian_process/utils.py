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

import cPickle, gzip, numpy
import theano
import theano.tensor as T
import numpy as np 
import math


'''  assume to be diagonal without loss of much performance '''
def multi_gauss_var(feature_vector, mean_vector, covariance_matrix,log=True):

    dimension = mean_vector.size
    detDiagCovMatrix = numpy.sqrt(numpy.prod(numpy.diag(covariance_matrix)))

    #frac = (2*numpy.pi)**(-dimension/2.0) * (1/detDiagCovMatrix)
    frac = (2*numpy.pi)**(-dimension/2.0) * (1/detDiagCovMatrix)
    fprime = feature_vector - mean_vector
    fprime **= 2

    if log:
      logValue = -0.5*numpy.dot(fprime, 1/numpy.diag(covariance_matrix))
      logValue += numpy.log(frac)
      return logValue

    else:
      print "You should compute in Log Domain, might be unstable!"
      return frac * numpy.exp(-0.5*numpy.dot(fprime, 1/numpy.diag(covariance_matrix)))

'''  Compute multivariate Guassian with ful Covariance Matrix'''
def multi_gauss_cov(feature_vector, mean_vector, covariance_matrix,log=True):
    K = covariance_matrix.shape[0]

    #Init
    inverse_covariance = numpy.linalg.inv(covariance_matrix)
    #print 'minimum covariance value %f' % numpy.min(numpy.absolute(inverse_covariance))
    #print 'maximum covariance value %f' % numpy.max(numpy.absolute(inverse_covariance))

    #first_term = numpy.log(numpy.linalg.det(covariance_matrix) * (2 * numpy.pi) ** K)
    first_term = numpy.log(numpy.linalg.det(inverse_covariance) * (2 * numpy.pi) ** K)
    dist = feature_vector - mean_vector
    second_term = numpy.dot(dist.T, numpy.dot(inverse_covariance, dist))
   

    if first_term == -numpy.infty:
        import pdb; pdb.set_trace()

    return - 0.5 * ( -first_term + second_term )




