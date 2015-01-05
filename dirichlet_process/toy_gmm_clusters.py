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
import matplotlib .pyplot as plt


'''Chinese restaurant process demo.
num_data_cluster - number of data points per cluster 
num_gaussian     - number of clusters
variance         - variance of the Gaussian
max/min_space    - data boundaries'''
def gen_gmm_cluster(num_data_cluster, num_gaussian, variances, max_space, min_space):

    data  = np.zeros((num_data_cluster * num_gaussian, 2))
    label = np.zeros((num_data_cluster * num_gaussian, ))
    means = np.zeros((num_gaussian, 2))

    for i in xrange(num_gaussian):

        mean_i = (max_space - min_space) * np.random.rand(2) + min_space
        samples = np.random.multivariate_normal(mean_i, np.diag(variances), num_data_cluster)

        data[i*num_data_cluster:(i+1)*num_data_cluster,:] = samples
        label[i*num_data_cluster:(i+1)*num_data_cluster] = i 
        means[i,:] = mean_i
        
    return data, label, means


def view_clusters(data, label, means):

    num_clusters = int(np.max(label)+1)
    plt.figure()

    for i in xrange(num_clusters):
        indice = np.argwhere(label == i)
        plt.scatter(data[indice,0], data[indice,1], color=(0,i/float(num_clusters),0,1))
    
    plt.show()

if __name__ == '__main__':

    num_data_cluster = 30
    num_gaussian     = 10
    variances        = np.asarray([.4, .4])
    max_space        = np.asarray([10, 10])
    min_space        = np.zeros((2,)).T

    data, label, means = gen_gmm_cluster(num_data_cluster, num_gaussian, variances, max_space, min_space)
    view_clusters(data, label, means)
    pass

