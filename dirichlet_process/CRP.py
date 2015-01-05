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
import matplotlib.pyplot as plt 
import os,sys
from toy_gmm_clusters import * 


from  multi_gauss_pdf import * 

'''Plot the tables'''
def plotTables(data, assigned_tables, current_num_tables, mean_tables):

    plt.figure()
    for table_i in xrange(current_num_tables):
        indice = np.argwhere(assigned_tables == table_i).flatten()
        plt.scatter(data[indice,0], data[indice,1], color=(0,table_i/float( current_num_tables), 0, 1))
        plt.scatter(mean_tables[table_i,0], mean_tables[table_i,1], 'bx')
    plt.show()

'''Chinese restaurant process demo.
alpha           - the hyper-parameters
init_variance   - initial variance value
init_num_tables - initial number of tables
max_epoch       - number of epoches '''
def fit_chinese_restaurant_process(data, alpha, init_variances, init_num_tables, max_epoch):

    N,D = data.shape 
    current_num_tables = init_num_tables 
    assigned_tables = np.random.randint(current_num_tables, size=N) 
    mean_tables = np.zeros((current_num_tables, D)) 
    #variance_tables =  np.tile(init_variances, np.ones(current_num_tables)) 
    variance_tables = np.kron(np.ones((current_num_tables,1)), init_variances)
    num_ppl_tables = np.zeros((current_num_tables,))
    
    for c in xrange(current_num_tables):
        indices = np.argwhere(assigned_tables == c) 
        mean_c = np.mean(data[indices,:], axis=0) 
        mean_tables[c]=mean_c
        num_ppl_tables[c] = indices.shape[0]

    #plotTables(data, assigned_tables, current_num_tables, mean_tables)

    for epoch in xrange(max_epoch):

        if epoch % 5 ==0 :
            print 'Number of People in the Table: '
            print num_ppl_tables.astype('int16')

        #Compute the likelihood 
        for i in xrange(N):    

            max_likeli = 0    
            likeli_tables = np.zeros((current_num_tables+1,))
            for z_c in xrange(current_num_tables):
                assigned_table = assigned_tables[i] 
                #print i, np.max(assigned_tables), mean_tables.shape[0], mean_tables[assigned_table],variance_tables[assigned_table]
                p_x_c = multi_gauss_var(data[i,:], mean_tables[assigned_table], np.sqrt(np.diag(variance_tables[assigned_table])),log=False)
                
                if assigned_tables[i] == z_c:
                    prior = (num_ppl_tables[z_c]-1) / (N-1+alpha)
                else:
                    prior = num_ppl_tables[z_c] / (N-1+alpha)
                p_z_x = p_x_c * prior
                likeli_tables[z_c] = p_z_x
                max_likeli = max(max_likeli, p_x_c)

            #Computer the likelihood of choose each tables
            likelihood_new_table = (max_likeli * alpha) / (N-1+alpha)
            likeli_tables[z_c+1] = likelihood_new_table
            #import pdb; pdb.set_trace()
            likeli_tables = likeli_tables / np.sum(likeli_tables)

            #Assign a new table to a data i 
            sample_table_index = int(np.argwhere(np.random.multinomial(1, likeli_tables, size=1).flatten()==1))
            previous_assignd_indice_i = assigned_tables[i]

            if sample_table_index < current_num_tables:
                assigned_tables[i] = sample_table_index
                indice = np.argwhere(assigned_tables == sample_table_index)
                mean_c  = np.mean(data[indice,:], axis=0) 
            else:
                assigned_tables[i] = sample_table_index
                indice         = np.argwhere(assigned_tables == sample_table_index) 
                mean_c          = np.mean(data[indice,:], axis=0) 

            #Update parameters 
            indice_pre = np.argwhere(assigned_tables == previous_assignd_indice_i) 
            mean_pc = np.mean(data[indice_pre,:], axis=0) 
            mean_tables[previous_assignd_indice_i,:] = mean_pc
            num_ppl_tables[previous_assignd_indice_i] -= 1

            if sample_table_index < current_num_tables:
                mean_tables[sample_table_index] = mean_c
                num_ppl_tables[sample_table_index] += 1
            else:
                current_num_tables +=1 
                mean_tables     = np.concatenate((mean_tables, np.asarray(mean_c)), axis=0)
                num_ppl_tables  = np.concatenate((num_ppl_tables, np.asarray([1])), axis=1)
                variance_tables = np.kron(np.ones((current_num_tables,1)), init_variances)

            #Eliminate the empty tables
            table_index = np.where(num_ppl_tables == 0)[0].flatten()
            if int(table_index.shape[0]) != 0:
                indice = np.argwhere(assigned_tables >= table_index[0]).flatten()
                assigned_tables[indice] -= 1

            indice = np.argwhere(num_ppl_tables > 0).flatten()
            current_num_tables = int(indice.shape[0])
            mean_tables = mean_tables[indice,:]
            variance_tables = variance_tables[indice,:]
            num_ppl_tables = num_ppl_tables[indice]
  
            if np.sum(num_ppl_tables <= 0) > 0:
                print 'Warning: Number of People per Table cannot be negative'
                import pdb; pdb.set_trace()


if __name__ == '__main__': 

    num_data_cluster = 50
    num_gaussian     = 10
    variances        = np.asarray([.4, .4])
    max_space        = np.asarray([10, 10])
    min_space        = np.zeros((2,)).T
    data, label, means = gen_gmm_cluster(num_data_cluster, num_gaussian, variances, max_space, min_space)

    init_num_tables = 1
    alpha = 1.8
    init_variance = np.asarray([0.4, 0.4]) 
    max_epoch = 1000
    fit_chinese_restaurant_process(data, alpha, init_variance, init_num_tables, max_epoch) 
    pass 
