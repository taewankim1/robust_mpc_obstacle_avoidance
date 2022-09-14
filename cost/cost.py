
# coding: utf-8

# In[1]:

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import cvxpy as cp

def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))
class OptimalcontrolCost(object) :
    def __init__(self,name,ix,iu,N) :
        self.name = name
        self.ix = ix
        self.iu = iu
        self.N = N

    def estimate_cost(self) :
        print("this is in parent class")
        pass

    def estimate_final_cost(self,x,u) :
        return self.estimate_cost(x,u)

    # def diff_cost(self,x,u):
        
    #     # state & input size
    #     ix = self.ix
    #     iu = self.iu
        
    #     ndim = np.ndim(x)
    #     if ndim == 1: # 1 step state & input
    #         N = 1

    #     else :
    #         N = np.size(x,axis = 0)

    #     # numerical difference
    #     h = pow(2,-17)
    #     eps_x = np.identity(ix)
    #     eps_u = np.identity(iu)

    #     # expand to tensor
    #     # print_np(x)
    #     x_mat = np.expand_dims(x,axis=2)
    #     u_mat = np.expand_dims(u,axis=2)

    #     # diag
    #     x_diag = np.tile(x_mat,(1,1,ix))
    #     u_diag = np.tile(u_mat,(1,1,iu))

    #     # augmented = [x_diag x], [u, u_diag]
    #     x_aug = x_diag + eps_x * h
    #     x_aug = np.dstack((x_aug,np.tile(x_mat,(1,1,iu))))
    #     x_aug = np.reshape( np.transpose(x_aug,(0,2,1)), (N*(iu+ix),ix))
        
    #     u_aug = u_diag + eps_u * h
    #     u_aug = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug))
    #     u_aug = np.reshape( np.transpose(u_aug,(0,2,1)), (N*(iu+ix),iu))

    #     # numerical difference
    #     c_nominal = self.estimate_cost(x,u)
    #     c_change = self.estimate_cost(x_aug,u_aug)
    #     c_change = np.reshape(c_change,(N,1,iu+ix))

    #     c_diff = ( c_change - np.reshape(c_nominal,(N,1,1)) ) / h
    #     c_diff = np.reshape(c_diff,(N,iu+ix))
            
    #     return  np.squeeze(c_diff)
    
    # def hess_cost(self,x,u):
        
    #     # state & input size
    #     ix = self.ix
    #     iu = self.iu
        
    #     ndim = np.ndim(x)
    #     if ndim == 1: # 1 step state & input
    #         N = 1

    #     else :
    #         N = np.size(x,axis = 0)
        
    #     # numerical difference
    #     h = pow(2,-17)
    #     eps_x = np.identity(ix)
    #     eps_u = np.identity(iu)

    #     # expand to tensor
    #     x_mat = np.expand_dims(x,axis=2)
    #     u_mat = np.expand_dims(u,axis=2)

    #     # diag
    #     x_diag = np.tile(x_mat,(1,1,ix))
    #     u_diag = np.tile(u_mat,(1,1,iu))

    #     # augmented = [x_diag x], [u, u_diag]
    #     x_aug = x_diag + eps_x * h
    #     x_aug = np.dstack((x_aug,np.tile(x_mat,(1,1,iu))))
    #     x_aug = np.reshape( np.transpose(x_aug,(0,2,1)), (N*(iu+ix),ix))

    #     u_aug = u_diag + eps_u * h
    #     u_aug = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug))
    #     u_aug = np.reshape( np.transpose(u_aug,(0,2,1)), (N*(iu+ix),iu))


    #     # numerical difference
    #     c_nominal = self.diff_cost(x,u)
    #     c_change = self.diff_cost(x_aug,u_aug)
    #     c_change = np.reshape(c_change,(N,iu+ix,iu+ix))
    #     c_hess = ( c_change - np.reshape(c_nominal,(N,1,ix+iu)) ) / h
    #     c_hess = np.reshape(c_hess,(N,iu+ix,iu+ix))
         
    #     return np.squeeze(c_hess)



    
    
