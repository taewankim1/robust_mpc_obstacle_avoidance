
# coding: utf-8

# In[ ]:

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.linalg
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))




class OptimalcontrolConstraints(object) :
    def __init__(self,name,ix,iu) :
        self.name = name
        self.ix = ix
        self.iu = iu

    def forward(self) :
        print("this is in parent class")
        pass

    def diff(self) :
        print("this is in parent class")
        pass

    def diff_numeric(self,x,u) :
        # state & input size
        ix = self.ix
        iu = self.iu
        ih = self.ih
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            # x = np.expand_dims(x,axis=0)
            # u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
        
        # numerical difference
        h = pow(2,-17)
        eps_x = np.identity(ix)
        eps_u = np.identity(iu)

        # expand to tensor
        x_mat = np.expand_dims(x,axis=2)
        u_mat = np.expand_dims(u,axis=2)

        # diag
        x_diag = np.tile(x_mat,(1,1,ix))
        u_diag = np.tile(u_mat,(1,1,iu))

        # augmented = [x_aug x], [u, u_aug]
        x_aug = x_diag + eps_x * h
        x_aug = np.dstack((x_aug,np.tile(x_mat,(1,1,iu))))
        x_aug = np.reshape( np.transpose(x_aug,(0,2,1)), (N*(iu+ix),ix))

        u_aug = u_diag + eps_u * h
        u_aug = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug))
        u_aug = np.reshape( np.transpose(u_aug,(0,2,1)), (N*(iu+ix),iu))

        # numerical difference
        f_nominal = self.forward(x,u,) 
        f_change = self.forward(x_aug,u_aug)
        # print_np(f_change)
        f_change = np.reshape(f_change,(N,ix+iu,ih))
        # print_np(f_nominal)
        # print_np(f_change)
        f_diff = ( f_change - np.reshape(f_nominal,(N,1,ih)) ) / h
        # print_np(f_diff)
        f_diff = np.transpose(f_diff,[0,2,1])
        fx = f_diff[:,:,0:ix]
        fu = f_diff[:,:,ix:ix+iu]
        
        return np.squeeze(fx), np.squeeze(fu)

