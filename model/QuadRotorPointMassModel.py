
# coding: utf-8

# In[ ]:

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))

from model import OptimalcontrolModel

class quadrotorpm(OptimalcontrolModel):
    def __init__(self,name,ix,iu,delT,linearization="analytic"):
        super().__init__(name,ix,iu,delT,linearization)
        self.g = 9.81
        
    def forward(self,x,u,idx=None,discrete=True):
        
        xdim = np.ndim(x)
        if xdim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
        else :
            N = np.size(x,axis = 0)
        udim = np.ndim(u)
        if udim == 1 :
            u = np.expand_dims(u,axis=0)
     
        # state & input
        rx = x[:,0]
        ry = x[:,1]
        rz = x[:,2]
        vx = x[:,3]
        vy = x[:,4]
        vz = x[:,5]
        
        ax = u[:,0]
        ay = u[:,1]
        az = u[:,2]
        
        # output
        f = np.zeros_like(x)
        f[:,0] = vx
        f[:,1] = vy
        f[:,2] = vz
        f[:,3] = ax
        f[:,4] = ay
        f[:,5] = az-self.g

        if discrete is True :
            return np.squeeze(x + f * self.delT)
        else :
            return f

    def diff(self,x,u,idx=None,discrete=False) :
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)

        fx = np.zeros((N,ix,ix))
        fu = np.zeros((N,ix,iu))

        fx[:,0,3] = 1
        fx[:,1,4] = 1
        fx[:,2,5] = 1
    
        fu[:,3,0] = 1
        fu[:,4,1] = 1
        fu[:,5,2] = 1

        if discrete == False :
            return fx.squeeze(), fu.squeeze()

        else :
            return np.repeat(np.expand_dims(np.eye(ix),0),N,0)+fu*self.delT,fu*self.delT