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

from constraints.constraints import OptimalcontrolConstraints

class UnicycleConstraints(OptimalcontrolConstraints):
    def __init__(self,name,ix,iu,c,H):
        super().__init__(name,ix,iu)
        self.idx_bc_f = slice(0, ix)

        self.vmax = 2.0
        self.vmin = 0.0

        self.wmax = np.deg2rad(60)
        self.wmin = -np.deg2rad(60)

        self.c = c
        self.H = H
        
    def forward(self,x,u,xbar,ybar,refobs=None):

        v = u[0]
        w = u[1]

        h = []
        h.append(v <= self.vmax)
        h.append(v >= self.vmin)
        h.append(w<=self.wmax)
        h.append(w>=self.wmin)
        # obstacle avoidance constraint
        if refobs is not None :
            for obs in refobs :
                h.append(obs[3] + obs[0:2].T@x[0:2]<=obs[2])

        return h

    def bc_final(self,x_cvx,xf):
        h = []
        h.append(x_cvx == xf)

        return h

# naming is terrible..
class UnicycleMPCConstraints(OptimalcontrolConstraints):
    def __init__(self,name,ix,iu,num_obs=0):
        self.ix = ix
        self.iu = iu

        self.vmax = 3.0
        self.vmin = 0.0

        self.wmax = np.deg2rad(60)
        self.wmin = -np.deg2rad(60)

        self.num_obs = num_obs

    def forward(self,x,u,xbar,ybar,refobs=None,bf=None):
        v = u[0]
        w = u[1]

        h = []
        h.append(v <= self.vmax)
        h.append(v >= self.vmin)
        h.append(w<=self.wmax)
        h.append(w>=self.wmin)
        # obstacle avoidance constraint
        if refobs is not None :
            for obs in refobs :
                h.append(obs[3] + obs[0:2].T@x[0:2] + bf<=obs[2])
        return h
        


