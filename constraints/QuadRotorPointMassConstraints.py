import matplotlib.pyplot as plt
import numpy as np
import time
import random
import cvxpy as cvx

def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))

from constraints import OptimalcontrolConstraints
import IPython

class quadrotorpm(OptimalcontrolConstraints):
    def __init__(self,name,ix,iu,c=None,H=None):
        super().__init__(name,ix,iu)
        # self.theta_max = np.deg2rad(30) # tilt angle
        self.idx_bc_f = slice(0, 6)
        self.T_min = 5
        self.T_max = 30
        self.c = c
        self.H = H
        self.delta_max = np.deg2rad(30)

    def set_obstacle(self,c,H) :
        self.c = c
        self.H = H
        
    def forward(self,x,u,xbar,ubar,idx=None):

        h = []
        # state constraints
        # obstacle avoidance
        def get_obs_const(c1,H1) :
            return 1 - np.linalg.norm(H1@(xbar[0:3]-c1)) - (H1.T@H1@(xbar[0:3]-c1)/np.linalg.norm(H1@(xbar[0:3]-c1))).T@(x[0:3]-xbar[0:3]) <=0
        if self.H is not None :
            for c1,H1 in zip(self.c,self.H) :
                h.append(get_obs_const(c1,H1))

        # input constraints
        h.append(cvx.norm(u) <= self.T_max)
        h.append(self.T_min - np.transpose(np.expand_dims(ubar,1))@u / np.linalg.norm(ubar) <= 0)
        h.append(np.cos(self.delta_max) * cvx.norm(u) <= u[2])
        return h

    def bc_final(self,x_cvx,xf):
        h = []
        h.append(x_cvx[self.idx_bc_f] == xf[self.idx_bc_f])
        return h
    

    
