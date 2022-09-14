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

class quadrotorsa(OptimalcontrolConstraints):
    def __init__(self,name,ix,iu,ih,c=None,H=None):
        super().__init__(name,ix,iu,ih)
        self.idx_bc_f = slice(0, 12)
        self.T_min = 5
        self.T_max = 20
        self.c = c
        self.H = H
        self.roll_max = np.deg2rad(20)
        self.rolldot_max = np.deg2rad(40)
        self.pitch_max = np.deg2rad(20)
        self.pitchdot_max = np.deg2rad(40)
    def set_obstacle(self,c,H) :
        self.c = c
        self.H = H
        
    def forward(self,x,u,xbar,ubar):

        h = []
        # state constraints
        h.append(x[3]<=self.roll_max)
        h.append(-self.roll_max <= x[3])
        h.append(x[4]<=self.pitch_max)
        h.append(-self.pitch_max <= x[4])

        h.append(x[9]<=self.rolldot_max)
        h.append(-self.rolldot_max <= x[9])
        h.append(x[10]<=self.pitchdot_max)
        h.append(-self.pitchdot_max <= x[10])

        # obstacle avoidance
        def get_obs_const(c1,H1) :
            return 1 - np.linalg.norm(H1@(xbar[0:3]-c1)) - (H1.T@H1@(xbar[0:3]-c1)/np.linalg.norm(H1@(xbar[0:3]-c1))).T@(x[0:3]-xbar[0:3]) <=0
        if self.H is not None :
            for c1,H1 in zip(self.c,self.H) :
                h.append(get_obs_const(c1,H1))

        # input constraints
        h.append(u[0] <= self.T_max)
        h.append(self.T_min <= u[0])
        return h

    def bc_final(self,x_cvx,xf):
        h = []
        h.append(x_cvx[self.idx_bc_f] == xf[self.idx_bc_f])
        return h
    

    
