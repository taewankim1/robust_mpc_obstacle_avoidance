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


from constraints import OptimalcontrolConstraints

class UnicycleConstraints(OptimalcontrolConstraints):
    def __init__(self,name,ix,iu):
        super().__init__(name,ix,iu)
        self.idx_bc_f = slice(0, ix)
        self.ih = 4
        self.vlim = 2
        self.wlim = np.deg2rad(60)
        
    def forward(self,x,u,xbar=None,ybar=None,idx=None):

        v = u[0]
        w = u[1]

        h = []
        h.append(v <= self.vlim)
        h.append(v >= -self.vlim)
        h.append(w<=self.wlim)
        h.append(w>=-self.wlim)

        return h

    def bc_final(self,x_cvx,xf):
        h = []
        h.append(x_cvx == xf)

        return h

# naming is terrible..
class unicycleMPC(OptimalcontrolConstraints):
    def __init__(self,name,ix,iu):
        self.ix = ix
        self.iu = iu
        # self.c = c
        # self.H = H
        
    def forward(self,x,u,xbar=None,ubar=None,c=None,H=None,bf=None):
        h = []
        # obstacle avoidance
        def get_obs_const(c1,H1) :
            if bf is None :
                return (1 - np.linalg.norm(H1@(xbar[0:2]-c1)) - (H1.T@H1@(xbar[0:2]-c1)/np.linalg.norm(H1@(xbar[0:2]-c1))).T@(x[0:2]-xbar[0:2])<=0)
            else :
                return (1 - np.linalg.norm(H1@(xbar[0:2]-c1)) - (H1.T@H1@(xbar[0:2]-c1)/np.linalg.norm(H1@(xbar[0:2]-c1))).T@(x[0:2]-xbar[0:2]) + bf<=0)
        if H is not None :
            h.append(get_obs_const(c,H))

        return h


