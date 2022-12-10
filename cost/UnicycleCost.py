from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import cvxpy as cvx

def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))

from cost.cost import OptimalcontrolCost
class unicycle(OptimalcontrolCost):
    def __init__(self,name,ix,iu,N):
        super().__init__(name,ix,iu,N)
        self.ix = 3
        self.iu = 2
        self.N = N

        self.S = 1*np.identity(ix)
        self.R = 1 * np.identity(iu)

    def bc_final(self,x_cvx,xf):
        h = []
        h.append(x_cvx == xf)
        return h

    def estimate_cost_cvx(self,x,u,idx=None):
        # dimension
        cost_total = cvx.quad_form(x, self.S) + cvx.quad_form(u,self.R)
        
        return cost_total

class UnicycleMPCCost(OptimalcontrolCost):
    def __init__(self,name,ix,iu,N):
        super().__init__(name,ix,iu,N)
        self.ix = 3
        self.iu = 2
        self.N = N

        self.S = 10*np.identity(ix)
        self.R = 1*np.identity(iu)

    def bc_final(self,x_cvx,xf):
        h = []
        h.append(x_cvx == xf)
        return h

    def estimate_cost_cvx(self,x,u,xtrj,utrj,idx=None):
        # dimension
        cost_total = cvx.quad_form(x-xtrj, self.S) + cvx.quad_form(u,self.R)
        # cost_total = cvx.quad_form(x-xtrj, self.S) + cvx.quad_form(u-utrj,self.R)
        
        return cost_total