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

from cost import OptimalcontrolCost

class Finaltime(OptimalcontrolCost):
    def __init__(self,name,ix,iu,N):
        super().__init__(name,ix,iu,N)
        
    def estimate_cost_cvx(self,sigma):
        return sigma