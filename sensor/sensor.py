import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
#     print ("Values are: \n%s" % (x))

class Sensor(object) :
    def __init__(self,name,ix,iu) :
        self.name = name
        self.ix = ix
        self.iu = iu

    def forward(self,x,u,idx=None,discrete=True):
        print("this is in parent class")
        pass

    def diff(self) :
        print("this is in parent class")
        pass