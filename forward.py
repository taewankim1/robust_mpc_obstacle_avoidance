import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))
import IPython

def forward_full(model,x0,u,N,type_discretization="zoh") :
    ix = model.ix
    iu = model.iu

    def dfdt(t,x,um,up) :
        if type_discretization == "zoh" :
            u = um
        elif type_discretization == "foh" :
            alpha = (model.delT - t) / model.delT
            beta = t / model.delT
            u = alpha * um + beta * up
        return np.squeeze(model.forward(x,u,discrete=False))

    xnew = np.zeros((N+1,ix))
    xnew[0] = x0
    cnew = np.zeros(N+1)

    for i in range(N) :
        sol = solve_ivp(dfdt,(0,model.delT),xnew[i],args=(u[i],u[i+1]),method='RK45',rtol=1e-6,atol=1e-10)
        xnew[i+1] = sol.y[:,-1]
        # cnew[i] = self.cost.estimate_cost(xnew[i],u[i])
    # cnew[N] = self.cost.estimate_cost(xnew[N],np.zeros(self.model.iu))

    return xnew,u


def forward_one_step(model,x,u) :
    ix = model.ix
    iu = model.iu    

    N = np.size(x,axis = 0)

    def dfdt(t,x,u) :
        x_ = np.reshape(x,(N,ix)) 
        u_ = np.reshape(u,(N,iu)) 
        x_dot = np.squeeze(model.forward(x_,u_,discrete=False))
        x_dot = np.reshape(x_dot,(ix*N))
        return x_dot

    x = np.reshape(x,(ix*N)) 
    u = np.reshape(u,(iu*N)) 
    sol = solve_ivp(dfdt,(0,model.delT),x,args=(u,),method='RK45',rtol=1e-6,atol=1e-10)
    x_next = sol.y[:,-1]
    x_next = np.reshape(x_next,(N,ix)) 
    return x_next