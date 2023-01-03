import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))
from scipy.integrate import solve_ivp
from model.model import OptimalcontrolModel

class unicycle(OptimalcontrolModel):
    def __init__(self,name,ix,iu,linearzation):
        self.iw = 2
        self.iq = 2
        self.ip = 2
        super().__init__(name,ix,iu,linearzation)
        self.C = np.array([[0,0,1],[0,0,0]])
        self.D = np.array([[0,0],[1,0]])
        self.E = np.array([[1,0],[0,1],[0,0]])
        self.G = np.zeros((self.iq,self.iw))

        
    def forward(self,x,u,idx=None):
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
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        
        v = u[:,0]
        w = u[:,1]
        
        # output
        f = np.zeros_like(x)
        f[:,0] = v * np.cos(x3)
        f[:,1] = v * np.sin(x3)
        f[:,2] = w

        return f



    def diff(self,x,u):

        # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
        
        # state & input
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        
        v = u[:,0]
        w = u[:,1]    
        
        fx = np.zeros((N,self.ix,self.ix))
        fx[:,0,0] = 0.0
        fx[:,0,1] = 0.0
        fx[:,0,2] = - v * np.sin(x3)
        fx[:,1,0] = 0.0
        fx[:,1,1] = 0.0
        fx[:,1,2] = v * np.cos(x3)
        fx[:,2,0] = 0.0
        fx[:,2,1] = 0.0
        fx[:,2,2] = 0.0
        
        fu = np.zeros((N,self.ix,self.iu))
        fu[:,0,0] = np.cos(x3)
        fu[:,0,1] = 0.0
        fu[:,1,0] = np.sin(x3)
        fu[:,1,1] = 0.0
        fu[:,2,0] = 0.0
        fu[:,2,1] = 1.0
        
        return np.squeeze(fx) , np.squeeze(fu)

    def forward_uncertain(self,x,u,w,idx=None):
        xdim = np.ndim(x)
        udim = np.ndim(u)
        wdim = np.ndim(w)
        assert xdim == udim
        assert wdim == udim
        if xdim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
            w = np.expand_dims(w,axis=0)
        else :
            N = np.size(x,axis = 0)
     
        # state & input
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        
        v = u[:,0]
        omega = u[:,1]
        
        w1 = w[:,0]
        w2 = w[:,1]

        # output
        f = np.zeros_like(x)
        f[:,0] = v * np.cos(x3) + 0.1 * w1
        f[:,1] = v * np.sin(x3) + 0.1 * w2
        f[:,2] = omega

        return f

    def diff_F(self,x,u,w):

        # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
            w = np.expand_dims(w,axis=0)
        else :
            N = np.size(x,axis = 0)
        
        # state & input
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        
        uv = u[:,0]
        uw = u[:,1]    
        
        fw = np.zeros((N,self.ix,2))
        fw[:,0,0] = 0.1
        fw[:,0,1] = 0.0
        fw[:,1,0] = 0.0
        fw[:,1,1] = 0.1
        fw[:,2,0] = 0.0
        fw[:,2,1] = 0.0
        
        return np.squeeze(fw)
    
    def diff_discrete_zoh_noise(self,x,u,w,delT,tf) :
        # delT = self.delT
        ix = self.ix
        iu = self.iu
        iw = self.iw

        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
            w = np.expand_dims(w,axis=0)
        else :
            N = np.size(x,axis = 0)

        def dvdt(t,V,u,w,length) :
            V = V.reshape((length,ix + ix*ix + ix*iu + ix*iw + ix + ix)).transpose()
            x = V[:ix].transpose()
            Phi = V[ix:ix*ix + ix]
            Phi = Phi.transpose().reshape((length,ix,ix))
            Phi_inv = np.linalg.inv(Phi)
            f = self.forward(x,u)
            if self.type_linearization == "numeric_central" :
                A,B = self.diff_numeric_central(x,u)
            elif self.type_linearization == "numeric_forward" :
                A,B = self.diff_numeric(x,u)
            elif self.type_linearization == "analytic" :
                A,B = self.diff(x,u)
            F = self.diff_F(x,u,w)
            dpdt = np.matmul(A,Phi).reshape((length,ix*ix)).transpose()
            dbmdt = np.matmul(Phi_inv,B).reshape((length,ix*iu)).transpose()
            dbpdt = np.matmul(Phi_inv,F).reshape((length,ix*iu)).transpose()
            dsdt = np.squeeze(np.matmul(Phi_inv,np.expand_dims(f,2))).transpose() / tf
            dzdt = np.squeeze(np.matmul(Phi_inv,-np.matmul(A,np.expand_dims(x,2)) - np.matmul(B,np.expand_dims(u,2)))).transpose()
            dv = np.vstack((f.transpose(),dpdt,dbmdt,dbpdt,dsdt,dzdt))
            return dv.flatten(order='F')
        
        A0 = np.eye(ix).flatten()
        B0 = np.zeros((ix*iu))
        F0 = np.zeros((ix*iw))
        s0 = np.zeros(ix)
        z0 = np.zeros(ix)
        V0 = np.array([np.hstack((x[i],A0,B0,F0,s0,z0)) for i in range(N)]).transpose()
        V0_repeat = V0.flatten(order='F')

        sol = solve_ivp(dvdt,(0,delT),V0_repeat,args=(u[0:N],u[1:],N),rtol=1e-6,atol=1e-10)
        # sol = solve_ivp(dvdt,(0,delT),V0_repeat,args=(u[0:N],u[1:],N))
        # IPython.embed()
        idx_state = slice(0,ix)
        idx_A = slice(ix,ix+ix*ix)
        idx_B = slice(ix+ix*ix,ix+ix*ix+ix*iu)
        idx_F = slice(ix+ix*ix+ix*iu,ix+ix*ix+ix*iu+ix*iw)
        idx_s = slice(ix+ix*ix+ix*iu+ix*iw,ix+ix*ix+ix*iu+ix*iw+ix)
        idx_z = slice(ix+ix*ix+ix*iu+ix*iw+ix,ix+ix*ix+ix*iu+ix*iw+ix+ix)
        sol = sol.y[:,-1].reshape((N,-1))
        # xnew = np.zeros((N+1,ix))
        # xnew[0] = x[0]
        # xnew[1:] = sol[:,:ix]
        x_prop = sol[:,idx_state].reshape((-1,ix))
        A = sol[:,idx_A].reshape((-1,ix,ix))
        B = np.matmul(A,sol[:,idx_B].reshape((-1,ix,iu)))
        F = np.matmul(A,sol[:,idx_F].reshape((-1,ix,iu)))
        s = np.matmul(A,sol[:,idx_s].reshape((-1,ix,1))).squeeze()
        z = np.matmul(A,sol[:,idx_z].reshape((-1,ix,1))).squeeze()

        return A,B,F,s,z,x_prop


class unicycle2(unicycle):
    def __init__(self,name,ix,iu,linearzation):
        super().__init__(name,ix,iu,linearzation)
        # re-define
        self.iw = 2
        self.iq = 3
        self.ip = 2
        self.C = np.array([[0,0,1],[0,0,0],[0,0,0]])
        self.D = np.array([[0,0],[1,0],[0,0]])
        self.E = np.array([[1,0],[0,1],[0,0]])
        self.G = np.array([[0,0],[0,0],[1,0]])

    # def forward
    # def diff 
    def forward_uncertain(self,x,u,w,idx=None):
        xdim = np.ndim(x)
        udim = np.ndim(u)
        wdim = np.ndim(w)
        assert xdim == udim
        assert wdim == udim
        if xdim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
            w = np.expand_dims(w,axis=0)
        else :
            N = np.size(x,axis = 0)
     
        # state & input
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        
        v = u[:,0]
        omega = u[:,1]
        
        w1 = w[:,0]
        w2 = w[:,1]

        # output
        f = np.zeros_like(x)
        f[:,0] = (v + 0.1*w1) * np.cos(x3)
        f[:,1] = (v + 0.1*w1) * np.sin(x3)
        f[:,2] = omega + 0.05*w2

        return f

    def diff_F(self,x,u,w):

        # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
            w = np.expand_dims(w,axis=0)
        else :
            N = np.size(x,axis = 0)
        
        # state & input
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        
        uv = u[:,0]
        uw = u[:,1]    
        
        fw = np.zeros((N,self.ix,2))
        fw[:,0,0] = 0.1*np.cos(x3)
        fw[:,0,1] = 0.0
        fw[:,1,0] = 0.1*np.sin(x3)
        fw[:,1,1] = 0.0
        fw[:,2,0] = 0.0
        fw[:,2,1] = 0.05
        
        return np.squeeze(fw)
