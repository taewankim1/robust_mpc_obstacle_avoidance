import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import numpy as np
import cvxpy as cvx
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))

import cost
import model
import IPython

from Scaling import TrajectoryScaling

class funlopt :
    def __init__(self,ix,iu,iq,ip,iw,N,myScaling,
            alpha=0.99,lambda_mu=0.1, 
            w_tr=1,flag_nonlinearity=True) :
        self.ix = ix
        self.iu = iu
        self.iq = iq
        self.ip = ip
        self.iw = iw
        self.N = N
        self.small = 1e-12
        self.w_tr = w_tr
        self.flag_nl = flag_nonlinearity 
        if self.flag_nl == True :
            print("nonlinear funnel")
        elif self.flag_nl == False :
            print("linear funnel")
        else :
            print("Put True or False")

        self.alpha = alpha
        self.lambda_mu = lambda_mu
        assert self.alpha > self.lambda_mu
        self.Sx,self.iSx,self.sx,self.Su,self.iSu,self.su = myScaling.get_scaling()

    def cvx_initialize(self,Qini,Qf=None) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw


        # fixed parameter
        alpha,lambda_mu = self.alpha,self.lambda_mu

        # optimization variables
        Qcvx = []
        Ycvx = []
        nu_p = []
        for i in range(N+1) :
            Qcvx.append(cvx.Variable((ix,ix), PSD=True))
            if i < N :
                Ycvx.append(cvx.Variable((iu,ix)))
                nu_p.append(cvx.Variable(pos=True))
        nu_Q = cvx.Variable(N+1)
        nu_K = cvx.Variable(N)

        # parameters
        A,B,F = [],[],[]
        for i in range(N) :
            A.append(cvx.Parameter((ix,ix)))
            B.append(cvx.Parameter((ix,iu)))
            F.append(cvx.Parameter((ix,iw)))
        C = cvx.Parameter((iq,ix))
        D = cvx.Parameter((iq,iu))
        E = cvx.Parameter((ix,ip))
        G = cvx.Parameter((iq,iw))

        Qbar_unscaled = []
        Ybar_unscaled = []
        for i in range(N+1) :
            Qbar_unscaled.append(cvx.Parameter((ix,ix), PSD=True))
            if i < N :
                Ybar_unscaled.append(cvx.Parameter((iu,ix)))
        gamma_inv_squared = []
        for i in range(N) :
            gamma_inv_squared.append(cvx.Parameter(pos=True))

        constraints = []
        for i in range(N) :
            Qi = self.Sx@Qcvx[i]@self.Sx
            Yi = self.Su@Ycvx[i]@self.Sx
            Qi_next = self.Sx@Qcvx[i+1]@self.Sx
            if self.flag_nl == True :
                self.LMI_nonlinear(Qi,Yi,Qi_next,
                    nu_p[i],
                    A[i],B[i],C,D,E,F[i],G,
                    alpha,lambda_mu,
                    gamma_inv_squared[i],
                    constraints)
            else :
                self.LMI_linear(Qi,Yi,Qi_next,
                    nu_p[i],
                    A[i],B[i],C,D,E,F[i],G,
                    alpha,lambda_mu,
                    constraints)

        for i in range(N+1) :
            Qi = self.Sx@Qcvx[i]@self.Sx
            constraints.append(Qi << nu_Q[i]*np.eye(ix))
            constraints.append(Qi >> np.eye(ix)*self.small) # PD
            # constraints.append(Qi << 50*Qini)

        for i in range(N) :
            Yi = self.Su@Ycvx[i]@self.Sx
            Qi = self.Sx@Qcvx[i]@self.Sx
            tmp1 = cvx.hstack((nu_K[i]*np.eye(iu),Yi))
            tmp2 = cvx.hstack((Yi.T,Qi))
            constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)

        # initial condition
        Qi = self.Sx@Qcvx[0]@self.Sx    
        constraints.append(Qi >> Qini)
        # final condition
        Qi = self.Sx@Qcvx[-1]@self.Sx   
        if Qf is not None :
            constraints.append(Qi << Qf )

        l_state = cvx.sum(nu_Q)
        l_input = cvx.sum(nu_K)
        l = l_state + l_input

        self.prob = cvx.Problem(cvx.Minimize(l), constraints)
        print("Is DPP? ",self.prob.is_dcp(dpp=True))

        # save variables
        self.cvx_variables = {}
        self.cvx_variables['Qcvx'] = Qcvx
        self.cvx_variables['Ycvx'] = Ycvx
        self.cvx_variables['nu_Q'] = nu_Q
        self.cvx_variables['nu_K'] = nu_K
        self.cvx_variables['nu_p'] = nu_p
        # save params
        self.cvx_params = {}
        self.cvx_params['A'] = A
        self.cvx_params['B'] = B
        self.cvx_params['C'] = C
        self.cvx_params['D'] = D
        self.cvx_params['E'] = E
        self.cvx_params['F'] = F
        self.cvx_params['G'] = G
        self.cvx_params['Qbar_unscaled'] = Qbar_unscaled
        self.cvx_params['Ybar_unscaled'] = Ybar_unscaled
        self.cvx_params['gamma_inv_squared'] = gamma_inv_squared
        # save cost
        self.cvx_cost = {}
        self.cvx_cost['l_state'] = l_state
        self.cvx_cost['l_input'] = l_input
        self.cvx_cost['l'] = l

    def LMI_linear(self,Qi,Yi,Qi_next,nu_p,A,B,C,D,E,F,G,alpha,lambda_mu,constraints) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw
        
        LMI11 = alpha*Qi - lambda_mu * Qi
        LMI21 = np.zeros((iw,ix))
        LMI31 = A@Qi+B@Yi

        LMI22 = lambda_mu * np.eye(iw)
        LMI32 = F

        LMI33 = Qi_next


        row1 = cvx.hstack((LMI11,LMI21.T,LMI31.T))
        row2 = cvx.hstack((LMI21,LMI22,LMI32.T))
        row3 = cvx.hstack((LMI31,LMI32,LMI33))
        LMI = cvx.vstack((row1,row2,row3))

        constraints.append(LMI  >> 0)
    
    def LMI_nonlinear(self,Qi,Yi,Qi_next,nu_p,A,B,C,D,E,F,G,alpha,lambda_mu,gamma_inv_squared,constraints) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw
        
        LMI11 = alpha*Qi - lambda_mu * Qi
        LMI21 = np.zeros((ip,ix))
        LMI31 = np.zeros((iw,ix))
        LMI41 = A@Qi+B@Yi
        LMI51 = C@Qi+D@Yi
        
        LMI22 = nu_p * np.eye(ip)
        LMI32 = np.zeros((iw,ip))
        LMI42 = nu_p*E
        LMI52 = np.zeros((iq,ip))

        LMI33 = lambda_mu * np.eye(iw)
        LMI43 = F
        LMI53 = G

        LMI44 = Qi_next
        LMI54 = np.zeros((iq,ix))

        # LMI55 = self.nu_p * (1/gamma[i]**2) * np.eye(iq)
        LMI55 = nu_p * gamma_inv_squared * np.eye(iq)

        row1 = cvx.hstack((LMI11,LMI21.T,LMI31.T,LMI41.T,LMI51.T))
        row2 = cvx.hstack((LMI21,LMI22,LMI32.T,LMI42.T,LMI52.T))
        row3 = cvx.hstack((LMI31,LMI32,LMI33,LMI43.T,LMI53.T))
        row4 = cvx.hstack((LMI41,LMI42,LMI43,LMI44,LMI54.T))
        row5 = cvx.hstack((LMI51,LMI52,LMI53,LMI54,LMI55))
        LMI = cvx.vstack((row1,row2,row3,row4,row5))

        constraints.append(LMI  >> 0)


    def solve(self,gamma,Qhat,Yhat,A,B,C,D,E,F,G) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw

        for i in range(N) :
            self.cvx_params['A'][i].value = A[i]
            self.cvx_params['B'][i].value = B[i]
            self.cvx_params['F'][i].value = F[i]
            self.cvx_params['gamma_inv_squared'][i].value = 1 / (gamma[i]**2)
            self.cvx_params['Qbar_unscaled'][i].value = self.iSx@Qhat[i]@self.iSx
            self.cvx_params['Ybar_unscaled'][i].value = self.iSu@Yhat[i]@self.iSx
        self.cvx_params['Qbar_unscaled'][-1].value = self.iSx@Qhat[-1]@self.iSx
        self.cvx_params['C'].value = C
        self.cvx_params['D'].value = D
        self.cvx_params['E'].value = E
        self.cvx_params['G'].value = G

        self.prob.solve(solver=cvx.MOSEK,ignore_dpp=True)
        Qnew = []
        Ynew = []
        nu_p = []
        for i in range(N+1) :
            Qnew.append(self.Sx@self.cvx_variables['Qcvx'][i].value@self.Sx)
            if i < N :
                Ynew.append(self.Su@self.cvx_variables['Ycvx'][i].value@self.Sx)
                nu_p.append(self.cvx_variables['nu_p'][i].value)
        Knew = []
        for i in range(N) :
            Knew.append(Ynew[i]@np.linalg.inv(Qnew[i]))
        Knew = np.array(Knew)
        Qnew = np.array(Qnew)
        Ynew = np.array(Ynew)

        return Qnew,Knew,Ynew,self.prob.status, \
            self.cvx_cost['l_state'].value, \
            self.cvx_cost['l_input'].value

class funlopt_with_LQR(funlopt) :
    def __init__(self,ix,iu,iq,ip,iw,N,myScaling,
            alpha=0.99,lambda_mu=0.1, 
            w_tr=1,flag_nonlinearity=True) :
        super().__init__(ix,iu,iq,ip,iw,N,myScaling,
            alpha=0.99,lambda_mu=0.1, 
            w_tr=1,flag_nonlinearity=True)

    def cvx_initialize(self,Qini,Qf=None) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw


        # fixed parameter
        alpha,lambda_mu = self.alpha,self.lambda_mu

        # Q
        Qcvx = []
        nu_p = []
        for i in range(N+1) :
            Qcvx.append(cvx.Variable((ix,ix), PSD=True))
            if i < N :
                nu_p.append(cvx.Variable(pos=True))

        Kcvx = []
        Ycvx = []
        for i in range(N) :
            Qi = self.Sx@Qcvx[i]@self.Sx
            Kcvx.append(cvx.Parameter((iu,ix)))
            Ki = Kcvx[i]
            Ycvx.append(Ki@Qi)

        nu_Q = cvx.Variable(N+1)
        nu_K = cvx.Variable(N)

        # parameters
        A,B,F = [],[],[]
        for i in range(N) :
            A.append(cvx.Parameter((ix,ix)))
            B.append(cvx.Parameter((ix,iu)))
            F.append(cvx.Parameter((ix,iw)))
        C = cvx.Parameter((iq,ix))
        D = cvx.Parameter((iq,iu))
        E = cvx.Parameter((ix,ip))
        G = cvx.Parameter((iq,iw))

        Qbar_unscaled = []
        for i in range(N+1) :
            Qbar_unscaled.append(cvx.Parameter((ix,ix), PSD=True))
        gamma_inv_squared = []
        for i in range(N) :
            gamma_inv_squared.append(cvx.Parameter(pos=True))

        constraints = []
        for i in range(N) :
            Qi = self.Sx@Qcvx[i]@self.Sx
            Yi = Ycvx[i]
            Qi_next = self.Sx@Qcvx[i+1]@self.Sx
            if self.flag_nl == True :
                self.LMI_nonlinear(Qi,Yi,Qi_next,
                    nu_p[i],
                    A[i],B[i],C,D,E,F[i],G,
                    alpha,lambda_mu,
                    gamma_inv_squared[i],
                    constraints)
            else :
                self.LMI_linear(Qi,Yi,Qi_next,
                    nu_p[i],
                    A[i],B[i],C,D,E,F[i],G,
                    alpha,lambda_mu,
                    constraints)

        for i in range(N+1) :
            Qi = self.Sx@Qcvx[i]@self.Sx
            constraints.append(Qi << nu_Q[i]*np.eye(ix))
            constraints.append(Qi >> np.eye(ix)*self.small) # PD
            # constraints.append(Qi << 50*Qini)

        for i in range(N) :
            Yi = Ycvx[i]
            Qi = self.Sx@Qcvx[i]@self.Sx
            tmp1 = cvx.hstack((nu_K[i]*np.eye(iu),Yi))
            tmp2 = cvx.hstack((Yi.T,Qi))
            constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)

        # initial condition
        Qi = self.Sx@Qcvx[0]@self.Sx    
        constraints.append(Qi >> Qini)
        # final condition
        Qi = self.Sx@Qcvx[-1]@self.Sx   
        if Qf is not None :
            constraints.append(Qi << Qf )

        l_state = cvx.sum(nu_Q)
        l_input = cvx.sum(nu_K)
        l = l_state + l_input

        self.prob = cvx.Problem(cvx.Minimize(l), constraints)
        print("Is DPP? ",self.prob.is_dcp(dpp=True))

        # save variables
        self.cvx_variables = {}
        self.cvx_variables['Qcvx'] = Qcvx
        self.cvx_variables['Ycvx'] = Ycvx
        self.cvx_variables['nu_Q'] = nu_Q
        self.cvx_variables['nu_K'] = nu_K
        self.cvx_variables['nu_p'] = nu_p
        # save params
        self.cvx_params = {}
        self.cvx_params['A'] = A
        self.cvx_params['B'] = B
        self.cvx_params['C'] = C
        self.cvx_params['D'] = D
        self.cvx_params['E'] = E
        self.cvx_params['F'] = F
        self.cvx_params['G'] = G
        self.cvx_params['Kgain'] = Kcvx
        self.cvx_params['Qbar_unscaled'] = Qbar_unscaled
        self.cvx_params['gamma_inv_squared'] = gamma_inv_squared
        # save cost
        self.cvx_cost = {}
        self.cvx_cost['l_state'] = l_state
        self.cvx_cost['l_input'] = l_input
        self.cvx_cost['l'] = l

    def solve(self,gamma,Qhat,Kgain,A,B,C,D,E,F,G) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw

        for i in range(N) :
            self.cvx_params['A'][i].value = A[i]
            self.cvx_params['B'][i].value = B[i]
            self.cvx_params['F'][i].value = F[i]
            self.cvx_params['Kgain'][i].value = Kgain[i]
            self.cvx_params['gamma_inv_squared'][i].value = 1 / (gamma[i]**2)
            self.cvx_params['Qbar_unscaled'][i].value = self.iSx@Qhat[i]@self.iSx
        self.cvx_params['Qbar_unscaled'][-1].value = self.iSx@Qhat[-1]@self.iSx
        self.cvx_params['C'].value = C
        self.cvx_params['D'].value = D
        self.cvx_params['E'].value = E
        self.cvx_params['G'].value = G

        self.prob.solve(solver=cvx.MOSEK,ignore_dpp=True)
        Qnew = []
        nu_p = []
        for i in range(N+1) :
            Qnew.append(self.Sx@self.cvx_variables['Qcvx'][i].value@self.Sx)
            if i < N :
                nu_p.append(self.cvx_variables['nu_p'][i].value)
        Qnew = np.array(Qnew)

        return Qnew,self.prob.status, \
            self.cvx_cost['l_state'].value, \
            self.cvx_cost['l_input'].value

class funlopt_with_LQR_fixed_Q(funlopt) :
    def __init__(self,ix,iu,iq,ip,iw,N,myScaling,
            alpha=0.99,lambda_mu=0.1, 
            w_tr=1,flag_nonlinearity=True) :
        super().__init__(ix,iu,iq,ip,iw,N,myScaling,
            alpha=0.99,lambda_mu=0.1, 
            w_tr=1,flag_nonlinearity=True)

    def cvx_initialize(self,Qnom,Qini,Qf=None) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw

        # fixed parameter
        alpha,lambda_mu = self.alpha,self.lambda_mu

        # c
        ccvx = cvx.Variable(N+1,pos=True)
        Qcvx = []
        nu_p = []
        for i in range(N+1) :
            Qcvx.append(ccvx[i]*Qnom)
            if i < N :
                nu_p.append(cvx.Variable(pos=True))

        Kcvx = []
        Ycvx = []
        for i in range(N) :
            Qi = Qcvx[i]
            Kcvx.append(cvx.Parameter((iu,ix)))
            Ki = Kcvx[i]
            Ycvx.append(Ki@Qi)

        nu_Q = cvx.Variable(N+1)
        nu_K = cvx.Variable(N)

        # parameters
        A,B,F = [],[],[]
        for i in range(N) :
            A.append(cvx.Parameter((ix,ix)))
            B.append(cvx.Parameter((ix,iu)))
            F.append(cvx.Parameter((ix,iw)))
        C = cvx.Parameter((iq,ix))
        D = cvx.Parameter((iq,iu))
        E = cvx.Parameter((ix,ip))
        G = cvx.Parameter((iq,iw))

        Qbar_unscaled = []
        for i in range(N+1) :
            Qbar_unscaled.append(cvx.Parameter((ix,ix), PSD=True))
        gamma_inv_squared = []
        for i in range(N) :
            gamma_inv_squared.append(cvx.Parameter(pos=True))

        constraints = []
        for i in range(N) :
            Qi = Qcvx[i]
            Yi = Ycvx[i]
            Qi_next = Qcvx[i+1]
            if self.flag_nl == True :
                self.LMI_nonlinear(Qi,Yi,Qi_next,
                    nu_p[i],
                    A[i],B[i],C,D,E,F[i],G,
                    alpha,lambda_mu,
                    gamma_inv_squared[i],
                    constraints)
            else :
                self.LMI_linear(Qi,Yi,Qi_next,
                    nu_p[i],
                    A[i],B[i],C,D,E,F[i],G,
                    alpha,lambda_mu,
                    constraints)

        for i in range(N+1) :
            Qi = Qcvx[i]
            constraints.append(Qi << nu_Q[i]*np.eye(ix))
            constraints.append(Qi >> np.eye(ix)*self.small) # PD
            # constraints.append(Qi << 50*Qini)

        for i in range(N) :
            Yi = Ycvx[i]
            Qi = Qcvx[i]
            tmp1 = cvx.hstack((nu_K[i]*np.eye(iu),Yi))
            tmp2 = cvx.hstack((Yi.T,Qi))
            constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)

        # initial condition
        Qi = Qcvx[0]
        constraints.append(Qi >> Qini)
        # final condition
        Qi = Qcvx[-1]
        if Qf is not None :
            constraints.append(Qi << Qf )

        l_state = cvx.sum(nu_Q)
        l_input = cvx.sum(nu_K)
        l = 1*l_state + 1*l_input

        self.prob = cvx.Problem(cvx.Minimize(l), constraints)
        print("Is DPP? ",self.prob.is_dcp(dpp=True))

        # save variables
        self.cvx_variables = {}
        self.cvx_variables['ccvx'] = ccvx
        self.cvx_variables['Qcvx'] = Qcvx
        self.cvx_variables['Ycvx'] = Ycvx
        self.cvx_variables['nu_Q'] = nu_Q
        self.cvx_variables['nu_K'] = nu_K
        self.cvx_variables['nu_p'] = nu_p
        # save params
        self.cvx_params = {}
        self.cvx_params['A'] = A
        self.cvx_params['B'] = B
        self.cvx_params['C'] = C
        self.cvx_params['D'] = D
        self.cvx_params['E'] = E
        self.cvx_params['F'] = F
        self.cvx_params['G'] = G
        self.cvx_params['Kgain'] = Kcvx
        self.cvx_params['Qbar_unscaled'] = Qbar_unscaled
        self.cvx_params['gamma_inv_squared'] = gamma_inv_squared
        # save cost
        self.cvx_cost = {}
        self.cvx_cost['l_state'] = l_state
        self.cvx_cost['l_input'] = l_input
        self.cvx_cost['l'] = l

    def solve(self,gamma,Qhat,Kgain,A,B,C,D,E,F,G) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw

        for i in range(N) :
            self.cvx_params['A'][i].value = A[i]
            self.cvx_params['B'][i].value = B[i]
            self.cvx_params['F'][i].value = F[i]
            self.cvx_params['Kgain'][i].value = Kgain[i]
            self.cvx_params['gamma_inv_squared'][i].value = 1 / (gamma[i]**2)
            self.cvx_params['Qbar_unscaled'][i].value = self.iSx@Qhat[i]@self.iSx
        self.cvx_params['Qbar_unscaled'][-1].value = self.iSx@Qhat[-1]@self.iSx
        self.cvx_params['C'].value = C
        self.cvx_params['D'].value = D
        self.cvx_params['E'].value = E
        self.cvx_params['G'].value = G

        self.prob.solve(solver=cvx.MOSEK,ignore_dpp=True)
        Qnew = []
        cnew = []
        nu_p = []
        for i in range(N+1) :
            Qnew.append(self.cvx_variables['Qcvx'][i].value)
            cnew.append(self.cvx_variables['ccvx'][i].value)
            if i < N :
                nu_p.append(self.cvx_variables['nu_p'][i].value)
        Qnew = np.array(Qnew)

        return Qnew,cnew,self.prob.status, \
            self.cvx_cost['l_state'].value, \
            self.cvx_cost['l_input'].value


class funlopt_at_final :
    def __init__(self,ix,iu,iq,ip,iw,N,myScaling,
            alpha=0.99,lambda_mu=0.1, 
            w_tr=1,flag_nonlinearity=True) :
        self.ix = ix
        self.iu = iu
        self.iq = iq
        self.ip = ip
        self.iw = iw
        self.N = N
        self.small = 1e-12
        self.w_tr = w_tr
        self.flag_nl = flag_nonlinearity 

        self.alpha = alpha
        self.lambda_mu = lambda_mu
        assert self.alpha > self.lambda_mu
        self.Sx,self.iSx,self.sx,self.Su,self.iSu,self.su = myScaling.get_scaling()

    def cvx_initialize(self,Qini) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw


        # fixed parameter
        alpha,lambda_mu = self.alpha,self.lambda_mu

        # optimization variables
        nu_p = []
        Qcvx = cvx.Variable((ix,ix), PSD=True) 
        Ycvx = cvx.Variable((iu,ix))
        nu_p = cvx.Variable(pos=True)
        nu_Q = cvx.Variable(1)
        nu_K = cvx.Variable(1)

        # parameters
        A = cvx.Parameter((ix,ix))
        B = cvx.Parameter((ix,iu))
        F = cvx.Parameter((ix,iw))
        C = cvx.Parameter((iq,ix))
        D = cvx.Parameter((iq,iu))
        E = cvx.Parameter((ix,ip))
        G = cvx.Parameter((iq,iw))

        gamma_inv_squared = cvx.Parameter(pos=True)

        constraints = []
        if self.flag_nl == True :
            self.LMI_nonlinear(Qcvx,Ycvx,
                nu_p,
                A,B,C,D,E,F,G,
                alpha,lambda_mu,
                gamma_inv_squared,
                constraints)
        else :
            self.LMI_linear(Qcvx,Ycvx,
                nu_p,
                A,B,C,D,E,F,G,
                alpha,lambda_mu,
                constraints)

        Qi = self.Sx@Qcvx@self.Sx
        # constraints.append(Qi << nu_Q*np.eye(ix))
        # constraints.append(Qi >> np.eye(ix)*self.small) # PD
        # constraints.append(Qi << 50*Qini)

        Yi = self.Su@Ycvx@self.Sx
        Qi = self.Sx@Qcvx@self.Sx
        tmp1 = cvx.hstack((nu_K*np.eye(iu),Yi))
        tmp2 = cvx.hstack((Yi.T,Qi))
        # constraints.append( cvx.vstack((tmp1,tmp2)) >> 0)

        # initial condition
        Qi = self.Sx@Qcvx@self.Sx    
        constraints.append(Qi >> Qini)

        Qi = self.Sx@Qcvx@self.Sx 

        # l = cvx.sum(nu_Q) + 0*nu_Q[-1] + 0*cvx.sum(nu_K)
        l = cvx.trace(Qi)
        l_all = l

        self.prob = cvx.Problem(cvx.Minimize(l_all), constraints)
        print("Is DPP? ",self.prob.is_dcp(dpp=True))

        # save variables
        self.cvx_variables = {}
        self.cvx_variables['Qcvx'] = Qcvx
        self.cvx_variables['Ycvx'] = Ycvx
        self.cvx_variables['nu_Q'] = nu_Q
        self.cvx_variables['nu_K'] = nu_K
        self.cvx_variables['nu_p'] = nu_p
        # save params
        self.cvx_params = {}
        self.cvx_params['A'] = A
        self.cvx_params['B'] = B
        self.cvx_params['C'] = C
        self.cvx_params['D'] = D
        self.cvx_params['E'] = E
        self.cvx_params['F'] = F
        self.cvx_params['G'] = G
        self.cvx_params['gamma_inv_squared'] = gamma_inv_squared
        # save cost
        self.cvx_cost = {}
        self.cvx_cost['l_all'] = l_all
        self.cvx_cost['l'] = l

    def solve(self,gamma,A,B,C,D,E,F,G) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw

        self.cvx_params['A'].value = A
        self.cvx_params['B'].value = B
        self.cvx_params['F'].value = F
        self.cvx_params['gamma_inv_squared'].value = 1 / (gamma**2)
        self.cvx_params['C'].value = C
        self.cvx_params['D'].value = D
        self.cvx_params['E'].value = E
        self.cvx_params['G'].value = G

        self.prob.solve(solver=cvx.MOSEK)
        print(self.prob.status)
        Qnew = self.Sx@self.cvx_variables['Qcvx'].value@self.Sx
        Ynew = self.Su@self.cvx_variables['Ycvx'].value@self.Sx
        nu_p = self.cvx_variables['nu_p'].value

        Knew = Ynew@np.linalg.inv(Qnew)

        Knew = np.array(Knew)
        Qnew = np.array(Qnew)
        Ynew = np.array(Ynew)

        return Qnew,Knew,Ynew,self.prob.status,self.cvx_cost['l'].value

    def LMI_linear(self,Qcvx,Ycvx,nu_p,A,B,C,D,E,F,G,alpha,lambda_mu,constraints) :
        print("linear funnel")
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw

        Qi = self.Sx@Qcvx@self.Sx
        Yi = self.Su@Ycvx@self.Sx
        
        LMI11 = alpha*Qi - lambda_mu * Qi
        LMI21 = np.zeros((iw,ix))
        LMI31 = A@Qi+B@Yi

        LMI22 = lambda_mu * np.eye(iw)
        LMI32 = F

        LMI33 = Qi


        row1 = cvx.hstack((LMI11,LMI21.T,LMI31.T))
        row2 = cvx.hstack((LMI21,LMI22,LMI32.T))
        row3 = cvx.hstack((LMI31,LMI32,LMI33))
        LMI = cvx.vstack((row1,row2,row3))

        constraints.append(LMI  >> 0)
    
    def LMI_nonlinear(self,Qcvx,Ycvx,nu_p,A,B,C,D,E,F,G,alpha,lambda_mu,gamma_inv_squared,constraints) :
        print("nonlinear funnel")
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw

        Qi = self.Sx@Qcvx@self.Sx
        Yi = self.Su@Ycvx@self.Sx
        
        LMI11 = alpha*Qi - lambda_mu * Qi
        LMI21 = np.zeros((ip,ix))
        LMI31 = np.zeros((iw,ix))
        LMI41 = A@Qi+B@Yi
        LMI51 = C@Qi+D@Yi
        
        LMI22 = nu_p * np.eye(ip)
        LMI32 = np.zeros((iw,ip))
        LMI42 = nu_p*E
        LMI52 = np.zeros((iq,ip))

        LMI33 = lambda_mu * np.eye(iw)
        LMI43 = F
        LMI53 = G

        LMI44 = Qi
        LMI54 = np.zeros((iq,ix))

        # LMI55 = self.nu_p * (1/gamma[i]**2) * np.eye(iq)
        LMI55 = nu_p * gamma_inv_squared * np.eye(iq)

        row1 = cvx.hstack((LMI11,LMI21.T,LMI31.T,LMI41.T,LMI51.T))
        row2 = cvx.hstack((LMI21,LMI22,LMI32.T,LMI42.T,LMI52.T))
        row3 = cvx.hstack((LMI31,LMI32,LMI33,LMI43.T,LMI53.T))
        row4 = cvx.hstack((LMI41,LMI42,LMI43,LMI44,LMI54.T))
        row5 = cvx.hstack((LMI51,LMI52,LMI53,LMI54,LMI55))
        LMI = cvx.vstack((row1,row2,row3,row4,row5))

        constraints.append(LMI  >> 0)
