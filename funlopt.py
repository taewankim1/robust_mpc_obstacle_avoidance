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

        self.alpha = alpha
        self.lambda_mu = lambda_mu
        assert self.alpha > self.lambda_mu
        self.Sx,self.iSx,self.sx,self.Su,self.iSu,self.su = myScaling.get_scaling()

    def cvx_initialize(self,Qini,Qf=None,flag_time_varying=True) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw


        # fixed parameter
        alpha,lambda_mu = self.alpha,self.lambda_mu

        # optimization variables
        Qcvx = []
        Ycvx = []
        nu_p = []
        if flag_time_varying is True :
            for i in range(N+1) :
                Qcvx.append(cvx.Variable((ix,ix), PSD=True))
                if i < N :
                    Ycvx.append(cvx.Variable((iu,ix)))
                    nu_p.append(cvx.Variable(pos=True))
        else :
            Qtmp = cvx.Variable((ix,ix), PSD=True) 
            Ytmp = cvx.Variable((iu,ix))
            for i in range(N+1) :
                Qcvx.append(Qtmp)
                if i < N :
                    Ycvx.append(Ytmp)
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
            Qbar_unscaled.append(cvx.Variable((ix,ix), PSD=True))
            if i < N :
                Ybar_unscaled.append(cvx.Variable((iu,ix)))
        gamma_inv_squared = []
        for i in range(N) :
            gamma_inv_squared.append(cvx.Parameter(pos=True))

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

        for i in range(N+1) :
            Qi = self.Sx@Qcvx[i]@self.Sx
            constraints.append(Qi << nu_Q[i]*np.eye(ix))
            constraints.append(Qi >> np.eye(ix)*self.small) # PD

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

        # trust region
        objective_tr = []
        for i in range(N) :
            objective_tr.append(cvx.norm(Qcvx[i]-Qbar_unscaled[i],'fro')**2 
                + cvx.norm(Ycvx[i]-Ybar_unscaled[i],'fro')**2)
        objective_tr.append(cvx.norm(Qcvx[-1]-Qbar_unscaled[-1],'fro')**2)
        
        l = 1.5*cvx.sum(nu_Q) + 5*nu_Q[-1] + cvx.sum(nu_K)
        l_tr = cvx.sum(objective_tr)
        l_all = l + self.w_tr*l_tr

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
        self.cvx_params['Qbar_unscaled'] = Qbar_unscaled
        self.cvx_params['Ybar_unscaled'] = Ybar_unscaled
        self.cvx_params['gamma_inv_squared'] = gamma_inv_squared
        # save cost
        self.cvx_cost = {}
        self.cvx_cost['l_all'] = l_all
        self.cvx_cost['l'] = l
        self.cvx_cost['l_tr'] = l_tr

    def LMI_linear(self,Qcvx,Ycvx,nu_p,A,B,C,D,E,F,G,alpha,lambda_mu,constraints) :
        print("linear funnel")
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw
        for i in range(N) :
            Qi = self.Sx@Qcvx[i]@self.Sx
            Yi = self.Su@Ycvx[i]@self.Sx
            Qi_next = self.Sx@Qcvx[i+1]@self.Sx
            
            LMI11 = alpha*Qi - lambda_mu * Qi
            LMI21 = np.zeros((iw,ix))
            LMI31 = A[i]@Qi+B[i]@Yi

            LMI22 = lambda_mu * np.eye(iw)
            LMI32 = F[i]

            LMI33 = Qi_next


            row1 = cvx.hstack((LMI11,LMI21.T,LMI31.T))
            row2 = cvx.hstack((LMI21,LMI22,LMI32.T))
            row3 = cvx.hstack((LMI31,LMI32,LMI33))
            LMI = cvx.vstack((row1,row2,row3))

            constraints.append(LMI  >> 0)
    
    def LMI_nonlinear(self,Qcvx,Ycvx,nu_p,A,B,C,D,E,F,G,alpha,lambda_mu,gamma_inv_squared,constraints) :
        print("nonlinear funnel")
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw
        for i in range(N) :
            Qi = self.Sx@Qcvx[i]@self.Sx
            Yi = self.Su@Ycvx[i]@self.Sx
            Qi_next = self.Sx@Qcvx[i+1]@self.Sx
            
            LMI11 = alpha*Qi - lambda_mu * Qi
            LMI21 = np.zeros((ip,ix))
            LMI31 = np.zeros((iw,ix))
            LMI41 = A[i]@Qi+B[i]@Yi
            LMI51 = C@Qi+D@Yi
            
            LMI22 = nu_p[i] * np.eye(ip)
            LMI32 = np.zeros((iw,ip))
            LMI42 = nu_p[i]*E
            LMI52 = np.zeros((iq,ip))

            LMI33 = lambda_mu * np.eye(iw)
            LMI43 = F[i]
            LMI53 = G

            LMI44 = Qi_next
            LMI54 = np.zeros((iq,ix))

            # LMI55 = self.nu_p * (1/gamma[i]**2) * np.eye(iq)
            LMI55 = nu_p[i] * gamma_inv_squared[i] * np.eye(iq)

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

        self.prob.solve(solver=cvx.MOSEK)
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

        return Qnew,Knew,Ynew,self.prob.status,self.cvx_cost['l'].value

    def update_beta(self,Q,K,gamma,alpha) :
        ix,iu,N = self.ix,self.iu,self.N
        iq,ip,iw = self.iq,self.ip,self.iw
        
        beta_hat = np.zeros(N+1)
        beta_hat[0] = 1
        beta = np.zeros(N+1)
        for idx_beta in range(N) :
            A_cl = self.A[idx_beta] + self.B[idx_beta]@K[idx_beta]
            C_cl = self.C + self.D@K[idx_beta]

            S0 = np.hstack((A_cl,self.E,self.F[idx_beta])).T@np.linalg.inv(Q[idx_beta+1])@np.hstack((A_cl,self.E,self.F[idx_beta]))
            # print_np(S0)

            S1 = np.block([[np.linalg.inv(Q[idx_beta]),np.zeros((ix,ip+iw))],
                        [np.zeros((ip+iw,ix+ip+iw))]])
            # print_np(S1)
            S2_tmp = np.block([[C_cl,np.zeros((iq,ip)),self.G],
                            [np.zeros((ip,ix)),np.eye(ip),np.zeros((ip,iw))]])
            S2_tmp2 = np.block([[-(gamma[idx_beta]**2)*np.eye(iq),np.zeros((iq,ip))],[np.zeros((ip,iq)),np.eye(iq)]])
            S2 = S2_tmp.T@S2_tmp2@S2_tmp
            # print_np(S2)
            S3 = np.block([[np.zeros((ix+ip,ix+ip+iw))],
                        [np.zeros((iw,ix+ip)),np.eye(iw)]])
            # print_np(S3)

            lambda1 = cvx.Variable(1)
            lambda2 = cvx.Variable(1)
            lambda3 = cvx.Variable(1)

            constraints = []
            constraints.append(lambda1 >= 0)
            constraints.append(lambda2 >= 0)
            constraints.append(lambda3 >= 0)
            constraints.append(S0 - lambda1*S1 - lambda2*S2 - lambda3*S3 << 0)
            cost = lambda1 + lambda3

            prob = cvx.Problem(cvx.Minimize(cost), constraints)
            prob.solve(solver=cvx.MOSEK)
            beta_hat[idx_beta+1] = prob.value
            # print(idx_beta+1, prob.value)
        for idx_beta in range(N+1) :
            if idx_beta == 0 :
                beta[idx_beta] = 1
            elif idx_beta == 1 :
                beta[idx_beta] = beta_hat[idx_beta]
            else :
                beta[idx_beta] = max(alpha*beta[idx_beta-1],beta_hat[idx_beta])

        return beta


