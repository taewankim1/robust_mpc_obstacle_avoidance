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

class trajopt:
    def __init__(self,name,horizon,tf,maxIter,Model,Cost,Const,Scaling=None,
                        w_c=1,w_vc=1e4,w_tr=1e-3,tol_vc=1e-10,tol_tr=1e-3,tol_dyn=1e-3,
                        flag_policyopt=False,verbosity=True):
        self.name = name
        self.model = Model
        self.const = Const
        self.cost = Cost
        self.N = horizon
        self.tf = tf
        self.delT = tf/horizon
        if Scaling is None :
            self.Scaling = TrajectoryScaling() 
            self.flag_update_scale = True
        else :
            self.Scaling = Scaling
            self.flag_update_scale = False
        
        # cost optimization
        self.verbosity = verbosity
        self.w_c = w_c
        self.w_vc = w_vc
        self.w_tr = w_tr
        # self.tol_fun = 1e-6
        self.tol_tr = tol_tr
        self.tol_vc = tol_vc
        self.tol_dyn = tol_dyn
        self.maxIter = maxIter
        self.last_head = True
        self.flag_policyopt = flag_policyopt
        self.initialize()
        self.cvx_initialize()

    def initialize(self) :

        self.A = np.zeros((self.N,self.model.ix,self.model.ix))
        self.B = np.zeros((self.N,self.model.ix,self.model.iu))  
        self.Bm = np.zeros((self.N,self.model.ix,self.model.iu))  
        self.Bp = np.zeros((self.N,self.model.ix,self.model.iu))  
        self.s = np.zeros((self.N,self.model.ix))
        self.z = np.zeros((self.N,self.model.ix))

    def get_model(self) :
        return self.A,self.B,self.s,self.z,self.vc

    def forward_multiple(self,x,u,tf,iteration) :
        N = self.N
        delT = tf/N
        ix = self.model.ix
        iu = self.model.iu

        def dfdt(t,V,um,up) :
            u = um
            x = V.reshape((N,ix))
            f = self.model.forward(x,u)
            return f.flatten()

        x0 = x[0:N].flatten()
        if iteration < 10 :
            sol = solve_ivp(dfdt,(0,delT),x0,args=(u[0:N],u[1:]))
        else :
            sol = solve_ivp(dfdt,(0,delT),x0,args=(u[0:N],u[1:]),method='RK45',rtol=1e-6,atol=1e-10)
        xnew = np.zeros((N+1,ix))
        xnew[0] = x[0]
        xnew[1:] = sol.y[:,-1].reshape((N,-1))
        xprop = sol.y.T
        tprop = sol.t
        return xnew,np.copy(u),xprop,tprop

    def forward_single(self,x0,u,tf,iteration) :
        N = self.N
        delT = tf/N
        ix = self.model.ix
        iu = self.model.iu

        def dfdt(t,x,um,up) :
            if self.type_discretization == "zoh" :
                u = um
            elif self.type_discretization == "foh" :
                alpha = (delT - t) / delT
                beta = t / delT
                u = alpha * um + beta * up
            return np.squeeze(self.model.forward(x,u))

        xnew = np.zeros((N+1,ix))
        xnew[0] = x0
        xprop = []

        for i in range(N) :
            if iteration < 10 :
                sol = solve_ivp(dfdt,(0,delT),xnew[i],args=(u[i],u[i+1]))
            else :
                sol = solve_ivp(dfdt,(0,delT),xnew[i],args=(u[i],u[i+1]),method='RK45',rtol=1e-6,atol=1e-10)
            xnew[i+1] = sol.y[:,-1]
            xprop.append(sol.y[:,-1])
        return xnew,np.copy(u),xprop

    def cvx_initialize(self) :
        ix = self.model.ix
        iu = self.model.iu
        N = self.N

        Sx,iSx,sx,Su,iSu,su = self.Scaling.get_scaling()

        # optimization variables
        xcvx = cvx.Variable((N+1,ix))
        ucvx = cvx.Variable((N+1,iu))
        vc = cvx.Variable((N,ix))

        # reference trajectory
        xbar_unscaled = cvx.Parameter((N+1,ix))
        ubar_unscaled = cvx.Parameter((N+1,iu))

        # boundary  parameters
        xi = cvx.Parameter(ix)
        xf = cvx.Parameter(ix)

        # Matrices and Q,K
        A,B,s,z = [],[],[],[]
        Q,K = [],[]
        for i in range(N) :
            A.append(cvx.Parameter((ix,ix)))
            B.append(cvx.Parameter((ix,iu)))
            s.append(cvx.Parameter(ix))
            z.append(cvx.Parameter(ix))
            Q.append(cvx.Parameter((ix,ix)))
            K.append(cvx.Parameter((iu,ix)))
        Q.append(cvx.Parameter((ix,ix)))
                   
        num_obs = len(self.const.c)
        refobs = []
        aQav = []
        aQaw = []
        for i in range(N) :
            refobs.append(cvx.Parameter((num_obs,4))) # a,b, sqrt(a.TQa)
            aQav.append(cvx.Parameter((1,1)))# sqrt(aKQK.Ta)
            aQaw.append(cvx.Parameter((1,1)))# sqrt(aKQK.Ta)

        constraints = []
        # boundary conditions
        constraints.append(Sx@xcvx[0] + sx == xi)
        constraints.append(Sx@xcvx[-1] + sx == xf)

        # state and input contraints
        for i in range(0,N) : 
            constraints += self.const.forward(Sx@xcvx[i]+sx,
                Su@ucvx[i]+su,
                Sx@xbar_unscaled[i]+sx,
                Su@ubar_unscaled[i]+su,
                Q[i],K[i],
                refobs[i],
                aQav[i],aQaw[i])

        # model constraints
        for i in range(0,N) :
            constraints.append(Sx@xcvx[i+1]+sx == A[i]@(Sx@xcvx[i]+sx)+B[i]@(Su@ucvx[i]+su)
                +self.tf*s[i]
                +z[i]
                +vc[i])

                # cost
        c_control = []
        c_vc = []
        c_tr = []
        for i in range(0,N+1) :
            if i < N :
                c_vc.append(1 * cvx.norm(vc[i],1))
            c_control.append(1 * self.cost.estimate_cost_cvx(Sx@xcvx[i]+
                sx,Su@ucvx[i]+su,i))
            c_tr.append(cvx.quad_form(xcvx[i]-xbar_unscaled[i],np.eye(ix)) +
                    cvx.quad_form(ucvx[i]-ubar_unscaled[i],np.eye(iu)))

        l_control = cvx.sum(c_control)
        l_vc = cvx.sum(c_vc)
        l_tr = cvx.sum(c_tr)

        l_all = self.w_c*l_control + self.w_vc*l_vc + self.w_tr*l_tr
        self.prob = cvx.Problem(cvx.Minimize(l_all), constraints)
        print("Is DPP? ",self.prob.is_dcp(dpp=True))

        # save variables
        self.cvx_variables = {}
        self.cvx_variables['xcvx'] = xcvx
        self.cvx_variables['ucvx'] = ucvx
        self.cvx_variables['vc'] = vc
        # save params
        self.cvx_params = {}
        self.cvx_params['xbar_unscaled'] = xbar_unscaled
        self.cvx_params['ubar_unscaled'] = ubar_unscaled
        self.cvx_params['refobs'] = refobs
        self.cvx_params['aQav'] = aQav
        self.cvx_params['aQaw'] = aQaw
        self.cvx_params['xi'] = xi
        self.cvx_params['xf'] = xf
        self.cvx_params['A'] = A
        self.cvx_params['B'] = B
        self.cvx_params['s'] = s
        self.cvx_params['z'] = z
        self.cvx_params['Q'] = Q
        self.cvx_params['K'] = K
        # save cost
        self.cvx_cost = {}
        self.cvx_cost['l_all'] = l_all
        self.cvx_cost['l_control'] = l_control
        self.cvx_cost['l_vc'] = l_vc
        self.cvx_cost['l_tr'] = l_tr

    def cvxopt(self) :
        # state & input & horizon size
        ix = self.model.ix
        iu = self.model.iu
        N = self.N

        Sx,iSx,sx,Su,iSu,su = self.Scaling.get_scaling()

        # params
        for i in range(N) :
            self.cvx_params['A'][i].value = self.A[i]
            self.cvx_params['B'][i].value = self.B[i]
            self.cvx_params['s'][i].value = self.s[i]
            self.cvx_params['z'][i].value = self.z[i]
            self.cvx_params['Q'][i].value = self.Q[i]
            self.cvx_params['K'][i].value = self.K[i]
        self.cvx_params['Q'][-1].value = self.Q[-1]
        self.cvx_params['xi'].value = self.xi
        self.cvx_params['xf'].value = self.xf

        xbar_unscaled = np.zeros_like(self.x)
        ubar_unscaled = np.zeros_like(self.u)
        for i in range(N+1) :
            xbar_unscaled[i] = iSx@(self.x[i]-sx)
            ubar_unscaled[i] = iSu@(self.u[i]-su)
        self.cvx_params['xbar_unscaled'].value = xbar_unscaled
        self.cvx_params['ubar_unscaled'].value = ubar_unscaled

        num_obs = len(self.const.c)
        def get_obs_ab(c,H,xbar) :
            hr = 1 - np.linalg.norm(H@(xbar[0:2]-c))
            dhdr = - (H.T@H@(xbar[0:2]-c)/np.linalg.norm(H@(xbar[0:2]-c))).T
            a = dhdr
            b = dhdr@xbar[0:2] - hr
            return  a,b

        for i in range(N) :
            tmp = np.zeros((num_obs,4))
            for j in range(num_obs) :
                a,b = get_obs_ab(self.const.c[j],self.const.H[j],self.x[i])
                tmp[j] = np.hstack((a,b,np.sqrt(a.T@self.Q[i][0:2,0:2]@a)))
            self.cvx_params['refobs'][i].value = tmp
        av = np.expand_dims(np.array([1,0]),1)
        aw = np.expand_dims(np.array([0,1]),1)
        for i in range(N) :
            self.cvx_params['aQav'][i].value =  np.sqrt(av.T@self.K[i]@self.Q[i]@self.K[i].T@av)
            self.cvx_params['aQaw'][i].value =  np.sqrt(aw.T@self.K[i]@self.Q[i]@self.K[i].T@aw)

        error = False
        try : 
            # self.prob.solve(verbose=False,solver=cvx.ECOS,warm_start=False)
            # self.prob.solve(verbose=False,solver=cvx.ECOS,ignore_dpp=True)
            # self.prob.solve(verbose=False,solver=cvx.GUROBI,ignore_dpp=True)
            self.prob.solve(verbose=False,solver=cvx.GUROBI,warm_start=False)
        except cvx.error.SolverError :
            error = True

        if self.prob.status == cvx.OPTIMAL_INACCURATE :
            print("WARNING: inaccurate solution")
        try :
            xnew = np.zeros_like(self.x)
            unew = np.zeros_like(self.u)
            for i in range(N+1) :
                xnew[i] = Sx@self.cvx_variables['xcvx'][i].value + sx
                unew[i] = Su@self.cvx_variables['ucvx'][i].value + su
            vc = self.cvx_variables['vc'].value
        except ValueError :
            print(self.prob.status,"FAIL: ValueError")
            error = True
        except TypeError :
            print(self.prob.status,"FAIL: TypeError")
            error = True
        # print(lamnew)
        return self.cvx_cost['l_all'].value,self.cvx_cost['l_control'].value, \
                self.cvx_cost['l_vc'].value,self.cvx_cost['l_tr'].value, \
                xnew,unew,vc,error
        
    def run(self,x0,u0,xi,xf,Q=None,K=None):
        # state & input & horizon size
        ix = self.model.ix
        iu = self.model.iu
        N = self.N
        
        # initial input
        self.x = x0
        self.u = u0

        self.c_all = 0
        self.c_control = 0
        self.c_vc = 0
        self.c_tr = 0

        if Q is None :
            self.Q = np.tile(np.zeros((ix,ix)),(N+1,1,1)) 
            self.K = np.tile(np.zeros((iu,ix)),(N+1,1,1)) 
        else :
            self.Q = Q
            self.K = K


        # boundary condition
        self.xi = xi
        self.xf = xf

        # generate initial trajectory
        diverge = False
        stop = False

        self.c = 1e3
        self.cvc = 0
        self.ctr = 0

        # initialize history data structure
        history = []

        # iterations starts!!
        total_num_iter = 0
        tic_bottom = time.time()
        for iteration in range(self.maxIter) :
            history_iter = {}
            # step1. differentiate dynamics and cost
            tic = time.time()
            self.A,self.B,self.s,self.z,self._ = self.model.diff_discrete_zoh(self.x[0:N,:],self.u[0:N,:],self.delT,self.tf)
            history_iter['derivs'] = time.time() - tic
            eps_machine = np.finfo(float).eps
            self.A[np.abs(self.A) < eps_machine] = 0
            self.B[np.abs(self.B) < eps_machine] = 0
            self.Bm[np.abs(self.Bm) < eps_machine] = 0
            self.Bp[np.abs(self.Bp) < eps_machine] = 0

            # step2. cvxopt
            tic = time.time()
            l_all,l_control,l_vc,l_tr,self.xnew,self.unew,self.vcnew,error = self.cvxopt()
            history_iter['cvxopt'] = time.time() - tic
            history_iter['flag_cvxopt_error'] = error
            if error == True :
                if self.verbosity == True :
                    print("├──────┴───────────┴──────────┴─────────┴───────────┴────────────┴─────────┴─────────┴─────────┤\n")
                    print('│                FAIL : cvxopt failed                                                          │\n')
                    print("└──────────────────────────────────────────────────────────────────────────────────────────────┘\n")
                history.append(history_iter)
                break

            # step3. evaluate step
            reduction = self.c_all - l_all
            tic = time.time()
            self.xfwd,self.ufwd,xprop,tprop = self.forward_multiple(self.xnew,self.unew,self.tf,iteration)
            history_iter['forward'] = time.time() - tic
            self.dyn_error = np.max(np.linalg.norm(self.xfwd-self.xnew,axis=1))

            # step4. accept step
            self.x = self.xnew
            self.u = self.unew

            self.delT = self.tf/self.N
            self.vc = self.vcnew
            self.c_all = l_all
            self.c_control = l_control
            self.c_vc = l_vc 
            self.c_tr = l_tr
            self.c_tf = self.tf

            flag_vc = self.c_vc < self.tol_vc
            flag_tr = self.c_tr < self.tol_tr
            flag_dyn = self.dyn_error < self.tol_dyn

            history_iter['x'] = self.x
            history_iter['u'] = self.u
            history_iter['xfwd'] = self.xfwd
            history_iter['ufwd'] = self.ufwd
            history_iter['xprop'] = xprop
            history_iter['tprop'] = tprop
            history_iter['tf'] = self.tf
            history_iter['vc'] = self.vc

            history_iter['c_all'] = self.c_all
            history_iter['c_control'] = self.c_control
            history_iter['c_vc'] = self.c_vc
            history_iter['c_tr'] = self.c_tr

            history_iter['flag_vc'] = flag_vc
            history_iter['flag_tr'] = flag_tr
            history_iter['flag_dyn'] = flag_dyn

            history_iter['time_cumulative'] = time.time() - tic_bottom
            history.append(history_iter)

            if iteration == 0 and self.verbosity == True :
                print("┌──────────────────────────────────────────────────────────────────────────────────────────────┐\n")
                print("│                                              ..:: SCP ::..                                   │\n")
                print("├──────┬───────────┬──────────┬─────────┬───────────┬────────────┬─────────┬─────────┬─────────┤\n")
                print("│ iter │  total    │ final    │ input   │     -     │ total cost │ vc      │ tr      │ dyn     │\n")
                print("│      │  cost     │ time [s] │ energy  │     -     │ reduction  │ (log10) │         │         │\n")
                print("├──────┼───────────┼──────────┼─────────┼───────────┼────────────┼─────────┼─────────┼─────────┤\n")
            if self.verbosity == True:
                print("│%-6d│%-11.3f│%-10.3f│%-9.3g│%-11.3g│%-12.3g│%-1d(%-6.2f)│%-1d(%-6.3f)│%-1d(%-6.3f)│" % ( 
                                                                                    iteration+1,
                                                                                    self.c_all,
                                                                                    self.c_tf,
                                                                                    self.c_control,
                                                                                    0,
                                                                                    reduction,
                                                                                    flag_vc,
                                                                                    np.log10(self.c_vc),
                                                                                    flag_tr,
                                                                                    self.c_tr,
                                                                                    flag_dyn, 
                                                                                    self.dyn_error))
            if flag_vc and flag_tr and flag_dyn :
                if self.verbosity == True:
                    print("├──────┴───────────┴──────────┴─────────┴───────────┴────────────┴─────────┴─────────┴─────────┤\n")
                    print('│                SUCCEESS: virtual control and trust region < tol                              │\n')
                    print("└──────────────────────────────────────────────────────────────────────────────────────────────┘\n")
                    total_num_iter = iteration+1
                break
            if iteration == self.maxIter - 1 :
                if self.verbosity == True :
                    print("├──────┴───────────┴──────────┴─────────┴───────────┴────────────┴─────────┴─────────┴─────────┤\n")
                    print('│                NOT ENOUGH : reached to max iteration                                         │\n')
                    print("└──────────────────────────────────────────────────────────────────────────────────────────────┘\n")
                total_num_iter = iteration+1

        return self.xfwd,self.ufwd,self.xnew,self.unew,self.tf, \
            total_num_iter, \
            l_all,l_control,l_vc,l_tr, \
            history

    def print_eigenvalue(self,A_) :
        eig,eig_vec = np.linalg.eig(A_)
        print("(discrete) eigenvalue of A",np.max(np.real(eig)))
        if self.model.type_linearization == "numeric_central" :
            A,B = self.model.diff_numeric_central(self.x,self.u)
        elif self.model.type_linearization == "numeric_forward" :
            A,B = self.model.diff_numeric(self.x,self.u)
        elif self.model.type_linearization == "analytic" :
            A,B = self.model.diff(self.x,self.u)
        eig,eig_vec = np.linalg.eig(A)
        print("(continuous) eigenvalue of A",np.max(np.real(eig)))


        
        
        
        
        
        
        
        
        
        
        
        


