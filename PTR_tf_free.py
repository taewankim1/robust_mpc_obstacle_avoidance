from __future__ import division
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
from PTR import PTR

from Scaling import TrajectoryScaling

class PTR_tf_free(PTR):
    def __init__(self,name,horizon,tf,maxIter,Model,Cost,Const,Scaling=None,type_discretization='zoh',
        w_tf=1,
        w_vc=1e4,w_tr=1e-3,
        tol_vc=1e-10,tol_tr=1e-3,tol_dyn=1e-3,verbosity=True) :
        self.name = name
        self.model = Model
        self.cost = Cost
        self.const = Const
        self.N = horizon
        self.tf = tf
        self.delT = tf/horizon
        if Scaling is None :
            self.Scaling = TrajectoryScaling() 
            self.Scaling.S_sigma = 1
            self.flag_update_scale = True
        else :
            self.Scaling = Scaling
            self.flag_update_scale = False
        
        # cost optimization
        self.verbosity = verbosity
        self.w_tf = w_tf
        self.w_vc = w_vc
        self.w_tr = w_tr
        self.tol_tr = tol_tr
        self.tol_vc = tol_vc
        self.tol_dyn = tol_dyn
        self.maxIter = maxIter
        self.type_discretization = type_discretization   
        self.initialize()
        self.cvx_initialize()

    def initialize(self) :
        self.A = np.zeros((self.N,self.model.ix,self.model.ix))
        self.B = np.zeros((self.N,self.model.ix,self.model.iu))  
        self.Bm = np.zeros((self.N,self.model.ix,self.model.iu))  
        self.Bp = np.zeros((self.N,self.model.ix,self.model.iu))  
        self.s = np.zeros((self.N,self.model.ix))
        self.z = np.zeros((self.N,self.model.ix))

    def cvx_initialize(self) :
        ix = self.model.ix
        iu = self.model.iu
        N = self.N

        Sx,iSx,sx,Su,iSu,su = self.Scaling.get_scaling()
        Ssigma = self.Scaling.S_sigma

        # optimization variables
        xcvx = cvx.Variable((N+1,ix))
        ucvx = cvx.Variable((N+1,iu))
        vc = cvx.Variable((N,ix))
        sigma = cvx.Variable(nonneg=True)

        # reference trajectory
        xbar_unscaled = cvx.Parameter((N+1,ix))
        ubar_unscaled = cvx.Parameter((N+1,iu))
        sigmabar_unscaled = cvx.Parameter(nonneg=True)

        # boundary  parameters
        xi = cvx.Parameter(ix)
        xf = cvx.Parameter(ix)

        # Matrices
        A,s,z = [],[],[]
        for i in range(N) :
            A.append(cvx.Parameter((ix,ix)))
            s.append(cvx.Parameter(ix))
            z.append(cvx.Parameter(ix))

        if self.type_discretization == "zoh" :
            B = []
            for i in range(N) :
                B.append(cvx.Parameter((ix,iu)))
        elif self.type_discretization == "foh" :
            Bm,Bp = [],[]
            for i in range(N) :
                Bm.append(cvx.Parameter((ix,iu)))
                Bp.append(cvx.Parameter((ix,iu)))
        else :
            print("type discretization should be zoh or foh")

        num_obs = len(self.const.c)
        refobs = []
        for i in range(N+1) :
            if num_obs == 0 :
                refobs.append(None)
            else :
                refobs.append(cvx.Parameter((num_obs,4))) # a,b, sqrt(a.TQa)

        constraints = []
        # boundary conditions
        constraints.append(Sx@xcvx[0] + sx == xi)
        constraints.append(Sx@xcvx[-1] + sx == xf)

        # state and input contraints
        for i in range(0,N+1) : 
            constraints += self.const.forward(Sx@xcvx[i]+sx,
                Su@ucvx[i]+su,
                Sx@xbar_unscaled[i]+sx,
                Su@ubar_unscaled[i]+su,
                refobs[i],
                )

        # model constraints
        for i in range(0,N) :
            if self.type_discretization == "zoh" :
                constraints.append(xcvx[i+1]+iSx@sx == iSx@A[i]@(Sx@xcvx[i]+sx)
                    +iSx@B[i]@(Su@ucvx[i]+su)
                    +iSx@s[i]*Ssigma*sigma
                    +iSx@z[i]
                    +iSx@vc[i] 
                    )
            elif self.type_discretization == "foh" :
                constraints.append(xcvx[i+1]+iSx@sx == iSx@A[i]@(Sx@xcvx[i]+sx)
                    +iSx@Bm[i]@(Su@ucvx[i]+su)
                    +iSx@Bp[i]@(Su@ucvx[i+1]+su)
                    +iSx@s[i]*Ssigma*sigma
                    +iSx@z[i]
                    +iSx@vc[i] 
                    )

        # cost
        cost_tf = []
        cost_vc = []
        cost_tr = []

        cost_tf.append(sigma*Ssigma)
        for i in range(0,N+1) :
            if i < N :
                cost_vc.append(cvx.norm(vc[i],1))
            cost_tr.append(cvx.quad_form(xcvx[i]-xbar_unscaled[i],np.eye(ix)) +
                    cvx.quad_form(ucvx[i]-ubar_unscaled[i],np.eye(iu)))
        cost_tr.append((sigma-sigmabar_unscaled)**2)

        l_tf = cvx.sum(cost_tf)
        l_vc = cvx.sum(cost_vc)
        l_tr = cvx.sum(cost_tr)
        l_all = self.w_tf*l_tf + self.w_vc*l_vc + self.w_tr*l_tr
        self.prob = cvx.Problem(cvx.Minimize(l_all), constraints)
        print("Is DPP? ",self.prob.is_dcp(dpp=True))

        # save variables
        self.cvx_variables = {}
        self.cvx_variables['xcvx'] = xcvx
        self.cvx_variables['ucvx'] = ucvx
        self.cvx_variables['vc'] = vc
        self.cvx_variables['sigma'] = sigma
        # save params
        self.cvx_params = {}
        self.cvx_params['xbar_unscaled'] = xbar_unscaled
        self.cvx_params['ubar_unscaled'] = ubar_unscaled
        self.cvx_params['sigmabar_unscaled'] = sigmabar_unscaled
        self.cvx_params['refobs'] = refobs
        self.cvx_params['xi'] = xi
        self.cvx_params['xf'] = xf
        self.cvx_params['A'] = A
        if self.type_discretization == "zoh" :
            self.cvx_params['B'] = B
        elif self.type_discretization == "foh" :
            self.cvx_params['Bm'] = Bm
            self.cvx_params['Bp'] = Bp
        self.cvx_params['s'] = s
        self.cvx_params['z'] = z
        # save cost
        self.cvx_cost = {}
        self.cvx_cost['l_all'] = l_all
        self.cvx_cost['l_tf'] = l_tf
        self.cvx_cost['l_vc'] = l_vc
        self.cvx_cost['l_tr'] = l_tr

    def cvxopt(self) :
        # state & input & horizon size
        ix = self.model.ix
        iu = self.model.iu
        N = self.N

        Sx,iSx,sx,Su,iSu,su = self.Scaling.get_scaling()
        Ssigma = self.Scaling.S_sigma
        # params
        for i in range(N) :
            self.cvx_params['A'][i].value = self.A[i]
            if self.type_discretization == "zoh" : 
                self.cvx_params['B'][i].value = self.B[i]
            elif self.type_discretization == "foh" :
                self.cvx_params['Bm'][i].value = self.Bm[i]
                self.cvx_params['Bp'][i].value = self.Bp[i]
            self.cvx_params['s'][i].value = self.s[i]
            self.cvx_params['z'][i].value = self.z[i]
        self.cvx_params['xi'].value = self.xi
        self.cvx_params['xf'].value = self.xf

        xbar_unscaled = np.zeros_like(self.x)
        ubar_unscaled = np.zeros_like(self.u)
        for i in range(N+1) :
            xbar_unscaled[i] = iSx@(self.x[i]-sx)
            ubar_unscaled[i] = iSu@(self.u[i]-su)
        sigmabar_unscaled = self.tf/Ssigma

        self.cvx_params['xbar_unscaled'].value = xbar_unscaled
        self.cvx_params['ubar_unscaled'].value = ubar_unscaled
        self.cvx_params['sigmabar_unscaled'].value = sigmabar_unscaled

        num_obs = len(self.const.c)
        def get_obs_ab(c,H,xbar) :
            hr = 1 - np.linalg.norm(H@(xbar[0:2]-c))
            dhdr = - (H.T@H@(xbar[0:2]-c)/np.linalg.norm(H@(xbar[0:2]-c))).T
            a = dhdr
            b = dhdr@xbar[0:2] - hr
            return  a,b
        if num_obs != 0 :
            for i in range(N) :
                tmp = np.zeros((num_obs,4))
                for j in range(num_obs) :
                    a,b = get_obs_ab(self.const.c[j],self.const.H[j],self.x[i])
                    tmp[j] = np.hstack((a,b,0))
                self.cvx_params['refobs'][i].value = tmp

        error = False
        try : 
            # self.prob.solve(verbose=False,solver=cvx.ECOS,warm_start=False)
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
            sigmanew = self.cvx_variables['sigma'].value * Ssigma
            vc = self.cvx_variables['vc'].value
        except ValueError :
            print(self.prob.status,"FAIL: ValueError")
            error = True
        except TypeError :
            print(self.prob.status,"FAIL: TypeError")
            error = True
        # print(lamnew)
        return self.cvx_cost['l_all'].value,self.cvx_cost['l_tf'].value, \
                self.cvx_cost['l_vc'].value,self.cvx_cost['l_tr'].value, \
                xnew,unew,sigmanew,vc,error

    def run(self,x0,u0,xi,xf=None):
        # initial trajectory
        self.x = x0
        self.u = u0

        self.c_all = 0
        self.c_tf = 0
        self.c_vc = 0
        self.c_tr = 0

        # initial condition
        self.xi = xi

        # final condition
        self.xf = xf
        
        # state & input & horizon size
        ix = self.model.ix
        iu = self.model.iu
        N = self.N
        
        # initialize history data structure
        history = []
        
        # iterations starts!!
        total_num_iter = 0
        tic_bottom = time.time()
        for iteration in range(self.maxIter) :
            history_iter = {}
            # differentiate dynamics and cost
            tic = time.time()
            self.A,self.B,self.Bm,self.Bp,self.s,self.z,self.x_prop,self.x_prop_n = self.get_linearized_matrices(self.x,self.u,self.delT,self.tf)
            history_iter['derivs'] = time.time() - tic

            # step2. cvxopt
            tic = time.time()
            l_all,l_tf,l_vc,l_tr,self.xnew,self.unew,self.tfnew,self.vcnew,error = self.cvxopt()
            history_iter['cvxopt'] = time.time() - tic
            history_iter['flag_cvxopt_error'] = error
            if error is True :
                if self.verbosity == True :
                    print("├──────┴───────────┴──────────┴─────────┴───────────┴────────────┴─────────┴─────────┴─────────┤\n")
                    print('│                FAIL : cvxopt failed                                                          │\n')
                    print("└──────────────────────────────────────────────────────────────────────────────────────────────┘\n")
                history.append(history_iter)
                break

            # step3. evaluate step
            tic = time.time()
            # self.xfwd,self.ufwd,xprop = self.forward_single(self.xnew[0,:],self.unew,self.tfnew,iteration)
            self.xfwd,self.ufwd,xprop,tprop = self.forward_multiple(self.xnew,self.unew,self.tfnew,iteration)
            history_iter['forward'] = time.time() - tic

            # check the boundary condtion
            self.dyn_error = np.max(np.linalg.norm(self.xfwd-self.xnew,axis=1))
            l_vc_actual = np.sum(np.linalg.norm(self.xfwd-self.xnew,1,axis=1))
            l_actual = self.w_tf*l_tf + self.w_vc * l_vc_actual
            if iteration > 0 :
                reduction_actual = self.c_actual - l_actual
                reduction_linear = self.c_actual - (self.w_tf*l_tf + self.w_vc*l_vc)
                reduction_ratio = reduction_actual / reduction_linear
            else : 
                reduction_ratio = 0


            # step4. accept step, draw graphics, print status 
            # accept changes
            self.x = self.xnew
            self.u = self.unew
            self.tf = self.tfnew

            self.delT = self.tf/self.N
            self.vc = self.vcnew
            self.c_actual = l_actual
            self.c_all = l_all
            self.c_tf = l_tf
            self.c_vc = l_vc 
            self.c_tr = l_tr

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

            history_iter['c_actual'] = self.c_actual
            history_iter['c_all'] = self.c_all
            history_iter['c_tf'] = self.c_tf
            history_iter['c_vc'] = self.c_vc
            history_iter['c_tr'] = self.c_tr

            history_iter['flag_vc'] = flag_vc
            history_iter['flag_tr'] = flag_tr
            history_iter['flag_dyn'] = flag_dyn

            history_iter['w_tr'] = self.w_tr
            history_iter['time_cumulative'] = time.time() - tic_bottom
            history.append(history_iter)

            if iteration == 0 and self.verbosity == True :
                print("┌──────────────────────────────────────────────────────────────────────────────────────────────┐\n")
                print("│                          ..:: Aircraft Landing by SCP ::..                                   │\n")
                print("├──────┬───────────┬──────────┬─────────┬───────────┬────────────┬─────────┬─────────┬─────────┤\n")
                print("│ iter │  total    │ final    │ thrust  │ control   │ reduction  │ vc      │ tr      │ dyn     │\n")
                print("│      │  cost     │ time [s] │ energy  │ rate      │ ratio      │ (log10) │         │         │\n")
                print("├──────┼───────────┼──────────┼─────────┼───────────┼────────────┼─────────┼─────────┼─────────┤\n")

            if self.verbosity == True:
                print("│%-6d│%-11.3f│%-10.3f│%-9.3g│%-11.3g│%-12.3g│%-1d(%-6.2f)│%-1d(%-6.3f)│%-1d(%-6.3f)│" % ( 
                    iteration+1,
                    self.c_all,
                    self.c_tf,
                    0,
                    0,
                    reduction_ratio,
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

        return self.xfwd,self.ufwd,self.x,self.u,self.tf, \
                    total_num_iter, \
                    l_all,l_tf,l_vc,l_tr, \
                    history
        


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


