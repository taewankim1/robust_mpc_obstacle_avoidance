import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import cvxpy as cvx
import time
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))

from Scaling import TrajectoryScaling
from PTR import PTR

class PTR_in_MPC(PTR):
    def __init__(self,name,horizon,tf,maxIter,Model,Cost,Const,Scaling=None,type_discretization='zoh',
        w_c=1,
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
        self.w_c = w_c
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

        # optimization variables
        xcvx = cvx.Variable((N+1,ix))
        ucvx = cvx.Variable((N+1,iu))
        vc = cvx.Variable((N,ix))
        bf = cvx.Variable((N+1))

        # reference trajectory
        xbar_unscaled = cvx.Parameter((N+1,ix))
        ubar_unscaled = cvx.Parameter((N+1,iu))

        # nominal trajectory to follow
        xtrj = cvx.Parameter((N+1,ix))
        utrj = cvx.Parameter((N+1,iu))

        # boundary  parameters
        xi = cvx.Parameter(ix)

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

        num_obs = self.const.num_obs
        refobs = []
        for i in range(N+1) :
            if num_obs == 0 :
                refobs.append(None)
            else :
                refobs.append(cvx.Parameter((num_obs,4))) # a,b, sqrt(a.TQa)

        constraints = []
        # boundary conditions
        constraints.append(Sx@xcvx[0] + sx == xi)

        # state and input contraints
        for i in range(0,N+1) : 
            constraints += self.const.forward(Sx@xcvx[i]+sx,
                Su@ucvx[i]+su,
                Sx@xbar_unscaled[i]+sx,
                Su@ubar_unscaled[i]+su,
                refobs[i],bf[i]
                )

        # model constraints
        for i in range(0,N) :
            if self.type_discretization == "zoh" :
                constraints.append(xcvx[i+1]+iSx@sx == iSx@A[i]@(Sx@xcvx[i]+sx)
                    +iSx@B[i]@(Su@ucvx[i]+su)
                    +iSx@s[i]*self.tf
                    +iSx@z[i]
                    +iSx@vc[i] 
                    )
            elif self.type_discretization == "foh" :
                constraints.append(xcvx[i+1]+iSx@sx == iSx@A[i]@(Sx@xcvx[i]+sx)
                    +iSx@Bm[i]@(Su@ucvx[i]+su)
                    +iSx@Bp[i]@(Su@ucvx[i+1]+su)
                    +iSx@s[i]*self.tf
                    +iSx@z[i]
                    +iSx@vc[i] 
                    )

        # cost
        cost_c = []
        cost_vc = []
        cost_tr = []

        for i in range(0,N+1) :
            cost_c.append(self.w_c * self.cost.estimate_cost_cvx(Sx@xcvx[i]+sx,Su@ucvx[i]+su,xtrj[i],utrj[i],i))
            if i < N :
                cost_vc.append(cvx.norm(vc[i],1))
            cost_vc.append(cvx.norm(bf[i],1))
            cost_tr.append(cvx.quad_form(xcvx[i]-xbar_unscaled[i],np.eye(ix)) +
                    cvx.quad_form(ucvx[i]-ubar_unscaled[i],np.eye(iu)))

        l_c = cvx.sum(cost_c)
        l_vc = cvx.sum(cost_vc)
        l_tr = cvx.sum(cost_tr)
        l_all = self.w_c*l_c + self.w_vc*l_vc + self.w_tr*l_tr
        self.prob = cvx.Problem(cvx.Minimize(l_all), constraints)
        print("Is DPP? ",self.prob.is_dcp(dpp=True))

        # save variables
        self.cvx_variables = {}
        self.cvx_variables['xcvx'] = xcvx
        self.cvx_variables['ucvx'] = ucvx
        self.cvx_variables['vc'] = vc
        # save params
        self.cvx_params = {}
        self.cvx_params['xtrj'] = xtrj
        self.cvx_params['utrj'] = utrj
        self.cvx_params['xbar_unscaled'] = xbar_unscaled
        self.cvx_params['ubar_unscaled'] = ubar_unscaled
        self.cvx_params['refobs'] = refobs
        self.cvx_params['xi'] = xi
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
        self.cvx_cost['l_c'] = l_c
        self.cvx_cost['l_vc'] = l_vc
        self.cvx_cost['l_tr'] = l_tr

    def cvxopt(self,cobs,Hobs) :
        # state & input & horizon size
        ix = self.model.ix
        iu = self.model.iu
        N = self.N

        Sx,iSx,sx,Su,iSu,su = self.Scaling.get_scaling()
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

        self.cvx_params['xtrj'].value = self.xtrj
        self.cvx_params['utrj'].value = self.utrj

        xbar_unscaled = np.zeros_like(self.x)
        ubar_unscaled = np.zeros_like(self.u)
        for i in range(N+1) :
            xbar_unscaled[i] = iSx@(self.x[i]-sx)
            ubar_unscaled[i] = iSu@(self.u[i]-su)
        self.cvx_params['xbar_unscaled'].value = xbar_unscaled
        self.cvx_params['ubar_unscaled'].value = ubar_unscaled

        num_obs = self.const.num_obs
        assert num_obs <= 1 # TODO
        def get_obs_ab(c,H,xbar) :
            hr = 1 - np.linalg.norm(H@(xbar[0:2]-c))
            dhdr = - (H.T@H@(xbar[0:2]-c)/np.linalg.norm(H@(xbar[0:2]-c))).T
            a = dhdr
            b = dhdr@xbar[0:2] - hr
            return  a,b
        if num_obs != 0 :
            for i in range(N+1) :
                tmp = np.zeros((num_obs,4))
                a,b = get_obs_ab(cobs[i],Hobs[i],self.x[i])
                tmp[0] = np.hstack((a,b,0))
                self.cvx_params['refobs'][i].value = tmp

        error = False
        try : 
            self.prob.solve(verbose=False,solver=cvx.ECOS,warm_start=False)
            # self.prob.solve(verbose=False,solver=cvx.GUROBI,warm_start=False)
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
        return self.cvx_cost['l_all'].value,self.cvx_cost['l_c'].value, \
                self.cvx_cost['l_vc'].value,self.cvx_cost['l_tr'].value, \
                xnew,unew,vc,error
                   
        
    def run(self,x0,u0,xi,xtrj,utrj,cobs,Hobs):
        # initial trajectory
        self.x = x0
        self.u = u0

        self.c_all = 0
        self.c_c = 0
        self.c_vc = 0
        self.c_tr = 0

        # trajectory to track
        self.xtrj = xtrj
        self.utrj = utrj
        
        # initial condition
        self.xi = xi
        
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
            # differentiate dynamics
            tic = time.time()
            self.A,self.B,self.Bm,self.Bp,self.s,self.z,self.x_prop,self.x_prop_n = self.get_linearized_matrices(self.x,self.u,self.delT,self.tf)
            history_iter['derivs'] = time.time() - tic

            # step2. cvxopt
            tic = time.time()
            l_all,l_c,l_vc,l_tr,self.xnew,self.unew,self.vcnew,error = self.cvxopt(cobs,Hobs)
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
            self.xfwd,self.ufwd,xprop,tprop = self.forward_multiple(self.xnew,self.unew,self.tf,iteration)
            history_iter['forward'] = time.time() - tic

            # check the boundary condtion
            self.dyn_error = np.max(np.linalg.norm(self.xfwd-self.xnew,axis=1))
            l_vc_actual = np.sum(np.linalg.norm(self.xfwd-self.xnew,1,axis=1))
            l_actual = self.w_c*l_c + self.w_vc * l_vc_actual
            if iteration > 0 :
                reduction_actual = self.c_actual - l_actual
                reduction_linear = self.c_actual - (self.w_c*l_c + self.w_vc*l_vc)
                reduction_ratio = reduction_actual / reduction_linear
            else : 
                reduction_ratio = 0

            # step4. accept step, draw graphics, print status 
            # accept changes
            self.x = self.xnew
            self.u = self.unew

            self.delT = self.tf/self.N
            self.vc = self.vcnew
            self.c_actual = l_actual
            self.c_all = l_all
            self.c_c = l_c
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
            history_iter['c_c'] = self.c_c
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
                    self.tf,
                    self.c_c,
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
                else :
                    print("reach max iteration - ",self.maxIter)
                total_num_iter = iteration+1

        return self.xfwd,self.ufwd,self.x,self.u,self.tf, \
                    total_num_iter, \
                    l_all,l_c,l_vc,l_tr, \
                    history




  


        
        
        
        
        
        
        
        
        
        
        
        


