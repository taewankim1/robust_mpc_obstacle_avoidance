import matplotlib.pyplot as plt
import numpy as np
import time
import random
import scipy
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
#     print ("Values are: \n%s" % (x))
from scipy.integrate import solve_ivp
import IPython

def get_neighbor_vec(num,val,ix,list_e,list_all) :
    if num == -1 :
        list_e = []
    else :
        list_e.append(val)
    if num == ix - 1 :
        list_all.append(list_e)
        return
    get_neighbor_vec(num+1,1,ix,list_e.copy(),list_all)
    get_neighbor_vec(num+1,-1,ix,list_e.copy(),list_all)
    return list_all

def get_K(A,B,Q,R,N,ix,iu) :
    K_list = []
    for A_,B_ in zip(A,B) :
        P = scipy.linalg.solve_continuous_are(A_,B_,Q,R)
        K = -np.linalg.inv(R)@B_.T@P
        K_list.append(K)
    return np.array(K_list)

def get_K_discrete(A,B,Q,R,Q_final,N,ix,iu) :
    # Q = np.eye(ix)*1e2
    # R = np.eye(iu)
    P = np.zeros((N+1,ix,ix))
    K = np.zeros((N,iu,ix))
    P[N] = Q_final
    for i in range(N-1,-1,-1) :
        K[i] = np.linalg.inv(B[i].T@P[i+1]@B[i]+R)@B[i].T@P[i+1]@A[i]
        P[i] = (A[i]-B[i]@K[i]).T@P[i+1]@(A[i]-B[i]@K[i])+K[i].T@R@K[i]+Q
    return K

def project_ellipse(Q) : 
    # from https://github.com/tpreynolds/RSS_2020/blob/master/classes/%40cfga/cfga.m
    ix,_ = Q.shape
    Q_inv = np.linalg.inv(Q)
    ndims = 2
    T = np.zeros((ix,ndims))
    T[:,0] = np.eye(ix)[:,0]
    T[:,1] = np.eye(ix)[:,1]
    L = np.linalg.cholesky(Q_inv) 
    A = T.T@np.linalg.inv(L).T
    U,S,_ = np.linalg.svd(A,full_matrices=False)
    iS = np.linalg.inv(np.diag(S))
    P_proj = U@iS@iS@U.T
    Q_proj = np.linalg.inv(P_proj)
    return Q_proj

def get_radius_angle(Q_list) :
    radius_list = []
    angle_list = []
    for Q_ in Q_list :
        # eig,_ = np.linalg.eig(np.linalg.inv(Q_))
        # radius = np.sqrt(1/eig)
        # print("radius of x,y,theta",radius)
        A = np.array([[1,0,0],[0,1,0]])
        # Q_proj = project_ellipse(Q_) 
        Q_proj = A@Q_@A.T
        Q_inv = np.linalg.inv(Q_proj)
        eig,eig_vec = np.linalg.eig(Q_inv)
        radius = np.sqrt(1/eig)
        # print("radius of x and y",radius)
        rnew = eig_vec@np.array([[radius[0]],[0]])
        angle = np.arctan2(rnew[1],rnew[0])
        radius_list.append(radius)
        angle_list.append(angle)

    return radius_list,angle_list


def forward_full_with_K(x0,x0_s,xnom,unom,Q,Y,model,N,ix,iu,iw,delT,flag_noise) :
    # only for ZOH case
    def dfdt(t,x,unom,Qi,Qi_next,Y,w,tk,tk_next) :
        alpha = (tk_next-t)/(tk_next-tk)
        beta = (t-tk)/(tk_next-tk)
        Qt = Qi*alpha + Qi_next*beta
        K = Y@np.linalg.inv(Qt)
        xnom = x[:ix]
        xnew = x[ix:]
        unew = unom + K@(xnew-xnom)
        fnom = np.squeeze(model.forward(xnom,unom))
        fnew = np.squeeze(model.forward_noise_1(xnew,unew,w))
        return np.hstack((fnom,fnew))

    xnew = np.zeros((N+1,2*ix))
    # unew = np.zeros((N+1,iu))
    xnew[0,:ix] = xnom[0]
    xnew[0,ix:] = x0_s

    tsave = []
    usave = []
    xsave = []
    # z = np.random.randn(iw)
    # w = z / np.linalg.norm(z)
    for i in range(N) :
        if flag_noise == True :
            z = np.random.randn(iw)
            w = z / np.linalg.norm(z)
        else :
            w = np.zeros(iw)
        tk_next = delT*(i+1)
        tk = delT*i
        sol = solve_ivp(dfdt,(tk,tk_next),xnew[i],args=(unom[i],Q[i],Q[i+1],Y[i],w,tk,tk_next),max_step=delT/10,method='RK45',rtol=1e-6,atol=1e-10)
        for t_,x_ in zip(sol.t[:-1],sol.y[:,:-1].T) :
            alpha = (tk_next-t_)/(tk_next-tk)
            beta = (t_-tk)/(tk_next-tk)
            Qt = Q[i]*alpha + Q[i+1]*beta
            Kgain = Y[i]@np.linalg.inv(Qt)
            tsave.append(t_)
            xsave.append(x_)
            xnom_ = x_[:ix]
            xnew_ = x_[ix:]
            usave.append(unom[i] + Kgain@(xnew_-xnom_))
        # next state condition
        xnew[i+1] = sol.y[:,-1]
    tsave.append(sol.t[-1])
    xsave.append(sol.y[:,-1])
    xnom_ = xsave[-1][:ix]
    xnew_ = xsave[-1][ix:]
    Kgain = Y[-1]@np.linalg.inv(Q[-1])
    usave.append(unom[N-1] + Kgain@(xnew_-xnom_))
    tsave = np.array(tsave)
    xsave = np.array(xsave)
    usave = np.array(usave)
    return tsave,xsave[:,ix:],usave

def forward_full_with_K_discrete(x0,x0_s,xnom,unom,Q,Y,model,N,ix,iu,iw,delT,flag_noise) :
    # only for ZOH case
    def dfdt(t,x,u,w) :
        return np.squeeze(model.forward_uncertain(x,u,w))

    xnew = np.zeros((N+1,ix))
    # unew = np.zeros((N+1,iu))
    xnew[0] = x0_s

    tsave = []
    usave = []
    xsave = []
    wsave = []
    if flag_noise == True :
        z = np.random.randn(iw)
        w = z / np.linalg.norm(z)
    else :
        w = np.zeros(iw)
    for i in range(N) :
        tk_next = delT*(i+1)
        tk = delT*i
        K = Y[i]@np.linalg.inv(Q[i])
        u = unom[i] + K@(xnew[i]-xnom[i])
        sol = solve_ivp(dfdt,(tk,tk_next),xnew[i],args=(u,w),max_step=delT/10,method='RK45',rtol=1e-6,atol=1e-10)
        for t_,x_ in zip(sol.t[:-1],sol.y[:,:-1].T) :
            tsave.append(t_)
            xsave.append(x_)
            usave.append(u)
            wsave.append(w)
        # next state condition
        xnew[i+1] = sol.y[:,-1]
        radii = (xnew[i+1]-xnom[i+1]).T@np.linalg.inv(Q[i+1])@(xnew[i+1]-xnom[i+1])
        if radii > 1 :
            print("there is invariance violation")
    tsave.append(sol.t[-1])
    xsave.append(sol.y[:,-1])
    usave.append(u)
    wsave.append(w)
    tsave = np.array(tsave)
    xsave = np.array(xsave)
    usave = np.array(usave)
    wsave = np.array(wsave)
    return tsave,xsave,usave,wsave,xnew

def get_sample_trajectory(x0,x0_sample,xnom,unom,Q,Y,model,N,ix,iu,iw,delT,flag_noise=False) :
    tsam,xsam,usam,wsam,xsamp =[],[],[],[],[]
    for idx,x0_s in enumerate(x0_sample) :
        tsam_,xsam_,usam_,wsam_,xsamp_ = forward_full_with_K_discrete(x0,x0_s,xnom,unom,Q,Y,model,N,ix,iu,iw,delT,flag_noise)
        tsam.append(tsam_)
        xsam.append(xsam_)
        usam.append(usam_)
        wsam.append(wsam_)
        xsamp.append(xsamp_)
    return tsam,xsam,usam,wsam,xsamp
    # return np.array(tsam),np.array(xsam),np.array(usam),np.array(wsam)

def get_sample_eta_w(Q,zs_sample,zw_sample) :
    eta_sample = []
    w_sample = []

    for zs,zw in zip(zs_sample,zw_sample):
        e_s = scipy.linalg.sqrtm(Q)@zs
        eta_sample.append(e_s)

        w_sample.append(zw)
    eta_sample = np.array(eta_sample)
    w_sample = np.array(w_sample)
    return eta_sample,w_sample


def propagate_model(model,x0,u,delT) :
    ix = model.ix
    iu = model.iu
    def dfdt(t,x,u) :
        return np.squeeze(model.forward(x,u))

    sol = solve_ivp(dfdt,(0,delT),x0,args=(u,),method='RK45',rtol=1e-6,atol=1e-10)
    xnext = sol.y[:,-1]
    return xnext