import imageio
import os
from mpl_toolkits.mplot3d import art3d

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Rectangle
import numpy as np
import time
import random
from utils.utils_alg import get_radius_angle
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
#     print ("Values are: \n%s" % (x))

# vector scaling
thrust_scale = 0.1
attitude_scale = 0.3

def make_rocket3d_trajectory_fig(x,u,img_name='untitled') :
    filenames = []
    N = np.shape(x)[0]
    for k in range(N):
        
        fS = 18
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X, east')
        ax.set_ylabel('Y, north')
        ax.set_zlabel('Z, up')
        rx, ry, rz = x[k,1:4]
        qw, qx, qy, qz = x[k,7:11]

        CBI = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)],
            [2 * (qx * qy - qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz + qw * qx)],
            [2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

        dx, dy, dz = np.dot(np.transpose(CBI), np.array([0., 0., 1.]))
        if k != N-1 :
            Fx, Fy, Fz = np.dot(np.transpose(CBI), u[k, :])

        # attitude vector
        ax.quiver(rx, ry, rz, dx, dy, dz, length=attitude_scale, arrow_length_ratio=0.0, color='blue')

        # thrust vector
        ax.quiver(rx, ry, rz, -Fx, -Fy, -Fz, length=thrust_scale, arrow_length_ratio=0.0, color='red')
        scale = x[0, 3]
        ax.auto_scale_xyz([-scale / 2, scale / 2], [-scale / 2, scale / 2], [0, scale])

        pad = plt.Circle((0, 0), 0.2, color='lightgrey')
        ax.add_patch(pad)
        art3d.pathpatch_2d_to_3d(pad)

        ax.plot(x[:, 1], x[:, 2], x[:, 3])
        
        filename = '../images/{:d}.png'.format(k)
        plt.savefig(filename)
        filenames.append(filename)
        plt.close()

    with imageio.get_writer('../images/'+img_name+'.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    for filename in set(filenames):
        os.remove(filename)

def plot_rocket3d(fig, x, u, xppg=None):
    ax = fig.add_subplot(111, projection='3d')

    N = np.shape(x)[0]

    ax.set_xlabel('X, east')
    ax.set_ylabel('Y, north')
    ax.set_zlabel('Z, up')

    for k in range(N):
        rx, ry, rz = x[k,1:4]
        qw, qx, qy, qz = x[k,7:11]

        CBI = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)],
            [2 * (qx * qy - qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz + qw * qx)],
            [2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

        dx, dy, dz = np.dot(np.transpose(CBI), np.array([0., 0., 1.]))
        if k != N-1 :
            Fx, Fy, Fz = np.dot(np.transpose(CBI), u[k, :])

        # attitude vector
        ax.quiver(rx, ry, rz, dx, dy, dz, length=attitude_scale, arrow_length_ratio=0.0, color='blue')

        # thrust vector
        ax.quiver(rx, ry, rz, -Fx, -Fy, -Fz, length=thrust_scale, arrow_length_ratio=0.0, color='red')

    scale = x[0, 3]
    ax.auto_scale_xyz([-scale / 2, scale / 2], [-scale / 2, scale / 2], [0, scale])

    # pad = plt.Circle((0, 0), 0.2, color='lightgrey')
    # ax.add_patch(pad)
    # art3d.pathpatch_2d_to_3d(pad)

    ax.plot(x[:, 1], x[:, 2], x[:, 3])
    if xppg is not None :
        ax.plot(xppg[:, 1], xppg[:, 2], xppg[:, 3],'--')
#     ax.set_aspect('equal')

def make_rocket2d_trajectory_fig(x,u,img_name) :
    N = np.shape(x)[0] -1
    Fx = +np.sin(x[:,4] + u[:,0]) * u[:,1]
    Fy = -np.cos(x[:,4] + u[:,0]) * u[:,1]
    filenames = []
    for i in range(N+10) :
        fS = 18
        plt.figure(figsize=(10,10))
        plt.gca().set_aspect('equal', adjustable='box')
        if i <= N :
            index = i
        else :
            index = N
        plt.plot(x[:i+1,0], x[:i+1,1], linewidth=2.0) 
        plt.plot(0, 0,'*', linewidth=2.0)
        plt.quiver(x[index,0], x[index,1], -np.sin(x[index,4]), np.cos(x[index,4]), color='blue', width=0.003, scale=15, headwidth=1, headlength=0)
        if i < N :
            plt.quiver(x[index,0], x[index,1], Fx[index], Fy[index], color='red', width=0.003, scale=100, headwidth=1, headlength=0)
        plt.axis([-2, 5, -1, 5])
        plt.xlabel('X ()', fontsize = fS)
        plt.ylabel('Y ()', fontsize = fS)
        filename = '../images/{:d}.png'.format(i)
        plt.savefig(filename)
        filenames.append(filename)
        plt.close()

    with imageio.get_writer('../images/'+img_name+'.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    for filename in set(filenames):
        os.remove(filename)

def plot_Landing2D_trajectory (x,u,xppg=None,delT=0.1) :
    N = np.shape(x)[0]-1
    fS = 18
    Fx = +np.sin(x[:,4] + u[:,0]) * u[:,1]
    Fy = -np.cos(x[:,4] + u[:,0]) * u[:,1]
    plt.figure(1,figsize=(10,10))
    plt.plot(x[:,0], x[:,1], linewidth=2.0)
    if xppg is not None :
        plt.plot(xppg[:,0], xppg[:,1], '--',linewidth=2.0)
    plt.plot(0,0,'o')
    plt.gca().set_aspect('equal', adjustable='box')
    index = np.linspace(0,N-1,30)
    index = [int(i) for i in index]
    plt.quiver(x[index,0], x[index,1], -np.sin(x[index,4]), np.cos(x[index,4]), color='blue', width=0.003, scale=15, headwidth=1, headlength=0)
    plt.quiver(x[index,0], x[index,1], Fx[index], Fy[index], color='red', width=0.003, scale=100, headwidth=1, headlength=0)
    plt.quiver(x[N,0], x[N,1], -np.sin(x[N,4]), np.cos(x[N,4]), color='blue', width=0.003, scale=15, headwidth=1, headlength=0)
    plt.axis([-5, 5, -1, 7])
    plt.xlabel('X ()', fontsize = fS)
    plt.ylabel('Y ()', fontsize = fS)

    plt.figure(2,figsize=(10,15))
    plt.subplot(321)
    plt.plot(np.array(range(N+1))*delT, x[:,0], linewidth=2.0,label='naive')
    plt.xlabel('time (s)', fontsize = fS)
    plt.ylabel('rx ()', fontsize = fS)
    plt.subplot(322)
    plt.plot(np.array(range(N+1))*delT, x[:,1], linewidth=2.0,label='naive')
    plt.xlabel('time (s)', fontsize = fS)
    plt.ylabel('ry ()', fontsize = fS)
    plt.subplot(323)
    plt.plot(np.array(range(N+1))*delT, x[:,2], linewidth=2.0,label='naive')
    plt.xlabel('time (s)', fontsize = fS)
    plt.ylabel('vx ()', fontsize = fS)
    plt.subplot(324)
    plt.plot(np.array(range(N+1))*delT, x[:,3], linewidth=2.0,label='naive')
    plt.xlabel('time (s)', fontsize = fS)
    plt.ylabel('vy ()', fontsize = fS)
    plt.legend(fontsize=fS)
    plt.subplot(325)
    plt.plot(np.array(range(N+1))*delT, x[:,4]*180/np.pi, linewidth=2.0,label='naive')
    plt.xlabel('time (s)', fontsize = fS)
    plt.ylabel('theta (degree)', fontsize = fS)
    plt.subplot(326)
    plt.plot(np.array(range(N+1))*delT, x[:,5]*180/np.pi, linewidth=2.0,label='naive')
    plt.xlabel('time (s)', fontsize = fS)
    plt.ylabel('theta dot (rad/s)', fontsize = fS)
    plt.legend(fontsize=fS)
    plt.show()
    
    plt.figure(3)
    plt.subplot(121)
    plt.plot(np.array(range(N))*delT, u[:N,0]*180/np.pi, linewidth=2.0)
    plt.xlabel('time (s)', fontsize = fS)
    plt.ylabel('gimbal (degree)', fontsize = fS)
    plt.subplot(122)
    plt.plot(np.array(range(N))*delT, u[:N,1], linewidth=2.0)
    plt.xlabel('time (s)', fontsize = fS)
    plt.ylabel('thrust ()', fontsize = fS)
    plt.show()

def make_quadrotor_trajectory_fig(x,obs,c,H,r,img_name='quadrotor') :
    filenames = []
    N = np.shape(x)[0]
    for k in range(N):
        xp = x[k]
        lp = obs['point'][k]
        plt.figure(figsize=(5,8))
        fS = 18
        ax=plt.gca()
        for ce,He,re in zip(c,H,r) :
            circle1 = plt.Circle((ce[0],ce[1]),re,color='tab:red',alpha=0.5,fill=True)
            ax.add_patch(circle1)
        for ro in lp :
            plt.plot([xp[0],ro[0]],[xp[1],ro[1]],color='tab:blue')
            plt.plot(ro[0],ro[1],'o',color='tab:blue')
        plt.plot(xp[0],xp[1],'o',color='black')
        plt.axis([-2.5, 2.5, -1, 7])
        plt.gca().set_aspect('equal', adjustable='box')
        
        filename = '../images/{:d}.png'.format(k)
        plt.savefig(filename)
        filenames.append(filename)
        plt.close()

    with imageio.get_writer('../images/'+img_name+'.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    for filename in set(filenames):
        os.remove(filename)

def plot_traj_funnel(x,u,Q,xi=None,xf=None,Qi=None,Qf=None,plt=plt,flag_label=True,fS=15) :
    radius_list,angle_list = get_radius_angle(Q)

    # plt.figure(idx_plot,figsize=(7,7))
    plt.plot(x[:,0], x[:,1],'--',color='tab:orange',alpha=0.8,linewidth=2.0)
    ax=plt.gca()
    if Qi is not None :
        radius_f,angle_f = get_radius_angle([Qi])
        for radius,angle in zip(radius_f,angle_f) :
            ell = Ellipse((xi[0],xi[1]),radius[0]*2,radius[1]*2,angle=np.rad2deg(angle),
            color='tab:green',alpha=0.5,fill=True)
            ax.add_patch(ell)
    if Qf is not None :
        radius_f,angle_f = get_radius_angle([Qf])
        for radius,angle in zip(radius_f,angle_f) :
            ell = Ellipse((xf[0],xf[1]),radius[0]*2,radius[1]*2,angle=np.rad2deg(angle),
            color='tab:green',alpha=0.5,fill=True)
            ax.add_patch(ell)
    for x_,radius,angle in zip(x,radius_list,angle_list) :
        ell = Ellipse((x_[0],x_[1]),radius[0]*2,radius[1]*2,angle=np.rad2deg(angle),color='tab:blue',alpha=0.5,fill=True)
        ax.add_patch(ell)
    # if xf is not None :
    #     plt.plot(xf[0],xf[1],"o",label='goal')
    if flag_label == True :
        plt.plot(1e3,1e3,'--',color='tab:orange',label="nominal")
        plt.plot(1e3,1e3,'o',markersize=15,color='tab:blue',label="funnel") 
        plt.plot(1e3,1e3,'o',markersize=15,color='tab:green',label="initial and final ellipsoid") 
        # plt.plot(1e3,1e3,'o',markersize=15,color='tab:green',label="final") 
        plt.plot(1e3,1e3,'o',markersize=15,alpha=0.5,color='tab:red',label="obstacles") 

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis([-1.0, 6.0, -1.0, 6.0])
    plt.xlabel('$r_x$ (m)', fontsize = fS)
    plt.ylabel('$r_y$ (m)', fontsize = fS)
    if flag_label == True :
        plt.legend(fontsize=fS)
    ticks_font = "Times New Roman"
    for label in ax.get_xticklabels():
        label.set_fontproperties(ticks_font)

    for label in ax.get_yticklabels():
        label.set_fontproperties(ticks_font)

def plot_funnel(x,Q,ax) :
    radius_list,angle_list = get_radius_angle(Q)
    # ax=plt.gca()
    N = len(x)
    for idx,(x_,radius,angle) in enumerate(zip(x,radius_list,angle_list)) :
        # if idx < N -1 :
        ell = Ellipse((x_[0],x_[1]),radius[0]*2,radius[1]*2,angle=np.rad2deg(angle),color='tab:blue',alpha=0.5,fill=True)
        # else :
        #     ell = Ellipse((x_[0],x_[1]),radius[0]*2,radius[1]*2,angle=np.rad2deg(angle),color='tab:red',alpha=0.5,fill=True)

        ax.add_patch(ell)
    ax.plot(1e3,1e3,'o',markersize=15,color='tab:blue',label="funnel") 
    # ticks_font = "Times New Roman"
    # for label in ax.get_xticklabels():
    #     label.set_fontproperties(ticks_font)

    # for label in ax.get_yticklabels():
    #     label.set_fontproperties(ticks_font)