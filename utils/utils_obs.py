import imageio
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import math
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
#     print ("Values are: \n%s" % (x))
def get_H_obs(r) :
    return np.diag([1/r,1/r,0])
def get_H_safe(r,r_safe) :
    return np.diag([1/(r+r_safe),1/(r+r_safe),0])
def generate_obstacle(point_range,dice=-1,r_safe=0.3) :
#     dice = np.random.randint(low=1, high=4)
    c_list = []
    H_obs_list = []
    H_safe_list = []
    r_list = []
    if dice != 1 :
        c = np.array([1+np.random.uniform(-point_range,point_range),4+np.random.uniform(0,1),0])
        r = np.random.uniform(0.2,0.5)
        c_list.append(c)
        H_obs_list.append(get_H_obs(r))
        H_safe_list.append(get_H_safe(r,r_safe))
        r_list.append(r)
    if dice != 2 :
        c = np.array([1+np.random.uniform(-point_range,point_range),2+np.random.uniform(-1,0),0])
        r = np.random.uniform(0.2,0.5)
        c_list.append(c)
        H_obs_list.append(get_H_obs(r))
        H_safe_list.append(get_H_safe(r,r_safe))
        r_list.append(r)
    if dice != 3 :
        c = np.array([-1+np.random.uniform(-point_range,point_range),4+np.random.uniform(0,1),0])
        r = np.random.uniform(0.2,0.5)
        c_list.append(c)
        H_obs_list.append(get_H_obs(r))
        H_safe_list.append(get_H_safe(r,r_safe))
        r_list.append(r)
    if dice != 4 :
        c = np.array([-1+np.random.uniform(-point_range,point_range),2+np.random.uniform(-1,0),0])
        r = np.random.uniform(0.2,0.5)
        c_list.append(c)
        H_obs_list.append(get_H_obs(r))
        H_safe_list.append(get_H_safe(r,r_safe))
        r_list.append(r)
    assert len(c_list) == len(H_obs_list)
    assert len(c_list) == len(H_safe_list)
    num_obstacle = len(c)
    return c_list,H_obs_list,H_safe_list,r_list

def generate_obstacle_circle(r_safe=0.3) :
#     dice = np.random.randint(low=1, high=4)
    c_list = []
    H_obs_list = []
    H_safe_list = []
    r_list = []
    min_theta = np.arctan(0.8)
    angle = np.random.uniform(0,2*np.pi)
    angle += np.random.uniform(min_theta,2/3*np.pi-min_theta)
    length = np.random.uniform(1,2)
    c = np.array([length*np.cos(angle),length*np.sin(angle)+3,0])
    r = np.random.uniform(0.2,0.5)
    c_list.append(c)
    H_obs_list.append(get_H_obs(r))
    H_safe_list.append(get_H_safe(r,r_safe))
    r_list.append(r)

    for i in range(2) :
        angle += min_theta
        angle += np.random.uniform(min_theta,2/3*np.pi-min_theta)
        length = np.random.uniform(1,2)
        c = np.array([length*np.cos(angle),length*np.sin(angle)+3,0])
        r = np.random.uniform(0.2,0.5)
        c_list.append(c)
        H_obs_list.append(get_H_obs(r))
        H_safe_list.append(get_H_safe(r,r_safe))
        r_list.append(r)
    
    assert len(c_list) == len(H_obs_list)
    assert len(c_list) == len(H_safe_list)
    num_obstacle = len(c)
    return c_list,H_obs_list,H_safe_list,r_list

def generate_obstacle_massive(r_safe=0.3) :
    c_list = []
    H_obs_list = []
    H_safe_list = []
    r_list = []

    y_start = 1.75
    for j in range(3) :
        x_start = np.random.uniform(-2,-2+0.5)
        for i in range(3) :
            r = np.random.uniform(0.2,0.3)
            c = np.array([x_start,y_start+np.random.uniform(-0.1,0.1),0])
            c_list.append(c)
            H_obs_list.append(get_H_obs(r))
            H_safe_list.append(get_H_safe(r,r_safe))
            r_list.append(r)
            x_start += np.random.uniform(r+0.1+0.3+0.6,r+0.1+0.3+1.1)
        y_start += np.random.uniform(r+0.3+0.3+0.6,r+0.3+0.3+1.1)
    assert len(c_list) == len(H_obs_list)
    assert len(c_list) == len(H_safe_list)
    num_obstacle = len(c)
    return c_list,H_obs_list,H_safe_list,r_list
def generate_obstacle_random(r_safe=0.3,num_obs=None) :
    def euclidean_distance(x1, y1, x2, y2):
        return math.hypot((x1 - x2), (y1 - y2))
    run = True
    circle_list = []
    max_iter = 5000
    if num_obs is None :
        num_obstacle = np.random.randint(4,8)
        # num_obstacle = np.random.randint(4,6)
    else : 
        num_obstacle = num_obs
    for i in range(max_iter) :
        if i == max_iter -1 :
            print("reach to the max iter")
        if len(circle_list) == num_obstacle :
            break
        # r = np.random.uniform(0.5,0.6)
        # r = np.random.uniform(0.4,0.6)
        r = np.random.uniform(0.5,1.0)
        x = np.random.uniform(-1.0,1.0)
        y = np.random.uniform(0.7,5.3)
        if not any((x2, y2, r2) for x2, y2, r2 in circle_list if euclidean_distance(x, y, x2, y2) < r + r2):
            circle_list.append((x, y, r))  
    c_list = []
    H_obs_list = []
    H_safe_list = []
    r_list = []
    for circle in circle_list :
            x,y,r = circle[0],circle[1],circle[2]
            r -= r_safe
            c = np.array([x,y,0])
            c_list.append(c)
            H_obs_list.append(get_H_obs(r))
            H_safe_list.append(get_H_safe(r,r_safe))
            r_list.append(r)
    assert len(c_list) == len(H_obs_list)
    assert len(c_list) == len(H_safe_list)
    num_obstacle = len(c)
    return c_list,H_obs_list,H_safe_list,r_list