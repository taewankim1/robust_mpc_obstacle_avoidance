# Robust Model predictive control via sequential convex programming for obstacle avoidance
 - The current version is using Gurobi solver. If you don't have a license, change it to ECOS in PTR_in_MPC.py and PTR_tf_free.py

## Static obstacle avoidance for unicycle model
<img src="images/MPC_static_obstacle.gif" width="400" height="400">

## Moving obstacle avoidance for unicycle model
<img src="images/MPC_moving_obstacle_reactive.gif" width="400" height="400">

## Moving obstacle avoidance for unicycle model with prediction
<img src="images/MPC_moving_obstacle_proactive.gif" width="400" height="400">

## Moving obstacle avoidance for unicycle model with considering forward reachable sets of the obstacle 
<img src="images/MPC_moving_obstacle_Ellipse_extreme.gif" width="400" height="400">

## Moving obstacle avoidance with funnel
<img src="images/MPC_with_funnel_Q_K" width="400" height="400">

## Moving obstacle avoidance with funnel with LQR Kgain
<img src="images/MPC_with_funnel_Q_LQR_K" width="400" height="400">

## Moving obstacle avoidance with funnel with LQR Kgain and fixed Q
<img src="images/MPC_with_funnel_LQR_K_fix_Q" width="400" height="400">

## References
* Successive Convexification for 6-DoF Mars
Rocket Powered Landing with Free-Final-Time (https://arxiv.org/pdf/1802.03827.pdf)
* Joint synthesis (https://arxiv.org/abs/2209.03535)

