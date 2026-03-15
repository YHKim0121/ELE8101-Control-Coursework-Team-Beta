"""
STEP 2 BRAINSTORMING

Goal:
Extend the Step 1 Kalman Filter to track a vehicle moving along a curved
trajectory (e.g., a train on rails).

Key idea:
Instead of straight-line motion, the vehicle moves along a circular path.

Possible state representation:
x = [theta, v]

theta = angular position along the circular track
v     = speed along the track

Motion model:
theta_{t+1} = theta_t + (h / R) * v_t
v_{t+1} = v_t + noise

Measurement model:
Sensors measure distance to fixed beacons.

distance_i = sqrt((x - bx_i)^2 + (y - by_i)^2)

where

x = R cos(theta)
y = R sin(theta)

Because this measurement is nonlinear,
we must use an Extended Kalman Filter (EKF).

Tasks:
1. Modify motion model to circular motion
2. Convert linear KF to EKF
3. Add beacon range sensors
4. Plot circular trajectory
"""