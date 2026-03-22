import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

np.random.seed(0)

# ============================================================
# Step 3 state definition (only geometry changed at this stage)
# x = [p, v, e, ve]
# ============================================================

# Problem parameters
h = 0.01
Q = 0.01
R = 1.5**2

# Step 2 dynamics (UNCHANGED for now)
A = np.array([[1, h], [0, 1]])
G = np.array([[0], [1]])

# Road centre radius
R_c = 48.0

# Beacon positions
beacons = np.array([
    [0, -100],
    [100, 100],
    [-100, 100]
])
ny = len(beacons)
R_mat = R * np.eye(ny)

# Initial state (STILL Step 2 for now)
x0_tilde = np.array([[0], [10]])
P0 = np.array([[1, 0], [0, 1]])

# ============================================================
# 🔥 KEY CHANGE: symbolic state is now 4D
# ============================================================

nx = 4
x_sym = cs.SX.sym('x', nx)
v_sym = cs.SX.sym('v', ny)

# Extract variables
p_sym = x_sym[0]
e_sym = x_sym[2]

# ============================================================
# 🔥 KEY CHANGE: variable radius using lateral offset e
# ============================================================

theta = p_sym / R_c
radius = R_c + e_sym

pos_x = radius * cs.cos(theta)
pos_y = radius * cs.sin(theta)

# Measurement model (distance to beacons)
h_list = []
for i in range(ny):
    bx = beacons[i, 0]
    by = beacons[i, 1]

    dist = cs.sqrt((pos_x - bx)**2 + (pos_y - by)**2)
    h_list.append(dist + v_sym[i])

h_sym = cs.vertcat(*h_list)

# Jacobian for EKF
jhx = cs.Function('jhx', [x_sym, v_sym], [cs.jacobian(h_sym, x_sym)])
h_func = cs.Function('h_func', [x_sym, v_sym], [h_sym])