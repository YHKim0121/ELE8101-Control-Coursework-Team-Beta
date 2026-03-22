import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

np.random.seed(0)

# ============================================================
# Step 3: Add proper longitudinal + lateral dynamics
# x = [p, v, e, ve]
# ============================================================

# Time step
h = 0.01

# Noise
Q_long = 0.01
Q_lat = 0.05
Qw = np.diag([Q_long, Q_lat])

R_meas = 1.5**2

# Road
R_center = 48.0

# Speed control
v_ref = 10.0
k_v = 0.05

# Lateral dynamics
a_lat = 0.995
k_lat = 0.02

# ============================================================
# State model (IMPORTANT CHANGE)
# ============================================================

A = np.array([
    [1.0, h,         0.0, 0.0],
    [0.0, 1.0 - k_v, 0.0, 0.0],
    [0.0, 0.0,       1.0, h],
    [0.0, 0.0,   -k_lat, a_lat]
])

b = np.array([
    [0.0],
    [k_v * v_ref],
    [0.0],
    [0.0]
])

# Noise enters v and ve
G = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 0.0],
    [0.0, 1.0]
])

# ============================================================
# Beacons
# ============================================================

beacons = np.array([
    [0.0, -100.0],
    [100.0, 100.0],
    [-100.0, 100.0]
])

ny = len(beacons)
R_mat = R_meas * np.eye(ny)

# ============================================================
# Initial state
# ============================================================

x0_tilde = np.array([
    [0.0],
    [10.0],
    [0.5],
    [0.1]
])

P0 = np.diag([1.0, 1.0, 0.5, 0.1])

# ============================================================
# Measurement model (same as before)
# ============================================================

nx = 4
x_sym = cs.SX.sym('x', nx)
v_sym = cs.SX.sym('v', ny)

p_sym = x_sym[0]
e_sym = x_sym[2]

theta = p_sym / R_center
radius = R_center + e_sym

pos_x = radius * cs.cos(theta)
pos_y = radius * cs.sin(theta)

h_list = []
for i in range(ny):
    bx = beacons[i, 0]
    by = beacons[i, 1]
    dist = cs.sqrt((pos_x - bx)**2 + (pos_y - by)**2)
    h_list.append(dist + v_sym[i])

h_sym = cs.vertcat(*h_list)

jhx = cs.Function('jhx', [x_sym, v_sym], [cs.jacobian(h_sym, x_sym)])
h_func = cs.Function('h_func', [x_sym, v_sym], [h_sym])

# ============================================================
# EKF functions
# ============================================================

def measurement_update(P_pred, x_pred, y):
    C = np.array(jhx(x_pred, np.zeros(ny))).astype(float)
    y_pred = np.array(h_func(x_pred, np.zeros(ny))).astype(float).flatten()

    S = C @ P_pred @ C.T + R_mat
    K = P_pred @ C.T @ np.linalg.inv(S)

    x_upd = x_pred + K @ (y - y_pred).reshape(-1, 1)
    P_upd = (np.eye(nx) - K @ C) @ P_pred

    return P_upd, x_upd

def time_update(P_upd, x_upd):
    x_pred = A @ x_upd + b
    P_pred = A @ P_upd @ A.T + G @ Qw @ G.T
    return P_pred, x_pred

# ============================================================
# Simulation
# ============================================================

n_sim = 3000

x_true = np.random.multivariate_normal(
    x0_tilde.flatten(), P0
).reshape((-1, 1))

x_pred = x0_tilde.copy()
P_pred = P0.copy()

x_true_hist = np.zeros((n_sim, nx))
x_est_hist = np.zeros((n_sim - 1, nx))
P_hist = np.zeros((n_sim - 1, nx, nx))

x_true_hist[0, :] = x_true.flatten()

# ============================================================
# Main loop
# ============================================================

for t in range(n_sim - 1):
    # Measurement
    noise = np.random.normal(0, np.sqrt(R_meas), ny)
    y = np.array(h_func(x_true, noise)).astype(float).flatten()

    # EKF update
    P_upd, x_upd = measurement_update(P_pred, x_pred, y)

    x_est_hist[t, :] = x_upd.flatten()
    P_hist[t] = P_upd

    # Prediction
    P_pred, x_pred = time_update(P_upd, x_upd)

    # True system
    w = np.array([
        [np.random.normal(0, np.sqrt(Q_long))],
        [np.random.normal(0, np.sqrt(Q_lat))]
    ])

    x_true = A @ x_true + b + G @ w
    x_true_hist[t + 1, :] = x_true.flatten()

# ============================================================
# Plot lateral motion (IMPORTANT for Step 3)
# ============================================================

t_axis = np.arange(n_sim) * h

plt.figure(figsize=(10, 4))
plt.plot(t_axis, x_true_hist[:, 2], label='True e')
plt.plot(t_axis[:-1], x_est_hist[:, 2], '--', label='Estimated e')
plt.xlabel('Time [s]')
plt.ylabel('Lateral offset e [m]')
plt.title('Step 3: Lateral motion introduced')
plt.grid(True)
plt.legend()

plt.show()