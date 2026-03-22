import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

np.random.seed(0)

# ============================================================
# Step 3: Full 4-state simulation loop
# x = [p, v, e, ve]
# ============================================================

h = 0.01
R_meas = 1.5**2

# Process noise
Q_long = 0.01
Q_lat = 0.05
Qw = np.diag([Q_long, Q_lat])

# Road
R_center = 48.0

# Dynamics tuning
v_ref = 10.0
k_v = 0.05

a_lat = 0.995
k_lat = 0.02

# ============================================================
# System matrices
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
# Initial conditions
# ============================================================

x0_tilde = np.array([
    [0.0],
    [10.0],
    [0.5],
    [0.1]
])

P0 = np.diag([1.0, 1.0, 0.5, 0.1])

# ============================================================
# Measurement model
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

# Storage (IMPORTANT CHANGE)
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
# RMSE
# ============================================================

p_rmse  = np.sqrt(np.mean((x_true_hist[1:, 0] - x_est_hist[:, 0])**2))
v_rmse  = np.sqrt(np.mean((x_true_hist[1:, 1] - x_est_hist[:, 1])**2))
e_rmse  = np.sqrt(np.mean((x_true_hist[1:, 2] - x_est_hist[:, 2])**2))
ve_rmse = np.sqrt(np.mean((x_true_hist[1:, 3] - x_est_hist[:, 3])**2))

print("RMSE:")
print(f"p  = {p_rmse:.3f}")
print(f"v  = {v_rmse:.3f}")
print(f"e  = {e_rmse:.3f}")
print(f"ve = {ve_rmse:.3f}")

# ============================================================
# Plots
# ============================================================

t_axis = np.arange(n_sim) * h

fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

labels = ['p', 'v', 'e', 've']

for i in range(4):
    axs[i].plot(t_axis, x_true_hist[:, i], label=f'True {labels[i]}')
    axs[i].plot(t_axis[:-1], x_est_hist[:, i], '--', label=f'Est {labels[i]}')
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel('Time [s]')
plt.tight_layout()

plt.show()