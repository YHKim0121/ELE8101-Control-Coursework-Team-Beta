import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

np.random.seed(0)

# ============================================================
# Step 3 FULL version (before soft boundary)
# Includes:
# - 4-state EKF
# - full storage (state + covariance)
# - lateral plots
# - 2D trajectory
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

# ✅ FULL STORAGE (important commit)
x_true_hist = np.zeros((n_sim, nx))
x_est_hist = np.zeros((n_sim - 1, nx))
P_hist = np.zeros((n_sim - 1, nx, nx))

x_true_hist[0, :] = x_true.flatten()

# ============================================================
# Main loop
# ============================================================

for t in range(n_sim - 1):

    noise = np.random.normal(0, np.sqrt(R_meas), ny)
    y = np.array(h_func(x_true, noise)).astype(float).flatten()

    P_upd, x_upd = measurement_update(P_pred, x_pred, y)

    x_est_hist[t, :] = x_upd.flatten()
    P_hist[t] = P_upd

    P_pred, x_pred = time_update(P_upd, x_upd)

    w = np.array([
        [np.random.normal(0, np.sqrt(Q_long))],
        [np.random.normal(0, np.sqrt(Q_lat))]
    ])

    x_true = A @ x_true + b + G @ w
    x_true_hist[t + 1, :] = x_true.flatten()

# ============================================================
# Uncertainty
# ============================================================

sigma_p  = np.sqrt(P_hist[:, 0, 0])
sigma_v  = np.sqrt(P_hist[:, 1, 1])
sigma_e  = np.sqrt(P_hist[:, 2, 2])
sigma_ve = np.sqrt(P_hist[:, 3, 3])

t_axis = np.arange(n_sim) * h

# ============================================================
# Plot 1: Full states
# ============================================================

fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharex=True)

labels = ['p', 'v', 'e', 've']
sigmas = [sigma_p, sigma_v, sigma_e, sigma_ve]

for i in range(4):
    axs[i].plot(t_axis, x_true_hist[:, i], 'b-', label=f'True {labels[i]}')
    axs[i].plot(t_axis[:-1], x_est_hist[:, i], 'r-', label=f'Est {labels[i]}')
    axs[i].fill_between(
        t_axis[:-1],
        x_est_hist[:, i] - 3 * sigmas[i],
        x_est_hist[:, i] + 3 * sigmas[i],
        alpha=0.2
    )
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel('Time [s]')
plt.tight_layout()

# ============================================================
# Plot 2: Lateral motion (IMPORTANT for Step 3)
# ============================================================

plt.figure(figsize=(10, 4))
plt.plot(t_axis, x_true_hist[:, 2], label='True e')
plt.plot(t_axis[:-1], x_est_hist[:, 2], '--', label='Estimated e')
plt.axhline(2.0, linestyle='--', color='k', label='Lane bounds')
plt.axhline(-2.0, linestyle='--', color='k')
plt.xlabel('Time [s]')
plt.ylabel('Lateral offset [m]')
plt.title('Step 3 Lateral Motion')
plt.grid(True)
plt.legend()
plt.tight_layout()

# ============================================================
# Plot 3: 2D trajectory (IMPORTANT)
# ============================================================

fig2, ax2 = plt.subplots(figsize=(8, 8))

theta_true = x_true_hist[:, 0] / R_center
radius_true = R_center + x_true_hist[:, 2]

x_true_2d = radius_true * np.cos(theta_true)
y_true_2d = radius_true * np.sin(theta_true)

theta_est = x_est_hist[:, 0] / R_center
radius_est = R_center + x_est_hist[:, 2]

x_est_2d = radius_est * np.cos(theta_est)
y_est_2d = radius_est * np.sin(theta_est)

ax2.plot(x_true_2d, y_true_2d, label='True trajectory')
ax2.plot(x_est_2d, y_est_2d, '--', label='Estimated trajectory')

ax2.scatter(beacons[:, 0], beacons[:, 1], c='k', marker='^', label='Beacons')

circle_inner = plt.Circle((0, 0), R_center - 2.0, fill=False, linestyle=':')
circle_outer = plt.Circle((0, 0), R_center + 2.0, fill=False, linestyle=':')
circle_center = plt.Circle((0, 0), R_center, fill=False, linestyle='--')

ax2.add_patch(circle_inner)
ax2.add_patch(circle_outer)
ax2.add_patch(circle_center)

ax2.set_aspect('equal')
ax2.set_xlabel('X [m]')
ax2.set_ylabel('Y [m]')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()