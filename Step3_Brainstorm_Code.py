import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

np.random.seed(0)

# ============================================================
# Step 3 basic 4-state model
# x = [p, v, e, ve]
# p  = longitudinal position
# v  = longitudinal speed
# e  = lateral offset
# ve = lateral velocity
# ============================================================

# Problem parameters
h = 0.01
R = 1.5**2

# Process noise
Q_long = 0.01
Q_lat = 0.05
Qw = np.diag([Q_long, Q_lat])

# Curved road radius
R_c = 48.0

# Basic 4-state dynamics
A = np.array([
    [1.0, h,   0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, h],
    [0.0, 0.0, 0.0, 1.0]
])

# Noise enters v and ve
G = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 0.0],
    [0.0, 1.0]
])

# Beacon positions
beacons = np.array([
    [0.0, -100.0],
    [100.0, 100.0],
    [-100.0, 100.0]
])

ny = len(beacons)
R_mat = R * np.eye(ny)

# Initial estimate
x0_tilde = np.array([
    [0.0],   # p
    [10.0],  # v
    [0.0],   # e
    [0.1]    # ve
])

P0 = np.diag([1.0, 1.0, 0.5, 0.1])

# ============================================================
# Nonlinear measurement model
# ============================================================
nx = 4
x_sym = cs.SX.sym('x', nx)
v_sym = cs.SX.sym('v', ny)

p_sym = x_sym[0]
e_sym = x_sym[2]

theta = p_sym / R_c
radius = R_c + e_sym

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
    C_mat = np.array(jhx(x_pred, np.zeros(ny))).astype(float)
    y_pred = np.array(h_func(x_pred, np.zeros(ny))).astype(float).flatten()

    S = C_mat @ P_pred @ C_mat.T + R_mat
    K = P_pred @ C_mat.T @ np.linalg.inv(S)

    innovation = y - y_pred
    x_upd = x_pred + K @ innovation.reshape(-1, 1)
    P_upd = P_pred - P_pred @ C_mat.T @ np.linalg.solve(S, C_mat @ P_pred)

    return P_upd, x_upd

def time_update(P_upd, x_upd):
    x_pred = A @ x_upd
    P_pred = A @ P_upd @ A.T + G @ Qw @ G.T
    return P_pred, x_pred

# ============================================================
# Simulation setup
# ============================================================
n_sim = 5000

x_true = np.random.multivariate_normal(
    x0_tilde.flatten(), P0
).reshape((-1, 1))

P_pred = P0.copy()
x_pred = x0_tilde.copy()

x_true_hist = np.zeros((n_sim, 4))
x_est_hist = np.zeros((n_sim - 1, 4))
P_est_hist = np.zeros((n_sim - 1, 4, 4))

x_true_hist[0, :] = x_true.flatten()

# ============================================================
# EKF main loop
# ============================================================
for t in range(n_sim - 1):
    meas_noise = np.random.normal(0, np.sqrt(R), ny)
    y = np.array(h_func(x_true, meas_noise)).astype(float).flatten()

    P_upd, x_upd = measurement_update(P_pred, x_pred, y)

    x_est_hist[t, :] = x_upd.flatten()
    P_est_hist[t] = P_upd

    P_pred, x_pred = time_update(P_upd, x_upd)

    w = np.array([
        [np.random.normal(0, np.sqrt(Q_long))],
        [np.random.normal(0, np.sqrt(Q_lat))]
    ])

    x_true = A @ x_true + G @ w
    x_true_hist[t + 1, :] = x_true.flatten()

# ============================================================
# Uncertainty
# ============================================================
t_axis = np.arange(n_sim) * h

sigma_p = np.sqrt(P_est_hist[:, 0, 0])
sigma_v = np.sqrt(P_est_hist[:, 1, 1])
sigma_e = np.sqrt(P_est_hist[:, 2, 2])
sigma_ve = np.sqrt(P_est_hist[:, 3, 3])

# ============================================================
# Plot 1: state histories
# ============================================================
fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

axs[0].plot(t_axis, x_true_hist[:, 0], label='Actual p')
axs[0].plot(t_axis[:-1], x_est_hist[:, 0], label='Estimated p')
axs[0].fill_between(
    t_axis[:-1],
    x_est_hist[:, 0] - 3 * sigma_p,
    x_est_hist[:, 0] + 3 * sigma_p,
    alpha=0.2
)
axs[0].set_ylabel('p [m]')
axs[0].grid(True, alpha=0.3)
axs[0].legend()

axs[1].plot(t_axis, x_true_hist[:, 1], label='Actual v')
axs[1].plot(t_axis[:-1], x_est_hist[:, 1], label='Estimated v')
axs[1].fill_between(
    t_axis[:-1],
    x_est_hist[:, 1] - 3 * sigma_v,
    x_est_hist[:, 1] + 3 * sigma_v,
    alpha=0.2
)
axs[1].set_ylabel('v [m/s]')
axs[1].grid(True, alpha=0.3)
axs[1].legend()

axs[2].plot(t_axis, x_true_hist[:, 2], label='Actual e')
axs[2].plot(t_axis[:-1], x_est_hist[:, 2], label='Estimated e')
axs[2].fill_between(
    t_axis[:-1],
    x_est_hist[:, 2] - 3 * sigma_e,
    x_est_hist[:, 2] + 3 * sigma_e,
    alpha=0.2
)
axs[2].set_ylabel('e [m]')
axs[2].grid(True, alpha=0.3)
axs[2].legend()

axs[3].plot(t_axis, x_true_hist[:, 3], label='Actual ve')
axs[3].plot(t_axis[:-1], x_est_hist[:, 3], label='Estimated ve')
axs[3].fill_between(
    t_axis[:-1],
    x_est_hist[:, 3] - 3 * sigma_ve,
    x_est_hist[:, 3] + 3 * sigma_ve,
    alpha=0.2
)
axs[3].set_ylabel('ve [m/s]')
axs[3].set_xlabel('Time [s]')
axs[3].grid(True, alpha=0.3)
axs[3].legend()

plt.tight_layout()

# ============================================================
# Plot 2: 2D trajectory using variable radius
# ============================================================
fig2, ax2 = plt.subplots(figsize=(8, 8))

theta_true = x_true_hist[:, 0] / R_c
radius_true = R_c + x_true_hist[:, 2]
x_2d_true = radius_true * np.cos(theta_true)
y_2d_true = radius_true * np.sin(theta_true)

theta_est = x_est_hist[:, 0] / R_c
radius_est = R_c + x_est_hist[:, 2]
x_2d_est = radius_est * np.cos(theta_est)
y_2d_est = radius_est * np.sin(theta_est)

ax2.plot(x_2d_true, y_2d_true, 'b-', label='Actual trajectory')
ax2.plot(x_2d_est, y_2d_est, 'r--', label='Estimated trajectory')
ax2.scatter(beacons[:, 0], beacons[:, 1], c='k', marker='^', s=100, label='Beacons')

circle_inner = plt.Circle((0, 0), 46, color='gray', fill=False, linestyle=':')
circle_outer = plt.Circle((0, 0), 50, color='gray', fill=False, linestyle=':')
circle_center = plt.Circle((0, 0), 48, color='gray', fill=False, linestyle='--', alpha=0.6)

ax2.add_patch(circle_inner)
ax2.add_patch(circle_outer)
ax2.add_patch(circle_center)

ax2.set_aspect('equal')
ax2.set_xlabel('X [m]')
ax2.set_ylabel('Y [m]')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()