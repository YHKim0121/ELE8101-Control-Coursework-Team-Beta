import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

np.random.seed(0)

# ============================================================
# FINAL Step 3 EKF Model
# x = [p, v, e, ve]
# p  = longitudinal position along centreline
# v  = longitudinal speed
# e  = lateral offset from centreline
# ve = lateral velocity
# ============================================================

# ============================================================
# Problem parameters
# ============================================================
h = 0.01
R_meas = 1.5**2

# Process noise
Q_long = 0.01
Q_lat = 0.05
Qw = np.diag([Q_long, Q_lat])

# Road geometry
R_center = 48.0
lane_half_width = 2.0

# Speed regulation
v_ref = 10.0
k_v = 0.05

# Lateral dynamics
a_lat = 0.995
k_lat = 0.02

# Physical limits
v_min = 0.0
v_max = 25.0
ve_max = 0.55

# Soft boundary parameters
soft_zone = 1.6
k_soft = 0.20
k_outside = 0.60
edge_damping = 0.75

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
# Initial state and covariance
# ============================================================
x0_tilde = np.array([
    [0.0],    # p
    [10.0],   # v
    [0.5],    # e
    [0.1]     # ve
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

theta = p_sym / R_center
radius = R_center + e_sym

pos_x = radius * cs.cos(theta)
pos_y = radius * cs.sin(theta)

h_list = []
for i in range(ny):
    bx, by = beacons[i]
    dist = cs.sqrt((pos_x - bx)**2 + (pos_y - by)**2)
    h_list.append(dist + v_sym[i])

h_sym = cs.vertcat(*h_list)

jhx = cs.Function('jhx', [x_sym, v_sym], [cs.jacobian(h_sym, x_sym)])
h_func = cs.Function('h_func', [x_sym, v_sym], [h_sym])

# ============================================================
# Helper functions
# ============================================================
def enforce_constraints(x):
    """
    Soft lane-boundary handling:
    - clamp longitudinal speed and lateral velocity
    - gently push vehicle inward near lane edges
    - stronger correction if outside lane
    """
    x_new = x.copy()

    # Clamp physical speed limits
    x_new[1, 0] = np.clip(x_new[1, 0], v_min, v_max)
    x_new[3, 0] = np.clip(x_new[3, 0], -ve_max, ve_max)

    e_val = x_new[2, 0]
    ve_val = x_new[3, 0]

    # Soft zone correction
    if e_val > soft_zone:
        x_new[3, 0] -= k_soft * (e_val - soft_zone)
        if ve_val > 0:
            x_new[3, 0] *= edge_damping

    elif e_val < -soft_zone:
        x_new[3, 0] += k_soft * (-soft_zone - e_val)
        if ve_val < 0:
            x_new[3, 0] *= edge_damping

    # Stronger correction outside lane
    e_val = x_new[2, 0]
    ve_val = x_new[3, 0]

    if e_val > lane_half_width:
        x_new[3, 0] -= k_outside * (e_val - lane_half_width)
        if ve_val > 0:
            x_new[3, 0] *= 0.5

    elif e_val < -lane_half_width:
        x_new[3, 0] += k_outside * (-lane_half_width - e_val)
        if ve_val < 0:
            x_new[3, 0] *= 0.5

    # Final safety clip
    x_new[3, 0] = np.clip(x_new[3, 0], -ve_max, ve_max)

    return x_new


def state_to_xy(x_array):
    theta_vals = x_array[:, 0] / R_center
    radius_vals = R_center + x_array[:, 2]
    x_2d = radius_vals * np.cos(theta_vals)
    y_2d = radius_vals * np.sin(theta_vals)
    return x_2d, y_2d

# ============================================================
# EKF functions
# ============================================================
def measurement_update(P_pred, x_pred, y):
    C = np.array(jhx(x_pred, np.zeros(ny))).astype(float)
    y_pred = np.array(h_func(x_pred, np.zeros(ny))).astype(float).flatten()

    S = C @ P_pred @ C.T + R_mat
    K = P_pred @ C.T @ np.linalg.inv(S)

    innovation = y - y_pred
    x_upd = x_pred + K @ innovation.reshape(-1, 1)
    P_upd = (np.eye(nx) - K @ C) @ P_pred

    x_upd = enforce_constraints(x_upd)
    return P_upd, x_upd, innovation


def time_update(P_upd, x_upd):
    x_pred = A @ x_upd + b
    P_pred = A @ P_upd @ A.T + G @ Qw @ G.T

    x_pred = enforce_constraints(x_pred)
    return P_pred, x_pred

# ============================================================
# Simulation setup
# ============================================================
n_sim = 3000

x_true = np.random.multivariate_normal(
    x0_tilde.flatten(), P0
).reshape((-1, 1))
x_true = enforce_constraints(x_true)

x_pred = x0_tilde.copy()
P_pred = P0.copy()

x_true_hist = np.zeros((n_sim, nx))
x_est_hist = np.zeros((n_sim - 1, nx))
P_hist = np.zeros((n_sim - 1, nx, nx))
innov_hist = np.zeros((n_sim - 1, ny))

x_true_hist[0, :] = x_true.flatten()

# ============================================================
# Main EKF loop
# ============================================================
for t in range(n_sim - 1):
    # Measurement from true system
    meas_noise = np.random.normal(0, np.sqrt(R_meas), ny)
    y = np.array(h_func(x_true, meas_noise)).astype(float).flatten()

    # EKF measurement update
    P_upd, x_upd, innovation = measurement_update(P_pred, x_pred, y)

    x_est_hist[t, :] = x_upd.flatten()
    P_hist[t] = P_upd
    innov_hist[t, :] = innovation

    # EKF time update
    P_pred, x_pred = time_update(P_upd, x_upd)

    # True system propagation
    w = np.array([
        [np.random.normal(0, np.sqrt(Q_long))],
        [np.random.normal(0, np.sqrt(Q_lat))]
    ])

    x_true = A @ x_true + b + G @ w
    x_true = enforce_constraints(x_true)

    x_true_hist[t + 1, :] = x_true.flatten()

# ============================================================
# Uncertainty and RMSE
# ============================================================
t_axis = np.arange(n_sim) * h

sigma_p = np.sqrt(P_hist[:, 0, 0])
sigma_v = np.sqrt(P_hist[:, 1, 1])
sigma_e = np.sqrt(P_hist[:, 2, 2])
sigma_ve = np.sqrt(P_hist[:, 3, 3])

p_rmse = np.sqrt(np.mean((x_true_hist[1:, 0] - x_est_hist[:, 0])**2))
v_rmse = np.sqrt(np.mean((x_true_hist[1:, 1] - x_est_hist[:, 1])**2))
e_rmse = np.sqrt(np.mean((x_true_hist[1:, 2] - x_est_hist[:, 2])**2))
ve_rmse = np.sqrt(np.mean((x_true_hist[1:, 3] - x_est_hist[:, 3])**2))

print("RMSE results:")
print(f"p   RMSE = {p_rmse:.4f} m")
print(f"v   RMSE = {v_rmse:.4f} m/s")
print(f"e   RMSE = {e_rmse:.4f} m")
print(f"ve  RMSE = {ve_rmse:.4f} m/s")

# ============================================================
# Plot 1: Full state histories
# ============================================================
fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharex=True)

# p
axs[0].plot(t_axis, x_true_hist[:, 0], 'b-', label='Actual p')
axs[0].plot(t_axis[:-1], x_est_hist[:, 0], 'r-', label='Estimated p')
axs[0].fill_between(
    t_axis[:-1],
    x_est_hist[:, 0] - 3 * sigma_p,
    x_est_hist[:, 0] + 3 * sigma_p,
    color='red', alpha=0.2, label='±3σ'
)
axs[0].set_ylabel('p [m]')
axs[0].grid(True, alpha=0.3)
axs[0].legend()

# v
axs[1].plot(t_axis, x_true_hist[:, 1], 'b-', label='Actual v')
axs[1].plot(t_axis[:-1], x_est_hist[:, 1], 'r-', label='Estimated v')
axs[1].fill_between(
    t_axis[:-1],
    x_est_hist[:, 1] - 3 * sigma_v,
    x_est_hist[:, 1] + 3 * sigma_v,
    color='red', alpha=0.2, label='±3σ'
)
axs[1].axhline(v_ref, color='g', linestyle='--', label='Reference speed')
axs[1].axhline(v_max, color='k', linestyle='--', label='Speed bounds')
axs[1].axhline(v_min, color='k', linestyle='--')
axs[1].set_ylabel('v [m/s]')
axs[1].grid(True, alpha=0.3)
axs[1].legend()

# e
axs[2].plot(t_axis, x_true_hist[:, 2], 'b-', label='Actual e')
axs[2].plot(t_axis[:-1], x_est_hist[:, 2], 'r-', label='Estimated e')
axs[2].fill_between(
    t_axis[:-1],
    x_est_hist[:, 2] - 3 * sigma_e,
    x_est_hist[:, 2] + 3 * sigma_e,
    color='red', alpha=0.2, label='±3σ'
)
axs[2].axhline(lane_half_width, color='k', linestyle='--', label='Lane bounds')
axs[2].axhline(-lane_half_width, color='k', linestyle='--')
axs[2].axhline(soft_zone, color='gray', linestyle=':', label='Soft zone')
axs[2].axhline(-soft_zone, color='gray', linestyle=':')
axs[2].set_ylabel('e [m]')
axs[2].grid(True, alpha=0.3)
axs[2].legend()

# ve
axs[3].plot(t_axis, x_true_hist[:, 3], 'b-', label='Actual ve')
axs[3].plot(t_axis[:-1], x_est_hist[:, 3], 'r-', label='Estimated ve')
axs[3].fill_between(
    t_axis[:-1],
    x_est_hist[:, 3] - 3 * sigma_ve,
    x_est_hist[:, 3] + 3 * sigma_ve,
    color='red', alpha=0.2, label='±3σ'
)
axs[3].axhline(ve_max, color='k', linestyle='--', label='Lateral velocity bounds')
axs[3].axhline(-ve_max, color='k', linestyle='--')
axs[3].set_xlabel('Time [s]')
axs[3].set_ylabel('ve [m/s]')
axs[3].grid(True, alpha=0.3)
axs[3].legend()

plt.tight_layout()

# ============================================================
# Plot 2: Lateral motion only
# ============================================================
plt.figure(figsize=(10, 4))
plt.plot(t_axis, x_true_hist[:, 2], label='Actual lateral offset e')
plt.plot(t_axis[:-1], x_est_hist[:, 2], '--', label='Estimated lateral offset e')
plt.axhline(lane_half_width, color='k', linestyle='--', label='Lane bounds')
plt.axhline(-lane_half_width, color='k', linestyle='--')
plt.axhline(soft_zone, color='gray', linestyle=':', label='Soft zone')
plt.axhline(-soft_zone, color='gray', linestyle=':')
plt.xlabel('Time [s]')
plt.ylabel('Lateral offset [m]')
plt.title('Step 3 lateral motion with soft boundary')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# ============================================================
# Plot 3: 2D trajectory map
# ============================================================
fig2, ax2 = plt.subplots(figsize=(8, 8))

x_true_2d, y_true_2d = state_to_xy(x_true_hist)
x_est_2d, y_est_2d = state_to_xy(x_est_hist)

ax2.plot(x_true_2d, y_true_2d, color='blue', linewidth=2.5, label='Actual trajectory')
ax2.plot(x_est_2d, y_est_2d, color='red', linestyle='--', linewidth=1.5, label='Estimated trajectory')
ax2.scatter(beacons[:, 0], beacons[:, 1], c='k', marker='^', s=100, label='Beacons')

circle_inner = plt.Circle((0, 0), R_center - lane_half_width, color='gray', fill=False, linestyle=':')
circle_outer = plt.Circle((0, 0), R_center + lane_half_width, color='gray', fill=False, linestyle=':')
circle_center = plt.Circle((0, 0), R_center, color='gray', fill=False, linestyle='--', alpha=0.7)
circle_soft_in = plt.Circle((0, 0), R_center - soft_zone, color='lightgray', fill=False, linestyle='-.')
circle_soft_out = plt.Circle((0, 0), R_center + soft_zone, color='lightgray', fill=False, linestyle='-.')

ax2.add_patch(circle_inner)
ax2.add_patch(circle_outer)
ax2.add_patch(circle_center)
ax2.add_patch(circle_soft_in)
ax2.add_patch(circle_soft_out)

ax2.set_aspect('equal')
ax2.set_xlabel('X [m]')
ax2.set_ylabel('Y [m]')
ax2.set_title('Step 3 2D trajectory with lane constraints')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()