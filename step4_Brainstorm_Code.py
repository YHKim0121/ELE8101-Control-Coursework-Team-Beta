import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

np.random.seed(0)

# ============================================================
# Step 3: 4-State EKF with Ornstein-Uhlenbeck (OU) Dynamics
# ============================================================
h = 0.01  # Sample time (100 Hz)
R_c = 48.0
lane_half_width = 2.0

# Process and measurement noise
Q_long = 0.01
Q_lat = 0.001 
Qw = np.diag([Q_long, Q_lat])
R_meas = 1.5**2

# OU process parameters (Restoring force for lane keeping)
k_restore = 0.05
k_damp = 0.9
k_vel = 0.1
v_target = 10.0

# Initial state: x = [s, v_s, d, v_d]^T
nx = 4
x0_tilde = np.array([[0.0], [10.0], [0.0], [0.0]])
P0 = np.diag([1.0, 1.0, 0.5, 0.1])

# ============================================================
# System Dynamics (CasADi)
# ============================================================
s = cs.MX.sym('s')
v_s = cs.MX.sym('v_s')
d = cs.MX.sym('d')
v_d = cs.MX.sym('v_d')
x_sym = cs.vertcat(s, v_s, d, v_d)

w_vs = cs.MX.sym('w_vs')
w_vd = cs.MX.sym('w_vd')
w_sym = cs.vertcat(w_vs, w_vd)

# Discrete-time state equations
s_next = s + v_s * h
v_s_next = v_s + k_vel * (v_target - v_s) * h + w_vs
d_next = d + v_d * h
v_d_next = v_d - (k_damp * v_d + k_restore * d) * h + w_vd

x_next = cs.vertcat(s_next, v_s_next, d_next, v_d_next)

f_dynamics = cs.Function('f_dynamics', [x_sym, w_sym], [x_next])
A_jac = cs.Function('A_jac', [x_sym, w_sym], [cs.jacobian(x_next, x_sym)])
G_jac = cs.Function('G_jac', [x_sym, w_sym], [cs.jacobian(x_next, w_sym)])

# ============================================================
# Beacon Measurement Model
# ============================================================
beacons = np.array([[0, -100], [100, 100], [-100, 100]])
ny = len(beacons)
R_mat = R_meas * np.eye(ny)

v_sym = cs.MX.sym('v', ny)
theta_sym = x_sym[0] / R_c
radius_sym = R_c + x_sym[2]

pos_x = radius_sym * cs.cos(theta_sym)
pos_y = radius_sym * cs.sin(theta_sym)

h_list = []
for i in range(ny):
    bx, by = beacons[i]
    dist = cs.sqrt((pos_x - bx)**2 + (pos_y - by)**2)
    h_list.append(dist + v_sym[i])

h_sym = cs.vertcat(*h_list)
jhx = cs.Function('jhx', [x_sym, v_sym], [cs.jacobian(h_sym, x_sym)])
h_func = cs.Function('h_func', [x_sym, v_sym], [h_sym])

# ============================================================
# EKF Functions
# ============================================================
def measurement_update(P_pred, x_pred, y):
    C = np.array(jhx(x_pred, np.zeros(ny))).astype(float)
    y_pred = np.array(h_func(x_pred, np.zeros(ny))).astype(float).flatten()

    S = C @ P_pred @ C.T + R_mat
    K = P_pred @ C.T @ np.linalg.inv(S)

    innovation = y - y_pred
    x_upd = x_pred + K @ innovation.reshape(-1, 1)
    P_upd = (np.eye(nx) - K @ C) @ P_pred

    # Enforce covariance symmetry to prevent numerical drift
    P_upd = (P_upd + P_upd.T) / 2.0

    # Note: EKF does NOT use hard clipping to preserve the Gaussian assumption
    return P_upd, x_upd

def time_update(P_upd, x_upd):
    w_zero = np.zeros((2, 1))
    x_pred = np.array(f_dynamics(x_upd, w_zero))

    A_val = np.array(A_jac(x_upd, w_zero))
    G_val = np.array(G_jac(x_upd, w_zero))

    P_pred = A_val @ P_upd @ A_val.T + G_val @ Qw @ G_val.T
    return P_pred, x_pred

# ============================================================
# Simulation Setup & Main Loop
# ============================================================
n_sim = 5000
x_true = x0_tilde.copy()
x_pred = x0_tilde.copy()
P_pred = P0.copy()

x_true_hist = np.zeros((n_sim, nx))
x_est_hist = np.zeros((n_sim - 1, nx))
P_hist = np.zeros((n_sim - 1, nx, nx))

x_true_hist[0, :] = x_true.flatten()
boundary_violations = 0

for t in range(n_sim - 1):
    # 1. Generate true measurements
    meas_noise = np.random.normal(0, np.sqrt(R_meas), ny)
    y = np.array(h_func(x_true, meas_noise)).astype(float).flatten()

    # 2. EKF measurement update
    P_upd, x_upd = measurement_update(P_pred, x_pred, y)

    x_est_hist[t, :] = x_upd.flatten()
    P_hist[t] = P_upd

    # 3. EKF time update
    P_pred, x_pred = time_update(P_upd, x_upd)

    # 4. Propagate true system
    w_true = np.array([[np.random.normal(0, np.sqrt(Q_long))],
                       [np.random.normal(0, np.sqrt(Q_lat))]])
    x_true = np.array(f_dynamics(x_true, w_true))

    # Track boundary violations
    if abs(x_true[2, 0]) > lane_half_width:
        boundary_violations += 1

    x_true_hist[t + 1, :] = x_true.flatten()

# ============================================================
# Evaluation: RMSE & Metrics
# ============================================================
t_axis = np.arange(n_sim) * h

p_rmse = np.sqrt(np.mean((x_true_hist[1:, 0] - x_est_hist[:, 0])**2))
e_rmse = np.sqrt(np.mean((x_true_hist[1:, 2] - x_est_hist[:, 2])**2))

print("--- Step 3 EKF Performance ---")
print(f"Longitudinal (s) RMSE : {p_rmse:.4f} m")
print(f"Lateral (d) RMSE      : {e_rmse:.4f} m")
print(f"Boundary Violations   : {boundary_violations} times out of {n_sim} steps")

# ============================================================
# Visualization
# ============================================================
sigma_e = np.sqrt(P_hist[:, 2, 2])

plt.figure(figsize=(10, 4))
plt.plot(t_axis, x_true_hist[:, 2], 'b-', label='Actual lateral offset')
plt.plot(t_axis[:-1], x_est_hist[:, 2], 'r--', label='Estimated offset')
plt.fill_between(t_axis[:-1], x_est_hist[:, 2] - 3*sigma_e, x_est_hist[:, 2] + 3*sigma_e, color='red', alpha=0.2, label='±3 sigma')
plt.axhline(lane_half_width, color='k', linestyle='--', label='Lane bounds')
plt.axhline(-lane_half_width, color='k', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('Lateral offset [m]')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(8, 8))
theta_true = x_true_hist[:, 0] / R_c
x_2d_true = (R_c + x_true_hist[:, 2]) * np.cos(theta_true)
y_2d_true = (R_c + x_true_hist[:, 2]) * np.sin(theta_true)

theta_est = x_est_hist[:, 0] / R_c
x_2d_est = (R_c + x_est_hist[:, 2]) * np.cos(theta_est)
y_2d_est = (R_c + x_est_hist[:, 2]) * np.sin(theta_est)

ax2.plot(x_2d_true, y_2d_true, color='blue', linewidth=2, label='Actual trajectory')
ax2.plot(x_2d_est, y_2d_est, color='red', linestyle='--', linewidth=1.5, label='Estimated trajectory')
ax2.scatter(beacons[:, 0], beacons[:, 1], c='k', marker='^', s=100, label='Beacons')

ax2.add_patch(plt.Circle((0, 0), R_c - lane_half_width, color='gray', fill=False, linestyle=':'))
ax2.add_patch(plt.Circle((0, 0), R_c + lane_half_width, color='gray', fill=False, linestyle=':'))
ax2.add_patch(plt.Circle((0, 0), R_c, color='gray', fill=False, linestyle='--', alpha=0.7))

ax2.set_aspect('equal')
ax2.set_xlabel('X [m]')
ax2.set_ylabel('Y [m]')
ax2.grid(True, alpha=0.3)
ax2.legend()
plt.show()