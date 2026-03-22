import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

np.random.seed(0)

# Problem parameters
h = 0.01  # 100Hz beacon freq
Q = 0.01  # Small Q, assume constant longitudinal velocity (approx 10m/s)
R = 1.5**2  # Measurement noise variance

# Linear dynamics matrices (arc length and velocity)
A = np.array([[1, h], [0, 1]])
G = np.array([[0], [1]])

# Assume the vehicle stays exactly at the center of the lane
R_c = 48.0 

# Place beacons to avoid ambiguity: 2 outside, 1 inside
beacons = np.array([
    [0, -100],      
    [100, 100],    
    [-100, 100]    
])
ny = len(beacons)
R_mat = R * np.eye(ny)

# Assume initial state is relatively well known (small P0)
x0_tilde = np.array([[0], [10]]) 
P0 = np.array([[1, 0], [0, 1]])  

# CasADi symbolic variables for nonlinear measurement model
x_sym = cs.SX.sym('x', 2)
v_sym = cs.SX.sym('v', ny)

p_t = x_sym[0]
pos_x = R_c * cs.cos(p_t / R_c)
pos_y = R_c * cs.sin(p_t / R_c)

h_list = []
for i in range(ny):
    bx = beacons[i, 0]
    by = beacons[i, 1]
    # Nonlinear distance formula
    dist = cs.sqrt((pos_x - bx)**2 + (pos_y - by)**2)
    h_list.append(dist + v_sym[i])

h_sym = cs.vertcat(*h_list)

# Jacobian for EKF
jhx = cs.Function('jhx', [x_sym, v_sym], [cs.jacobian(h_sym, x_sym)])
h_func = cs.Function('h_func', [x_sym, v_sym], [h_sym])

def measurement_update(sigma_tu, x_tu, y):
    C_mat = np.array(jhx(x_tu, np.zeros(ny))).astype(float)
    Z = C_mat @ sigma_tu @ C_mat.T + R_mat
    y_pred = np.array(h_func(x_tu, np.zeros(ny))).flatten()
    
    x_mu = x_tu + (sigma_tu @ C_mat.T @ np.linalg.solve(Z, y - y_pred)).reshape(-1, 1)
    sigma_mu = sigma_tu - sigma_tu @ C_mat.T @ np.linalg.solve(Z, C_mat @ sigma_tu)
    return sigma_mu, x_mu

def time_update(sigma_mu, x_mu):
    x_tu = A @ x_mu
    sigma_tu = A @ sigma_mu @ A.T + Q * (G @ G.T)
    return sigma_tu, x_tu

n_sim = 5000
x = np.random.multivariate_normal(x0_tilde.flatten(), P0, 1).reshape((-1,1))
sigma_tu, x_tu = P0, x0_tilde.copy()

x_cache = np.zeros((n_sim, 2))
x_mu_cache = np.zeros((n_sim-1, 2))
sigma_mu_cache = np.zeros((n_sim-1, 2, 2))
x_cache[0, :] = x.T

# EKF main loop
for t in range(n_sim - 1):
    # Assume Gaussian noise for both measurement and process
    v = np.random.normal(0, np.sqrt(R), ny)
    y = np.array(h_func(x, v)).flatten()
    
    sigma_mu, x_mu = measurement_update(sigma_tu, x_tu, y)
    
    x_mu_cache[t, :] = x_mu.flatten()
    sigma_mu_cache[t] = sigma_mu
    
    sigma_tu, x_tu = time_update(sigma_mu, x_mu)
    
    w = np.random.normal(0, np.sqrt(Q))
    x = A @ x + G * w
    x_cache[t+1, :] = x.T

t_axis = np.arange(n_sim) * h

sigma_p  = np.sqrt(sigma_mu_cache[:, 0, 0])
sigma_vv = np.sqrt(sigma_mu_cache[:, 1, 1])

# Plot 1: States and Errors (1D perspective)
fig1, axs = plt.subplots(4, 1, figsize=(8, 12))

axs[0].plot(t_axis, x_cache[:, 0], 'b-', label='Actual p_t')
axs[0].plot(t_axis[:-1], x_mu_cache[:, 0], 'r-', label='Estimated p_t')
axs[0].fill_between(t_axis[:-1], x_mu_cache[:, 0] - 3*sigma_p,
                    x_mu_cache[:, 0] + 3*sigma_p, alpha=0.2, color='red', label='+/- 3 sigma')
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Position [m]')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

axs[1].plot(t_axis, x_cache[:, 1], 'b-', label='Actual v_t')
axs[1].plot(t_axis[:-1], x_mu_cache[:, 1], 'r-', label='Estimated v_t')
axs[1].fill_between(t_axis[:-1], x_mu_cache[:, 1] - 3*sigma_vv,
                    x_mu_cache[:, 1] + 3*sigma_vv, alpha=0.2, color='red', label='+/- 3 sigma')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Velocity [m/s]')
axs[1].legend()
axs[1].grid(True, alpha=0.3)

axs[2].plot(t_axis[:-1], x_cache[:-1, 0] - x_mu_cache[:, 0], 'g-', label='Position error')
axs[2].plot(t_axis[:-1],  3*sigma_p, 'r--', label='+/- 3 sigma')
axs[2].plot(t_axis[:-1], -3*sigma_p, 'r--')
axs[2].axhline(0, color='k', linewidth=0.5)
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Position error [m]')
axs[2].legend()
axs[2].grid(True, alpha=0.3)

axs[3].plot(t_axis[:-1], x_cache[:-1, 1] - x_mu_cache[:, 1], 'g-', label='Velocity error')
axs[3].plot(t_axis[:-1],  3*sigma_vv, 'r--', label='+/- 3 sigma')
axs[3].plot(t_axis[:-1], -3*sigma_vv, 'r--')
axs[3].axhline(0, color='k', linewidth=0.5)
axs[3].set_xlabel('Time [s]')
axs[3].set_ylabel('Velocity error [m/s]')
axs[3].legend()
axs[3].grid(True, alpha=0.3)

plt.tight_layout()

# Plot 2: 2D Trajectory Map
fig2, ax2 = plt.subplots(figsize=(8, 8))
theta_true = x_cache[:, 0] / R_c
x_2d_true = R_c * np.cos(theta_true)
y_2d_true = R_c * np.sin(theta_true)

theta_est = x_mu_cache[:, 0] / R_c
x_2d_est = R_c * np.cos(theta_est)
y_2d_est = R_c * np.sin(theta_est)

ax2.plot(x_2d_true, y_2d_true, 'b-', label='Actual Trajectory')
ax2.plot(x_2d_est, y_2d_est, 'r--', label='Estimated Trajectory', alpha=0.7)
ax2.scatter(beacons[:, 0], beacons[:, 1], c='k', marker='^', s=100, label='Beacons')

circle_inner = plt.Circle((0, 0), 46, color='gray', fill=False, linestyle=':')
circle_outer = plt.Circle((0, 0), 50, color='gray', fill=False, linestyle=':')
ax2.add_patch(circle_inner)
ax2.add_patch(circle_outer)

ax2.set_aspect('equal')
ax2.set_xlabel('X [m]')
ax2.set_ylabel('Y [m]')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.show()