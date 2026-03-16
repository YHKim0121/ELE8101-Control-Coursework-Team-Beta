import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ============================================================
# Problem data
# ============================================================
h = 0.01
Q = 0.01
R_meas = 1.5 ** 2

# State = [angle, angular_speed]
A = np.array([[1.0, h],
              [0.0, 1.0]])
G = np.array([[0.0],
              [1.0]])

# Circular track radius
R_track = 50.0

# Beacon positions
beacons = np.array([
    [0.0, 50.0],
    [50.0, 0.0],
    [-30.0, -50.0]
])

n_beacons = len(beacons)

# Initial estimate: [angle, angular speed]
x0_tilde = np.array([[0.0],
                     [10.0]])
P0 = np.array([[100.0, 0.0],
               [0.0, 1.0]])

# ============================================================
# Helper functions
# ============================================================
def wrap_angle(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi

def wrap_angle_array(theta_array):
    return (theta_array + np.pi) % (2.0 * np.pi) - np.pi

def circle_position(theta):
    px = R_track * np.cos(theta)
    py = R_track * np.sin(theta)
    return px, py

# ============================================================
# Nonlinear beacon measurement model
# ============================================================
def beacon_measurement(theta):
    px, py = circle_position(theta)
    distances = []
    for bx, by in beacons:
        d = np.sqrt((px - bx) ** 2 + (py - by) ** 2)
        distances.append(d)
    return np.array(distances).reshape(-1, 1)

def beacon_jacobian(theta):
    px, py = circle_position(theta)

    dpx_dtheta = -R_track * np.sin(theta)
    dpy_dtheta =  R_track * np.cos(theta)

    H = []
    for bx, by in beacons:
        dx = px - bx
        dy = py - by
        dist = np.sqrt(dx ** 2 + dy ** 2)

        if dist < 1e-12:
            dist = 1e-12

        dh_dtheta = (dx * dpx_dtheta + dy * dpy_dtheta) / dist
        H.append([dh_dtheta, 0.0])

    return np.array(H)

# ============================================================
# EKF update functions
# ============================================================
def measurement_update(sigma_tu, x_tu, y):
    theta = x_tu[0, 0]

    h_theta = beacon_measurement(theta)
    H = beacon_jacobian(theta)

    innovation = y - h_theta
    S = H @ sigma_tu @ H.T + R_meas * np.eye(n_beacons)
    K = sigma_tu @ H.T @ np.linalg.solve(S, np.eye(n_beacons))

    x_mu = x_tu + K @ innovation
    x_mu[0, 0] = wrap_angle(x_mu[0, 0])

    sigma_mu = sigma_tu - K @ H @ sigma_tu

    return sigma_mu, x_mu

def time_update(sigma_mu, x_mu):
    x_tu = A @ x_mu
    x_tu[0, 0] = wrap_angle(x_tu[0, 0])

    sigma_tu = A @ sigma_mu @ A.T + Q * (G @ G.T)
    return sigma_tu, x_tu

# ============================================================
# Simulation
# ============================================================
n_sim = 5000

x = np.random.multivariate_normal(x0_tilde.flatten(), P0, 1).reshape((-1, 1))
x[0, 0] = wrap_angle(x[0, 0])

sigma_tu = P0.copy()
x_tu = x0_tilde.reshape((-1, 1)).copy()
x_tu[0, 0] = wrap_angle(x_tu[0, 0])

x_cache = np.zeros((n_sim, 2))
x_mu_cache = np.zeros((n_sim - 1, 2))
sigma_mu_cache = np.zeros((n_sim - 1, 2, 2))

x_cache[0, :] = x.T

for t in range(n_sim - 1):
    theta_true = x[0, 0]
    y = beacon_measurement(theta_true) + np.random.normal(
        0.0, np.sqrt(R_meas), (n_beacons, 1)
    )

    sigma_mu, x_mu = measurement_update(sigma_tu, x_tu, y)

    x_mu_cache[t, :] = x_mu.T
    sigma_mu_cache[t] = sigma_mu

    sigma_tu, x_tu = time_update(sigma_mu, x_mu)

    w = np.random.normal(0.0, np.sqrt(Q), 1)
    x = A @ x + G * w
    x[0, 0] = wrap_angle(x[0, 0])

    x_cache[t + 1, :] = x.T

# ============================================================
# Post-processing
# ============================================================
t_axis = np.arange(n_sim) * h

sigma_theta = np.sqrt(sigma_mu_cache[:, 0, 0])
sigma_omega = np.sqrt(sigma_mu_cache[:, 1, 1])

angle_error = wrap_angle_array(x_cache[:-1, 0] - x_mu_cache[:, 0])
omega_error = x_cache[:-1, 1] - x_mu_cache[:, 1]

# Unwrap ONLY for plotting
theta_true_plot = np.unwrap(x_cache[:, 0])
theta_est_plot = np.unwrap(x_mu_cache[:, 0])

x_true = R_track * np.cos(x_cache[:, 0])
y_true = R_track * np.sin(x_cache[:, 0])

x_est = R_track * np.cos(x_mu_cache[:, 0])
y_est = R_track * np.sin(x_mu_cache[:, 0])

# ============================================================
# Plots
# ============================================================
fig, axs = plt.subplots(4, 1, figsize=(10, 12))

axs[0].plot(t_axis, theta_true_plot, 'b-', label='Actual angle')
axs[0].plot(t_axis[:-1], theta_est_plot, 'r-', label='Estimated angle')
axs[0].set_xlabel('Time, $t$ [s]')
axs[0].set_ylabel('Angle [rad]')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

axs[1].plot(t_axis, x_cache[:, 1], 'b-', label='Actual angular speed')
axs[1].plot(t_axis[:-1], x_mu_cache[:, 1], 'r-', label='Estimated angular speed')
axs[1].fill_between(
    t_axis[:-1],
    x_mu_cache[:, 1] - 3 * sigma_omega,
    x_mu_cache[:, 1] + 3 * sigma_omega,
    alpha=0.2,
    color='red',
    label=r'$\pm 3\sigma$'
)
axs[1].set_xlabel('Time, $t$ [s]')
axs[1].set_ylabel('Angular speed [rad/s]')
axs[1].legend()
axs[1].grid(True, alpha=0.3)

axs[2].plot(t_axis[:-1], angle_error, 'g-', label='Angle error')
axs[2].plot(t_axis[:-1], 3 * sigma_theta, 'r--', label=r'$\pm 3\sigma$')
axs[2].plot(t_axis[:-1], -3 * sigma_theta, 'r--')
axs[2].axhline(0, color='k', linewidth=0.5)
axs[2].set_xlabel('Time, $t$ [s]')
axs[2].set_ylabel('Angle error [rad]')
axs[2].legend()
axs[2].grid(True, alpha=0.3)

axs[3].plot(t_axis[:-1], omega_error, 'g-', label='Angular speed error')
axs[3].plot(t_axis[:-1], 3 * sigma_omega, 'r--', label=r'$\pm 3\sigma$')
axs[3].plot(t_axis[:-1], -3 * sigma_omega, 'r--')
axs[3].axhline(0, color='k', linewidth=0.5)
axs[3].set_xlabel('Time, $t$ [s]')
axs[3].set_ylabel('Angular speed error [rad/s]')
axs[3].legend()
axs[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# Circular trajectory plot
# ============================================================
plt.figure(figsize=(6, 6))
plt.plot(x_true, y_true, 'b-', label='Actual trajectory')
plt.plot(x_est, y_est, 'r--', label='Estimated trajectory')
plt.scatter(beacons[:, 0], beacons[:, 1], marker='x', s=100, label='Beacons')

plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Circular trajectory tracking with EKF')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()