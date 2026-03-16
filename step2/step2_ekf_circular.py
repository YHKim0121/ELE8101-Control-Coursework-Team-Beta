import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ============================================================
# Problem data
# ============================================================
h = 0.01
Q_omega = 1e-4
R_meas = 1.5 ** 2

# Circular track radius
R_track = 50.0

# Vehicle assumptions
v_nom = 10.0
v_max = 25.0

# Counter-clockwise motion
omega_nom = v_nom / R_track      # 0.2 rad/s
omega_min = 0.0
omega_max = v_max / R_track      # 0.5 rad/s

# State = [angle, angular_speed]
A = np.array([[1.0, h],
              [0.0, 1.0]])
G = np.array([[0.0],
              [1.0]])

# Beacon positions
beacons = np.array([
    [0.0, 50.0],
    [50.0, 0.0],
    [-30.0, -50.0]
])
n_beacons = len(beacons)

# Initial estimate
x0_tilde = np.array([[0.0],
                     [omega_nom]])

P0 = np.array([[0.5**2, 0.0],
               [0.0, 0.05**2]])

# ============================================================
# Helper functions
# ============================================================
def wrap_angle(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi

def wrap_angle_array(theta_array):
    return (theta_array + np.pi) % (2.0 * np.pi) - np.pi

def project_omega(omega):
    return np.clip(omega, omega_min, omega_max)

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
        dist = max(dist, 1e-12)

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
    x_mu[1, 0] = project_omega(x_mu[1, 0])

    sigma_mu = sigma_tu - K @ H @ sigma_tu
    return sigma_mu, x_mu

def time_update(sigma_mu, x_mu):
    x_tu = A @ x_mu
    x_tu[0, 0] = wrap_angle(x_tu[0, 0])
    x_tu[1, 0] = project_omega(x_tu[1, 0])

    sigma_tu = A @ sigma_mu @ A.T + Q_omega * (G @ G.T)
    return sigma_tu, x_tu

# ============================================================
# Simulation
# ============================================================
n_sim = 5000

# True initial state near nominal speed
x = np.array([[0.0],
              [omega_nom]]) + np.array([[0.1], [0.02]]) * np.random.randn(2, 1)
x[0, 0] = wrap_angle(x[0, 0])
x[1, 0] = project_omega(x[1, 0])

sigma_tu = P0.copy()
x_tu = x0_tilde.copy()
x_tu[0, 0] = wrap_angle(x_tu[0, 0])
x_tu[1, 0] = project_omega(x_tu[1, 0])

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

    w = np.random.normal(0.0, np.sqrt(Q_omega), 1)
    x = A @ x + G * w
    x[0, 0] = wrap_angle(x[0, 0])
    x[1, 0] = project_omega(x[1, 0])

    x_cache[t + 1, :] = x.T

# ============================================================
# Post-processing
# ============================================================
t_axis = np.arange(n_sim) * h

sigma_theta = np.sqrt(sigma_mu_cache[:, 0, 0])
sigma_omega = np.sqrt(sigma_mu_cache[:, 1, 1])

angle_error = wrap_angle_array(x_cache[:-1, 0] - x_mu_cache[:, 0])
omega_error = x_cache[:-1, 1] - x_mu_cache[:, 1]

# Unwrap only for plotting
theta_true_plot = np.unwrap(x_cache[:, 0])
theta_est_plot = np.unwrap(x_mu_cache[:, 0])

# Cartesian positions
x_true = R_track * np.cos(x_cache[:, 0])
y_true = R_track * np.sin(x_cache[:, 0])

x_est = R_track * np.cos(x_mu_cache[:, 0])
y_est = R_track * np.sin(x_mu_cache[:, 0])

# Convert angular speed to linear speed
v_true = R_track * x_cache[:, 1]
v_est = R_track * x_mu_cache[:, 1]
sigma_v = R_track * sigma_omega
v_error = v_true[:-1] - v_est

# ============================================================
# Plots
# ============================================================
fig, axs = plt.subplots(4, 1, figsize=(10, 12))

# Angle
axs[0].plot(t_axis, theta_true_plot, 'b-', linewidth=2, label='Actual angle')
axs[0].plot(t_axis[:-1], theta_est_plot, 'r-', linewidth=1.5, label='Estimated angle')
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Angle [rad]')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Linear speed
axs[1].plot(t_axis, v_true, 'b-', linewidth=2, label='Actual speed')
axs[1].plot(t_axis[:-1], v_est, 'r-', linewidth=1.5, label='Estimated speed')
axs[1].fill_between(
    t_axis[:-1],
    v_est - 3 * sigma_v,
    v_est + 3 * sigma_v,
    alpha=0.2,
    color='red',
    label=r'$\pm 3\sigma$'
)
axs[1].axhline(v_max, color='k', linestyle='--', linewidth=1, label='Max speed')
axs[1].axhline(0.0, color='gray', linestyle=':', linewidth=1)
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Speed [m/s]')
axs[1].legend()
axs[1].grid(True, alpha=0.3)

# Angle error
axs[2].plot(t_axis[:-1], angle_error, 'g-', label='Angle error')
axs[2].plot(t_axis[:-1], 3 * sigma_theta, 'r--', label=r'$\pm 3\sigma$')
axs[2].plot(t_axis[:-1], -3 * sigma_theta, 'r--')
axs[2].axhline(0, color='k', linewidth=0.5)
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Angle error [rad]')
axs[2].legend()
axs[2].grid(True, alpha=0.3)

# Speed error
axs[3].plot(t_axis[:-1], v_error, 'g-', label='Speed error')
axs[3].plot(t_axis[:-1], 3 * sigma_v, 'r--', label=r'$\pm 3\sigma$')
axs[3].plot(t_axis[:-1], -3 * sigma_v, 'r--')
axs[3].axhline(0, color='k', linewidth=0.5)
axs[3].set_xlabel('Time [s]')
axs[3].set_ylabel('Speed error [m/s]')
axs[3].legend()
axs[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# Circular trajectory plot
# ============================================================
plt.figure(figsize=(6, 6))
plt.plot(x_true, y_true, 'b-', linewidth=2, label='Actual trajectory')
plt.plot(x_est, y_est, 'r--', linewidth=1.5, label='Estimated trajectory')
plt.scatter(beacons[:, 0], beacons[:, 1], marker='x', s=100, label='Beacons')

# Mark start and end
plt.plot(x_true[0], y_true[0], 'go', markersize=8, label='Start')
plt.plot(x_true[-1], y_true[-1], 'mo', markersize=8, label='End')

plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Circular trajectory tracking with EKF')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()
