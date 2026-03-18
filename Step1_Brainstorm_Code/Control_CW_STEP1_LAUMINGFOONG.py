import numpy as np
import matplotlib.pyplot as plt


# Simulate the true 1D vehicle motion
# State x = [position, velocity]
def simulate_true_system(n_steps, dt, process_std, p0, v0):
    x_true = np.zeros((2, n_steps))
    x_true[:, 0] = [p0, v0]

    for k in range(1, n_steps):
        # Random change in velocity
        w = np.random.randn() * process_std

        # Previous position and velocity
        p_prev, v_prev = x_true[:, k - 1]

        # State update
        p_new = p_prev + dt * v_prev
        v_new = v_prev + w

        x_true[:, k] = [p_new, v_new]

    return x_true


# Generate noisy distance measurements from beacons
def generate_measurements(x_true, beacons, sigma_r):
    n_steps = x_true.shape[1]
    m = len(beacons)
    z = np.zeros((m, n_steps))

    for k in range(n_steps):
        p = x_true[0, k]

        for i, b in enumerate(beacons):
            # Distance measurement with noise
            z[i, k] = abs(p - b) + np.random.randn() * sigma_r

    return z


# State transition function
def f(x, dt):
    p, v = x
    return np.array([p + dt * v, v])


# Jacobian of the state transition
def F_jacobian(dt):
    return np.array([
        [1.0, dt],
        [0.0, 1.0]
    ])


# Measurement function
def h(x, beacons):
    p = x[0]
    return np.array([abs(p - b) for b in beacons])


# Jacobian of the measurement function
def H_jacobian(x, beacons):
    p = x[0]
    H = np.zeros((len(beacons), 2))

    for i, b in enumerate(beacons):
        if p > b:
            dp = 1.0
        elif p < b:
            dp = -1.0
        else:
            dp = 0.0

        H[i, 0] = dp
        H[i, 1] = 0.0

    return H


# Extended Kalman Filter
def ekf(beacons, z, dt, Q, R, x0_hat, P0):
    n_steps = z.shape[1]

    x_hat = np.zeros((2, n_steps))
    P_hist = np.zeros((2, 2, n_steps))

    x_hat[:, 0] = x0_hat
    P_hist[:, :, 0] = P0

    F = F_jacobian(dt)

    for k in range(1, n_steps):
        # Prediction step
        x_pred = f(x_hat[:, k - 1], dt)
        P_pred = F @ P_hist[:, :, k - 1] @ F.T + Q

        # Update step
        H = H_jacobian(x_pred, beacons)
        z_pred = h(x_pred, beacons)
        y = z[:, k] - z_pred

        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x_upd = x_pred + K @ y
        P_upd = (np.eye(2) - K @ H) @ P_pred

        x_hat[:, k] = x_upd
        P_hist[:, :, k] = P_upd

    return x_hat, P_hist


# Compute Root Mean Square Error
def compute_rmse(true_signal, est_signal):
    return np.sqrt(np.mean((true_signal - est_signal) ** 2))


def main():
    np.random.seed(42)

    # Simulation settings
    dt = 0.01
    T = 50
    n_steps = int(T / dt)

    # True initial state
    p0 = 10
    v0 = 10
    process_std = 0.2

    # Beacon positions and measurement noise
    beacons = np.array([-20.0, 120.0])
    sigma_r = 1.5

    # Simulate true system and measurements
    x_true = simulate_true_system(n_steps, dt, process_std, p0, v0)
    z = generate_measurements(x_true, beacons, sigma_r)

    # EKF tuning
    q_p = 0.01
    q_v = process_std ** 2
    Q = np.array([
        [q_p, 0.0],
        [0.0, q_v]
    ])

    R = (sigma_r ** 2) * np.eye(len(beacons))

    # Initial estimate
    x0_hat = np.array([0.0, 8.0])
    P0 = np.diag([25.0, 4.0])

    # Run EKF
    x_hat, _ = ekf(beacons, z, dt, Q, R, x0_hat, P0)

    # Compute errors
    pos_rmse = compute_rmse(x_true[0, :], x_hat[0, :])
    vel_rmse = compute_rmse(x_true[1, :], x_hat[1, :])

    print("===== STEP 1 RESULTS =====")
    print(f"Position RMSE: {pos_rmse:.3f} m")
    print(f"Velocity RMSE: {vel_rmse:.3f} m/s")

    # Time axis
    t = np.arange(n_steps) * dt

    # Plot position
    plt.figure(figsize=(10, 5))
    plt.plot(t, x_true[0, :], label="True position")
    plt.plot(t, x_hat[0, :], "--", label="Estimated position")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("True vs Estimated Position")
    plt.grid(True)
    plt.legend()

    # Plot velocity
    plt.figure(figsize=(10, 5))
    plt.plot(t, x_true[1, :], label="True velocity")
    plt.plot(t, x_hat[1, :], "--", label="Estimated velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.title("True vs Estimated Velocity")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
