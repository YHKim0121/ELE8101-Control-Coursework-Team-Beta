import numpy as np


def simulate_true_system(n_steps, dt, process_std, p0, v0):
    x_true = np.zeros((2, n_steps))
    x_true[:, 0] = [p0, v0]

    for k in range(1, n_steps):
        w = np.random.randn() * process_std
        p_prev, v_prev = x_true[:, k - 1]

        p_new = p_prev + dt * v_prev
        v_new = v_prev + w

        x_true[:, k] = [p_new, v_new]

    return x_true


def generate_measurements(x_true, beacons, sigma_r):
    n_steps = x_true.shape[1]
    m = len(beacons)
    z = np.zeros((m, n_steps))

    for k in range(n_steps):
        p = x_true[0, k]
        for i, b in enumerate(beacons):
            z[i, k] = abs(p - b) + np.random.randn() * sigma_r

    return z


def f(x, dt):
    p, v = x
    return np.array([p + dt * v, v])


def F_jacobian(dt):
    return np.array([
        [1.0, dt],
        [0.0, 1.0]
    ])


def h(x, beacons):
    p = x[0]
    return np.array([abs(p - b) for b in beacons])


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


def ekf(beacons, z, dt, Q, R, x0_hat, P0):
    n_steps = z.shape[1]
    x_hat = np.zeros((2, n_steps))
    P_hist = np.zeros((2, 2, n_steps))

    x_hat[:, 0] = x0_hat
    P_hist[:, :, 0] = P0

    F = F_jacobian(dt)

    for k in range(1, n_steps):
        x_pred = f(x_hat[:, k - 1], dt)
        P_pred = F @ P_hist[:, :, k - 1] @ F.T + Q

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


def compute_rmse(true_signal, est_signal):
    return np.sqrt(np.mean((true_signal - est_signal) ** 2))


def main():
    np.random.seed(42)

    dt = 0.01
    T = 5.0
    n_steps = int(T / dt)

    p0 = 10
    v0 = 10
    process_std = 0.2

    beacons = np.array([-20.0, 120.0])
    sigma_r = 1.5

    x_true = simulate_true_system(n_steps, dt, process_std, p0, v0)
    z = generate_measurements(x_true, beacons, sigma_r)

    q_p = 0.01
    q_v = process_std ** 2
    Q = np.array([
        [q_p, 0.0],
        [0.0, q_v]
    ])

    R = (sigma_r ** 2) * np.eye(len(beacons))

    x0_hat = np.array([0.0, 8.0])
    P0 = np.diag([25.0, 4.0])

    x_hat, _ = ekf(beacons, z, dt, Q, R, x0_hat, P0)

    pos_rmse = compute_rmse(x_true[0, :], x_hat[0, :])
    vel_rmse = compute_rmse(x_true[1, :], x_hat[1, :])

    print("===== STEP 1 RESULTS =====")
    print(f"Position RMSE: {pos_rmse:.3f} m")
    print(f"Velocity RMSE: {vel_rmse:.3f} m/s")


if __name__ == "__main__":
    main()