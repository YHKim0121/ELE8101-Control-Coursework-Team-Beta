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

    print("First 5 positions:")
    print(x_true[0, :5])

    print("First 5 velocities:")
    print(x_true[1, :5])

    print("First 5 measurements from beacon 1:")
    print(z[0, :5])

    print("First 5 measurements from beacon 2:")
    print(z[1, :5])


if __name__ == "__main__":
    main()