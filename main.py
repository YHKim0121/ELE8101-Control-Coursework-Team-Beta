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


def main():

    np.random.seed(42)

    dt = 0.01
    T = 5.0
    n_steps = int(T / dt)

    p0 = 10
    v0 = 10
    process_std = 0.2

    x_true = simulate_true_system(n_steps, dt, process_std, p0, v0)

    print("First 5 positions:")
    print(x_true[0, :5])

    print("First 5 velocities:")
    print(x_true[1, :5])


if __name__ == "__main__":
    main()