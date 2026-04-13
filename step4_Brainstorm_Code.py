import time
from dataclasses import dataclass
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# STEP 4 - ENGINEERED EKF WITH BIAS ESTIMATION
#
# State:
# x = [s, v_s, e, v_e, b]^T
#
# s   = longitudinal progress along track centreline [m]
# v_s = longitudinal speed [m/s]
# e   = lateral offset from centreline [m]
# v_e = lateral velocity [m/s]
# b   = constant bias on beacon 3 [m]
# ============================================================


# ============================================================
# 1. PARAMETERS
# ============================================================

@dataclass
class EstimatorParams:
    dt: float = 0.01
    v_target: float = 10.0
    k_vel: float = 0.1
    k_damp: float = 1.0
    k_restore: float = 0.5

    lane_half_width: float = 2.0
    true_bias: float = 1.5
    r_meas_std: float = 1.5

    # Track geometry
    R_left: float = 48.0
    R_right: float = 22.0
    d_centers: float = 100.0

    # Initial conditions
    x0: np.ndarray = None
    P0: np.ndarray = None

    # Noise covariances
    Q: np.ndarray = None
    R: np.ndarray = None

    # Beacon layout
    beacons: np.ndarray = None

    def __post_init__(self):
        if self.x0 is None:
            self.x0 = np.array([
                [0.0],
                [10.0],
                [0.0],
                [0.0],
                [0.0]
            ], dtype=float)

        if self.P0 is None:
            self.P0 = np.diag([1.0, 0.5, 0.5, 0.1, 2.0]).astype(float)

        if self.Q is None:
            self.Q = np.diag([0.001, 0.01, 0.001, 0.01, 1e-6]).astype(float)

        if self.R is None:
            self.R = (self.r_meas_std ** 2) * np.eye(4)

        if self.beacons is None:
            self.beacons = np.array([
                [-60.0,  60.0],   # B1
                [-60.0, -60.0],   # B2
                [130.0,  60.0],   # B3 (biased)
                [130.0, -60.0]    # B4
            ], dtype=float)


# ============================================================
# 2. TRACK MODEL
# ============================================================

class TrackGeometry:
    def __init__(self, params: EstimatorParams):
        self.p = params

        self.R1 = params.R_left
        self.R2 = params.R_right
        self.d = params.d_centers

        self.sin_alpha = (self.R1 - self.R2) / self.d
        self.alpha = np.arcsin(self.sin_alpha)

        self.L_str = self.d * np.cos(self.alpha)
        self.L_curveB = self.R2 * (np.pi - 2.0 * self.alpha)
        self.L_curveA = self.R1 * (np.pi + 2.0 * self.alpha)
        self.L_total = 2.0 * self.L_str + self.L_curveB + self.L_curveA

        self.P_A_top_x = self.R1 * np.sin(self.alpha)
        self.P_A_top_y = self.R1 * np.cos(self.alpha)
        self.P_B_bot_x = self.d + self.R2 * np.sin(self.alpha)
        self.P_B_bot_y = -self.R2 * np.cos(self.alpha)

    def wrap_progress(self, s):
        return np.mod(s, self.L_total)


# ============================================================
# 3. CASADI MEASUREMENT MODEL
# ============================================================

class CasadiMeasurementModel:
    def __init__(self, params: EstimatorParams, track: TrackGeometry):
        self.p = params
        self.track = track
        self.ny = params.beacons.shape[0]

        s_sym = cs.MX.sym("s")
        vs_sym = cs.MX.sym("vs")
        e_sym = cs.MX.sym("e")
        ve_sym = cs.MX.sym("ve")
        b_sym = cs.MX.sym("b")
        self.state_sym = cs.vertcat(s_sym, vs_sym, e_sym, ve_sym, b_sym)

        s_mod = cs.fmod(s_sym, track.L_total)

        # Segment 1: top straight
        x1 = track.P_A_top_x + s_mod * cs.cos(-track.alpha)
        y1 = track.P_A_top_y + s_mod * cs.sin(-track.alpha)
        psi1 = -track.alpha

        # Segment 2: right arc
        s2 = s_mod - track.L_str
        theta2 = (np.pi / 2.0 - track.alpha) - s2 / track.R2
        x2 = track.d + track.R2 * cs.cos(theta2)
        y2 = track.R2 * cs.sin(theta2)
        psi2 = theta2 - np.pi / 2.0

        # Segment 3: bottom straight
        s3 = s_mod - (track.L_str + track.L_curveB)
        x3 = track.P_B_bot_x + s3 * cs.cos(-np.pi + track.alpha)
        y3 = track.P_B_bot_y + s3 * cs.sin(-np.pi + track.alpha)
        psi3 = -np.pi + track.alpha

        # Segment 4: left arc
        s4 = s_mod - (2.0 * track.L_str + track.L_curveB)
        theta4 = (-np.pi / 2.0 + track.alpha) - s4 / track.R1
        x4 = track.R1 * cs.cos(theta4)
        y4 = track.R1 * cs.sin(theta4)
        psi4 = theta4 - np.pi / 2.0

        x_c = cs.if_else(
            s_mod < track.L_str, x1,
            cs.if_else(
                s_mod < track.L_str + track.L_curveB, x2,
                cs.if_else(
                    s_mod < 2.0 * track.L_str + track.L_curveB, x3, x4
                )
            )
        )

        y_c = cs.if_else(
            s_mod < track.L_str, y1,
            cs.if_else(
                s_mod < track.L_str + track.L_curveB, y2,
                cs.if_else(
                    s_mod < 2.0 * track.L_str + track.L_curveB, y3, y4
                )
            )
        )

        psi_c = cs.if_else(
            s_mod < track.L_str, psi1,
            cs.if_else(
                s_mod < track.L_str + track.L_curveB, psi2,
                cs.if_else(
                    s_mod < 2.0 * track.L_str + track.L_curveB, psi3, psi4
                )
            )
        )

        # Map lateral offset into global frame
        X_global = x_c - e_sym * cs.sin(psi_c)
        Y_global = y_c + e_sym * cs.cos(psi_c)

        h_list = []
        for i in range(self.ny):
            bx, by = params.beacons[i]
            dist = cs.sqrt((X_global - bx) ** 2 + (Y_global - by) ** 2)
            if i == 2:
                dist = dist + b_sym
            h_list.append(dist)

        h_sym = cs.vertcat(*h_list)

        self.pos_func = cs.Function("pos_func", [s_sym, e_sym], [X_global, Y_global])
        self.h_func = cs.Function("h_func", [self.state_sym], [h_sym])
        self.H_func = cs.Function("H_func", [self.state_sym], [cs.jacobian(h_sym, self.state_sym)])

    def measurement(self, x):
        return np.array(self.h_func(x)).astype(float).reshape(-1, 1)

    def jacobian(self, x):
        return np.array(self.H_func(x)).astype(float)

    def global_position(self, s, e):
        X, Y = self.pos_func(float(s), float(e))
        return float(X), float(Y)


# ============================================================
# 4. PROCESS MODEL
# ============================================================

class ProcessModel:
    def __init__(self, params: EstimatorParams):
        self.p = params

    def f(self, x):
        s, v_s, e, v_e, b = x.flatten()
        dt = self.p.dt

        return np.array([
            [s + v_s * dt],
            [v_s + self.p.k_vel * (self.p.v_target - v_s) * dt],
            [e + v_e * dt],
            [v_e - (self.p.k_damp * v_e + self.p.k_restore * e) * dt],
            [b]
        ], dtype=float)

    def A(self):
        dt = self.p.dt
        A = np.eye(5)
        A[0, 1] = dt
        A[1, 1] = 1.0 - self.p.k_vel * dt
        A[2, 3] = dt
        A[3, 2] = -self.p.k_restore * dt
        A[3, 3] = 1.0 - self.p.k_damp * dt
        return A


# ============================================================
# 5. EKF
# ============================================================

class EngineeredEKF:
    def __init__(self, params: EstimatorParams, process_model: ProcessModel, meas_model: CasadiMeasurementModel):
        self.p = params
        self.fm = process_model
        self.hm = meas_model
        self.nx = 5
        self.ny = params.beacons.shape[0]

    def predict(self, x_est, P_est):
        x_pred = self.fm.f(x_est)
        A = self.fm.A()
        P_pred = A @ P_est @ A.T + self.p.Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        return x_pred, P_pred

    def update(self, x_pred, P_pred, z):
        H = self.hm.jacobian(x_pred)
        z_hat = self.hm.measurement(x_pred)

        innovation = z - z_hat
        S = H @ P_pred @ H.T + self.p.R
        S = 0.5 * (S + S.T)

        K = P_pred @ H.T @ np.linalg.inv(S)
        x_upd = x_pred + K @ innovation

        # Joseph form covariance update
        I = np.eye(self.nx)
        P_upd = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ self.p.R @ K.T
        P_upd = 0.5 * (P_upd + P_upd.T)

        # NIS
        nis = (innovation.T @ np.linalg.solve(S, innovation)).item()

        return x_upd, P_upd, innovation, nis


# ============================================================
# 6. TRUE PLANT
# ============================================================

class TruePlant:
    def __init__(self, params: EstimatorParams, process_model: ProcessModel):
        self.p = params
        self.fm = process_model

    def step(self, x_true):
        w_true = np.random.multivariate_normal(np.zeros(5), self.p.Q).reshape(-1, 1)
        w_true[4, 0] = 0.0

        x_next = self.fm.f(x_true) + w_true
        x_next[4, 0] = self.p.true_bias

        # Hard boundary enforcement in the true plant
        if x_next[2, 0] > self.p.lane_half_width:
            x_next[2, 0] = self.p.lane_half_width
            x_next[3, 0] *= -0.5
        elif x_next[2, 0] < -self.p.lane_half_width:
            x_next[2, 0] = -self.p.lane_half_width
            x_next[3, 0] *= -0.5

        return x_next


# ============================================================
# 7. RUN SIMULATION
# ============================================================

def run_simulation(n_steps=4000):
    params = EstimatorParams()
    track = TrackGeometry(params)
    meas_model = CasadiMeasurementModel(params, track)
    process_model = ProcessModel(params)
    ekf = EngineeredEKF(params, process_model, meas_model)
    plant = TruePlant(params, process_model)

    x_true = params.x0.copy()
    x_est = params.x0.copy()
    P_est = params.P0.copy()

    x_true[4, 0] = params.true_bias

    true_hist = np.zeros((n_steps, 5))
    est_hist = np.zeros((n_steps, 5))
    P_hist = np.zeros((n_steps, 5, 5))
    innov_hist = np.zeros((n_steps, params.beacons.shape[0]))
    nis_hist = np.zeros(n_steps)

    boundary_hits = 0

    t0 = time.time()

    for k in range(n_steps):
        x_true_prev = x_true.copy()
        x_true = plant.step(x_true)

        if abs(x_true[2, 0]) >= params.lane_half_width and abs(x_true_prev[2, 0]) < params.lane_half_width:
            boundary_hits += 1

        z_true = meas_model.measurement(x_true)
        z = z_true + np.random.normal(0, params.r_meas_std, (params.beacons.shape[0], 1))

        x_pred, P_pred = ekf.predict(x_est, P_est)
        x_est, P_est, innovation, nis = ekf.update(x_pred, P_pred, z)

        true_hist[k] = x_true.flatten()
        est_hist[k] = x_est.flatten()
        P_hist[k] = P_est
        innov_hist[k] = innovation.flatten()
        nis_hist[k] = nis

    avg_step_time = (time.time() - t0) / n_steps

    return {
        "params": params,
        "track": track,
        "meas_model": meas_model,
        "true_hist": true_hist,
        "est_hist": est_hist,
        "P_hist": P_hist,
        "innov_hist": innov_hist,
        "nis_hist": nis_hist,
        "avg_step_time": avg_step_time,
        "boundary_hits": boundary_hits
    }


# ============================================================
# 8. METRICS
# ============================================================

def compute_metrics(results, burn_in=100):
    true_hist = results["true_hist"]
    est_hist = results["est_hist"]
    nis_hist = results["nis_hist"]
    params = results["params"]

    rmse_s = np.sqrt(np.mean((true_hist[burn_in:, 0] - est_hist[burn_in:, 0]) ** 2))
    rmse_e = np.sqrt(np.mean((true_hist[burn_in:, 2] - est_hist[burn_in:, 2]) ** 2))
    rmse_b = np.sqrt(np.mean((true_hist[burn_in:, 4] - est_hist[burn_in:, 4]) ** 2))
    final_bias_error = abs(params.true_bias - est_hist[-1, 4])
    mean_nis = np.mean(nis_hist[burn_in:])

    print("=" * 60)
    print("ENGINEERED STEP 4 EKF METRICS")
    print("=" * 60)
    print(f"Average execution time per step : {results['avg_step_time']:.6e} s")
    print(f"Longitudinal RMSE (s)           : {rmse_s:.4f} m")
    print(f"Lateral RMSE (e)                : {rmse_e:.4f} m")
    print(f"Bias RMSE (b)                   : {rmse_b:.4f} m")
    print(f"Final bias estimate             : {est_hist[-1, 4]:.4f} m")
    print(f"Final bias absolute error       : {final_bias_error:.4f} m")
    print(f"Mean NIS                        : {mean_nis:.4f}")
    print(f"Boundary hits                   : {results['boundary_hits']}")
    print("=" * 60)

    return {
        "rmse_s": rmse_s,
        "rmse_e": rmse_e,
        "rmse_b": rmse_b,
        "final_bias_error": final_bias_error,
        "mean_nis": mean_nis
    }


# ============================================================
# 9. PLOTTING
# ============================================================

def build_track_arrays(track, meas_model, lane_half_width, n_points=1200):
    s_vals = np.linspace(0.0, track.L_total, n_points)

    X_c, Y_c = [], []
    X_in, Y_in = [], []
    X_out, Y_out = [], []

    for s in s_vals:
        xc, yc = meas_model.global_position(s, 0.0)
        xin, yin = meas_model.global_position(s, -lane_half_width)
        xout, yout = meas_model.global_position(s, lane_half_width)

        X_c.append(xc)
        Y_c.append(yc)
        X_in.append(xin)
        Y_in.append(yin)
        X_out.append(xout)
        Y_out.append(yout)

    return np.array(X_c), np.array(Y_c), np.array(X_in), np.array(Y_in), np.array(X_out), np.array(Y_out)


def plot_results(results, metrics):
    params = results["params"]
    track = results["track"]
    meas_model = results["meas_model"]
    true_hist = results["true_hist"]
    est_hist = results["est_hist"]
    P_hist = results["P_hist"]
    innov_hist = results["innov_hist"]

    n_steps = true_hist.shape[0]
    t_axis = np.arange(n_steps) * params.dt

    # Plot 1: Lateral offset
    plt.figure(figsize=(10, 5))
    sigma_e = np.sqrt(P_hist[:, 2, 2])
    plt.plot(t_axis, true_hist[:, 2], 'b-', linewidth=1.5, label='True lateral offset')
    plt.plot(t_axis, est_hist[:, 2], 'r--', linewidth=1.5, label='Estimated lateral offset')
    plt.fill_between(
        t_axis,
        est_hist[:, 2] - 3 * sigma_e,
        est_hist[:, 2] + 3 * sigma_e,
        color='red',
        alpha=0.2,
        label=r'$\pm 3\sigma$'
    )
    plt.axhline(params.lane_half_width, color='k', linestyle='--', label='Lane bounds')
    plt.axhline(-params.lane_half_width, color='k', linestyle='--')
    plt.title(f'Lateral Offset Estimation (RMSE = {metrics["rmse_e"]:.4f} m)')
    plt.xlabel('Time [s]')
    plt.ylabel('Lateral offset [m]')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: Bias
    plt.figure(figsize=(10, 5))
    sigma_b = np.sqrt(P_hist[:, 4, 4])
    plt.plot(t_axis, true_hist[:, 4], 'b-', linewidth=2, label='True beacon-3 bias')
    plt.plot(t_axis, est_hist[:, 4], 'r--', linewidth=1.5, label='Estimated bias')
    plt.fill_between(
        t_axis,
        est_hist[:, 4] - 3 * sigma_b,
        est_hist[:, 4] + 3 * sigma_b,
        color='red',
        alpha=0.2,
        label=r'$\pm 3\sigma$'
    )
    plt.axhline(params.true_bias, color='k', linestyle='--', label='True bias')
    plt.ylim([0.0, 3.0])
    plt.title('Beacon 3 Bias Estimation')
    plt.xlabel('Time [s]')
    plt.ylabel('Bias [m]')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 3: Longitudinal progress
    plt.figure(figsize=(10, 5))
    plt.plot(t_axis, true_hist[:, 0], 'b-', linewidth=1.5, label='True s')
    plt.plot(t_axis, est_hist[:, 0], 'r--', linewidth=1.5, label='Estimated s')
    plt.title(f'Longitudinal Progress Estimation (RMSE = {metrics["rmse_s"]:.4f} m)')
    plt.xlabel('Time [s]')
    plt.ylabel('Progress s [m]')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 4: Track and trajectory
    X_c, Y_c, X_in, Y_in, X_out, Y_out = build_track_arrays(
        track, meas_model, params.lane_half_width
    )

    X_true, Y_true = [], []
    X_est, Y_est = [], []

    for i in range(0, n_steps, 5):
        xt, yt = meas_model.global_position(true_hist[i, 0], true_hist[i, 2])
        xe, ye = meas_model.global_position(est_hist[i, 0], est_hist[i, 2])
        X_true.append(xt)
        Y_true.append(yt)
        X_est.append(xe)
        Y_est.append(ye)

    plt.figure(figsize=(11, 6))
    plt.plot(X_out, Y_out, 'k-', linewidth=1.5, label='Outer boundary')
    plt.plot(X_in, Y_in, 'k-', linewidth=1.5, label='Inner boundary')
    plt.plot(X_c, Y_c, color='gray', linestyle='--', linewidth=1.0, label='Centreline')
    plt.plot(X_true, Y_true, 'b-', linewidth=1.0, label='True trajectory')
    plt.plot(X_est, Y_est, 'r--', linewidth=1.0, label='Estimated trajectory')

    plt.scatter(
        params.beacons[:, 0], params.beacons[:, 1],
        c='red', marker='^', s=160, edgecolors='black', label='Beacons'
    )

    for i, (bx, by) in enumerate(params.beacons):
        label = f'B{i+1} (biased)' if i == 2 else f'B{i+1}'
        plt.text(
            bx + 3, by + 3, label, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

    plt.title('Engineered Step 4 EKF on Coursework Track')
    plt.xlabel('Global X [m]')
    plt.ylabel('Global Y [m]')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 5: Innovations
    plt.figure(figsize=(10, 5))
    for i in range(innov_hist.shape[1]):
        plt.plot(t_axis, innov_hist[:, i], linewidth=1.0, label=f'Beacon {i+1} innovation')
    plt.title('Measurement Innovations')
    plt.xlabel('Time [s]')
    plt.ylabel('Innovation [m]')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 10. MAIN
# ============================================================

if __name__ == "__main__":
    results = run_simulation(n_steps=4000)
    metrics = compute_metrics(results, burn_in=100)
    plot_results(results, metrics)