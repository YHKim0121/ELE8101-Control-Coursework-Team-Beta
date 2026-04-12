import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

np.random.seed(0)


class EstimatorParams:
    def __init__(self):
        self.dt = 0.01
        self.v_target = 10.0
        self.k_vel = 0.1
        self.k_damp = 0.9
        self.k_restore = 0.05

        self.lane_half_width = 2.0

        self.R_left = 48.0
        self.R_right = 22.0
        self.d_centers = 100.0

        self.Q_long = 0.01
        self.Q_lat = 0.001
        self.Qw = np.diag([self.Q_long, self.Q_lat])

        self.R_meas = 1.5**2
        self.beacons = np.array([
            [0.0, -100.0],
            [100.0, 100.0],
            [-100.0, 100.0]
        ])

        self.nx = 4
        self.ny = self.beacons.shape[0]

        self.R_mat = self.R_meas * np.eye(self.ny)
        self.x0 = np.array([[0.0], [10.0], [0.0], [0.0]])
        self.P0 = np.diag([1.0, 1.0, 0.5, 0.1])


class TrackGeometry:
    def __init__(self, params: EstimatorParams):
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


class ProcessModel:
    def __init__(self, params: EstimatorParams):
        self.p = params

        s = cs.MX.sym('s')
        v_s = cs.MX.sym('v_s')
        e = cs.MX.sym('e')
        v_e = cs.MX.sym('v_e')
        x_sym = cs.vertcat(s, v_s, e, v_e)

        w_vs = cs.MX.sym('w_vs')
        w_ve = cs.MX.sym('w_ve')
        w_sym = cs.vertcat(w_vs, w_ve)

        s_next = s + v_s * self.p.dt
        v_s_next = v_s + self.p.k_vel * (self.p.v_target - v_s) * self.p.dt + w_vs
        e_next = e + v_e * self.p.dt
        v_e_next = v_e - (self.p.k_damp * v_e + self.p.k_restore * e) * self.p.dt + w_ve

        x_next = cs.vertcat(s_next, v_s_next, e_next, v_e_next)

        self.f_dynamics = cs.Function('f_dynamics', [x_sym, w_sym], [x_next])
        self.A_jac = cs.Function('A_jac', [x_sym, w_sym], [cs.jacobian(x_next, x_sym)])
        self.G_jac = cs.Function('G_jac', [x_sym, w_sym], [cs.jacobian(x_next, w_sym)])

    def propagate(self, x, w):
        return np.array(self.f_dynamics(x, w)).astype(float)

    def linearise(self, x, w):
        A = np.array(self.A_jac(x, w)).astype(float)
        G = np.array(self.G_jac(x, w)).astype(float)
        return A, G


class CasadiMeasurementModel:
    def __init__(self, params: EstimatorParams, track: TrackGeometry):
        self.p = params
        self.track = track
        self.ny = params.beacons.shape[0]

        s_sym = cs.MX.sym("s")
        vs_sym = cs.MX.sym("vs")
        e_sym = cs.MX.sym("e")
        ve_sym = cs.MX.sym("ve")
        self.state_sym = cs.vertcat(s_sym, vs_sym, e_sym, ve_sym)

        s_mod = cs.fmod(s_sym, track.L_total)

        x1 = track.P_A_top_x + s_mod * cs.cos(-track.alpha)
        y1 = track.P_A_top_y + s_mod * cs.sin(-track.alpha)
        psi1 = -track.alpha

        s2 = s_mod - track.L_str
        theta2 = (np.pi / 2.0 - track.alpha) - s2 / track.R2
        x2 = track.d + track.R2 * cs.cos(theta2)
        y2 = track.R2 * cs.sin(theta2)
        psi2 = theta2 - np.pi / 2.0

        s3 = s_mod - (track.L_str + track.L_curveB)
        x3 = track.P_B_bot_x + s3 * cs.cos(-np.pi + track.alpha)
        y3 = track.P_B_bot_y + s3 * cs.sin(-np.pi + track.alpha)
        psi3 = -np.pi + track.alpha

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

        X_global = x_c - e_sym * cs.sin(psi_c)
        Y_global = y_c + e_sym * cs.cos(psi_c)

        h_list = []
        for i in range(self.ny):
            bx, by = params.beacons[i]
            dist = cs.sqrt((X_global - bx)**2 + (Y_global - by)**2)
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


class EKF:
    def __init__(self, params: EstimatorParams, process_model: ProcessModel, meas_model: CasadiMeasurementModel):
        self.p = params
        self.fm = process_model
        self.hm = meas_model

    def measurement_update(self, P_pred, x_pred, z):
        H = self.hm.jacobian(x_pred)
        z_hat = self.hm.measurement(x_pred)

        innovation = z - z_hat
        S = H @ P_pred @ H.T + self.p.R_mat
        K = P_pred @ H.T @ np.linalg.inv(S)

        x_upd = x_pred + K @ innovation
        P_upd = (np.eye(self.p.nx) - K @ H) @ P_pred
        P_upd = 0.5 * (P_upd + P_upd.T)

        return x_upd, P_upd

    def time_update(self, P_upd, x_upd):
        w_zero = np.zeros((2, 1))
        x_pred = self.fm.propagate(x_upd, w_zero)
        A, G = self.fm.linearise(x_upd, w_zero)
        P_pred = A @ P_upd @ A.T + G @ self.p.Qw @ G.T
        return x_pred, P_pred


def run_simulation(n_sim=5000):
    params = EstimatorParams()
    track = TrackGeometry(params)
    process_model = ProcessModel(params)
    meas_model = CasadiMeasurementModel(params, track)
    ekf = EKF(params, process_model, meas_model)

    x_true = params.x0.copy()
    x_pred = params.x0.copy()
    P_pred = params.P0.copy()

    x_true_hist = np.zeros((n_sim, params.nx))
    x_est_hist = np.zeros((n_sim - 1, params.nx))
    P_hist = np.zeros((n_sim - 1, params.nx, params.nx))

    x_true_hist[0, :] = x_true.flatten()

    for t in range(n_sim - 1):
        z_true = meas_model.measurement(x_true)
        z = z_true + np.random.normal(0, np.sqrt(params.R_meas), (params.ny, 1))

        x_upd, P_upd = ekf.measurement_update(P_pred, x_pred, z)
        x_est_hist[t, :] = x_upd.flatten()
        P_hist[t] = P_upd

        x_pred, P_pred = ekf.time_update(P_upd, x_upd)

        w_true = np.array([
            [np.random.normal(0, np.sqrt(params.Q_long))],
            [np.random.normal(0, np.sqrt(params.Q_lat))]
        ])
        x_true = process_model.propagate(x_true, w_true)
        x_true_hist[t + 1, :] = x_true.flatten()

    return params, track, meas_model, x_true_hist, x_est_hist, P_hist


if __name__ == "__main__":
    params, track, meas_model, x_true_hist, x_est_hist, P_hist = run_simulation()

    s_rmse = np.sqrt(np.mean((x_true_hist[1:, 0] - x_est_hist[:, 0])**2))
    e_rmse = np.sqrt(np.mean((x_true_hist[1:, 2] - x_est_hist[:, 2])**2))

    print("--- Step 3 on Coursework Track ---")
    print(f"Longitudinal RMSE : {s_rmse:.4f} m")
    print(f"Lateral RMSE      : {e_rmse:.4f} m")