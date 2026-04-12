import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# PARAMETERS
# ============================================================

class EstimatorParams:
    def __init__(self):
        self.dt = 0.01
        self.v_target = 10.0
        self.k_vel = 0.1
        self.k_damp = 1.0
        self.k_restore = 0.5

        self.lane_half_width = 2.0
        self.true_bias = 1.5
        self.r_meas_std = 1.5

        self.R_left = 48.0
        self.R_right = 22.0
        self.d_centers = 100.0

        self.x0 = np.array([[0.0],[10.0],[0.0],[0.0],[0.0]])
        self.P0 = np.diag([1.0,0.5,0.5,0.1,2.0])

        self.Q = np.diag([0.001,0.01,0.001,0.01,1e-6])
        self.R = (self.r_meas_std**2) * np.eye(4)

        self.beacons = np.array([
            [-60,  60],
            [-60, -60],
            [130,  60],  # biased
            [130, -60]
        ])

# ============================================================
# TRACK GEOMETRY
# ============================================================

class TrackGeometry:
    def __init__(self, p):
        self.R1 = p.R_left
        self.R2 = p.R_right
        self.d = p.d_centers

        self.alpha = np.arcsin((self.R1 - self.R2)/self.d)

        self.L_str = self.d * np.cos(self.alpha)
        self.L_curveB = self.R2 * (np.pi - 2*self.alpha)
        self.L_curveA = self.R1 * (np.pi + 2*self.alpha)
        self.L_total = 2*self.L_str + self.L_curveB + self.L_curveA

    def wrap(self, s):
        return np.mod(s, self.L_total)

# ============================================================
# MEASUREMENT MODEL (CasADi)
# ============================================================

class CasadiMeasurementModel:
    def __init__(self, p, track):
        self.p = p
        self.track = track
        self.ny = p.beacons.shape[0]

        s = cs.MX.sym('s')
        vs = cs.MX.sym('vs')
        e = cs.MX.sym('e')
        ve = cs.MX.sym('ve')
        b = cs.MX.sym('b')

        x = cs.vertcat(s,vs,e,ve,b)

        s_mod = cs.fmod(s, track.L_total)

        # simple circular approx for now
        theta = s_mod / p.R_left
        X = (p.R_left + e) * cs.cos(theta)
        Y = (p.R_left + e) * cs.sin(theta)

        h_list = []
        for i in range(self.ny):
            bx, by = p.beacons[i]
            dist = cs.sqrt((X-bx)**2 + (Y-by)**2)

            if i == 2:
                dist = dist + b

            h_list.append(dist)

        h = cs.vertcat(*h_list)

        self.h_func = cs.Function('h', [x], [h])
        self.H_func = cs.Function('H', [x], [cs.jacobian(h,x)])

    def measurement(self, x):
        return np.array(self.h_func(x)).astype(float)

    def jacobian(self, x):
        return np.array(self.H_func(x)).astype(float)

# ============================================================
# PROCESS MODEL
# ============================================================

class ProcessModel:
    def __init__(self, p):
        self.p = p

    def f(self, x):
        s,v_s,e,v_e,b = x.flatten()
        dt = self.p.dt

        return np.array([
            [s + v_s*dt],
            [v_s + self.p.k_vel*(self.p.v_target - v_s)*dt],
            [e + v_e*dt],
            [v_e - (self.p.k_damp*v_e + self.p.k_restore*e)*dt],
            [b]
        ])

    def A(self):
        dt = self.p.dt
        A = np.eye(5)
        A[0,1] = dt
        A[1,1] = 1 - self.p.k_vel*dt
        A[2,3] = dt
        A[3,2] = -self.p.k_restore*dt
        A[3,3] = 1 - self.p.k_damp*dt
        return A

# ============================================================
# ENGINEERED EKF
# ============================================================

class EngineeredEKF:
    def __init__(self, p, fm, hm):
        self.p = p
        self.fm = fm
        self.hm = hm

    def predict(self, x, P):
        x_pred = self.fm.f(x)
        A = self.fm.A()
        P_pred = A @ P @ A.T + self.p.Q
        P_pred = 0.5*(P_pred + P_pred.T)
        return x_pred, P_pred

    def update(self, x_pred, P_pred, z):
        H = self.hm.jacobian(x_pred)
        z_hat = self.hm.measurement(x_pred)

        y = z - z_hat

        S = H @ P_pred @ H.T + self.p.R
        S = 0.5*(S + S.T)

        K = P_pred @ H.T @ np.linalg.inv(S)

        x_upd = x_pred + K @ y

        I = np.eye(5)
        P_upd = (I-K@H)@P_pred@(I-K@H).T + K@self.p.R@K.T
        P_upd = 0.5*(P_upd + P_upd.T)

        nis = (y.T @ np.linalg.solve(S,y)).item()

        return x_upd, P_upd, nis

# ============================================================
# TRUE PLANT
# ============================================================

class TruePlant:
    def __init__(self, p, fm):
        self.p = p
        self.fm = fm

    def step(self, x):
        w = np.random.multivariate_normal(np.zeros(5), self.p.Q).reshape(-1,1)
        w[4,0] = 0

        x_next = self.fm.f(x) + w

        x_next[4,0] = self.p.true_bias

        if x_next[2,0] > self.p.lane_half_width:
            x_next[2,0] = self.p.lane_half_width
            x_next[3,0] *= -0.5

        elif x_next[2,0] < -self.p.lane_half_width:
            x_next[2,0] = -self.p.lane_half_width
            x_next[3,0] *= -0.5

        return x_next

# ============================================================
# SIMULATION
# ============================================================

def run_simulation(n=3000):
    p = EstimatorParams()
    track = TrackGeometry(p)
    hm = CasadiMeasurementModel(p, track)
    fm = ProcessModel(p)
    ekf = EngineeredEKF(p, fm, hm)
    plant = TruePlant(p, fm)

    x_true = p.x0.copy()
    x_est = p.x0.copy()
    P = p.P0.copy()

    x_true[4,0] = p.true_bias

    true_hist = np.zeros((n,5))
    est_hist = np.zeros((n,5))
    nis_hist = np.zeros(n)

    for k in range(n):
        x_true = plant.step(x_true)

        z = hm.measurement(x_true) + np.random.normal(0,p.r_meas_std,(4,1))

        x_pred, P_pred = ekf.predict(x_est,P)
        x_est, P, nis = ekf.update(x_pred,P_pred,z)

        true_hist[k] = x_true.flatten()
        est_hist[k] = x_est.flatten()
        nis_hist[k] = nis

    return true_hist, est_hist, nis_hist

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    true, est, nis = run_simulation()

    rmse_s = np.sqrt(np.mean((true[:,0]-est[:,0])**2))
    rmse_e = np.sqrt(np.mean((true[:,2]-est[:,2])**2))

    print("STEP 4 (ENGINEERED EKF)")
    print(f"s RMSE: {rmse_s:.4f}")
    print(f"e RMSE: {rmse_e:.4f}")