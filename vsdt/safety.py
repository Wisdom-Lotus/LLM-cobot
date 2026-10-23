"""Transient-contact safety model linking joint stiffness to ISO/TS 15066.

Body-model values: ISO/TS 15066:2016, Annex A, Tables A.2 (maximum
permissible quasi-static force; transient limit = 2x) and A.3 (effective
spring constant, effective mass).  Skull/forehead and face: transient
contact is not permitted, so the verifier treats them as keep-out zones.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# region: (F_quasi_static [N], transient multiplier or None, k [N/m], m_H [kg])
ISO_BODY = {
    "skull":      (130.0, None, 150e3, 4.4),
    "face":       (65.0,  None, 75e3,  4.4),
    "neck":       (150.0, 2.0,  50e3,  1.2),
    "back":       (210.0, 2.0,  35e3,  40.0),
    "chest":      (140.0, 2.0,  25e3,  40.0),
    "abdomen":    (110.0, 2.0,  10e3,  40.0),
    "pelvis":     (180.0, 2.0,  25e3,  40.0),
    "upper_arm":  (150.0, 2.0,  30e3,  3.0),
    "lower_arm":  (160.0, 2.0,  40e3,  2.0),
    "hand":       (140.0, 2.0,  75e3,  0.6),
    "thigh":      (220.0, 2.0,  50e3,  75.0),
    "lower_leg":  (130.0, 2.0,  60e3,  75.0),
}


def transient_limit(region: str) -> float | None:
    F, mult, _, _ = ISO_BODY[region]
    return None if mult is None else F * mult


def iso_vmax(region: str, m_R: float) -> float:
    """ISO/TS 15066 Eq. (A.6): v_rel,max = F_max / sqrt(mu k)."""
    F = transient_limit(region)
    if F is None:
        return 0.0
    _, _, k, mH = ISO_BODY[region]
    mu = 1.0 / (1.0 / m_R + 1.0 / mH)
    return F / np.sqrt(mu * k)


@dataclass
class ImpactModel:
    """1-D reduced model along the contact normal u:

        m_m --[K_c, c_c]-- m_l --(k_H, unilateral)-- m_H

    m_m : motor-side (reflected rotor) Cartesian mass
    m_l : link-side Cartesian mass (incl. payload)
    K_c : Cartesian projection of the joint stiffness
    The system is linear while in contact, so the peak force scales
    linearly with the pre-impact speed v; ``gain`` returns F_peak / v.
    """
    zeta: float = 0.05   # damping ratio of the elastic transmission

    horizon: float = 0.6     # s, max simulated time after first contact
    _cache: dict | None = None

    def gain(self, m_m: float, m_l: float, K_c: float, k_H: float, m_H: float,
             first_phase_only: bool = False) -> float:
        """Peak contact force per unit impact speed (N per m/s).

        Exact piecewise-linear simulation with unilateral contact: the
        state transition matrices of the 'contact' and 'free' modes are
        computed once (matrix exponential) and the trajectory is stepped
        with a time step of 1/400 of the contact period.  Secondary
        impacts (the elastic joint re-accelerating the link into the
        human) are captured.  The system is positively homogeneous, so the
        gain is independent of the impact speed."""
        from scipy.linalg import expm
        key = (round(m_m, 4), round(m_l, 4), round(np.log10(K_c), 4), k_H, m_H, first_phase_only)
        if self._cache is None:
            self._cache = {}
        if key in self._cache:
            return self._cache[key]
        K_c = min(K_c, 1e9)
        c_c = 2 * self.zeta * np.sqrt(K_c * m_m * m_l / (m_m + m_l))

        def Amat(contact: bool):
            kh = k_H if contact else 0.0
            A = np.zeros((6, 6))
            A[0:3, 3:6] = np.eye(3)
            A[3, 0] = -K_c / m_m; A[3, 1] = K_c / m_m; A[3, 3] = -c_c / m_m; A[3, 4] = c_c / m_m
            A[4, 0] = K_c / m_l;  A[4, 1] = -(K_c + kh) / m_l; A[4, 2] = kh / m_l
            A[4, 3] = c_c / m_l;  A[4, 4] = -c_c / m_l
            A[5, 1] = kh / m_H;   A[5, 2] = -kh / m_H
            return A
        mu = m_l * m_H / (m_l + m_H)
        Tc = np.pi * np.sqrt(mu / k_H)
        Tj = 2 * np.pi * np.sqrt(m_m * m_l / (m_m + m_l) / K_c)
        if Tj < Tc / 50:   # transmission effectively rigid: single impact
            g = self.rigid_gain(m_m + m_l, k_H, m_H)
            self._cache[key] = g
            return g
        dt = min(Tc, Tj) / 400
        horizon = min(max(self.horizon, 3 * Tj), 8.0)
        dt_big = max(dt, min(Tj / 60, Tc / 4))
        Pc, Pf = expm(Amat(True) * dt), expm(Amat(False) * dt)
        Pf_big = expm(Amat(False) * dt_big)
        x = np.array([0, 0, 0, 1.0, 1.0, 0.0])
        Fmax, t, was_contact = 0.0, 0.0, True
        mr = m_m * m_l / (m_m + m_l)
        while t < horizon:
            pen = x[1] - x[2]
            contact = pen > 0 or t == 0.0
            if contact:
                Fmax = max(Fmax, k_H * pen)
                x = Pc @ x
                t += dt
                was_contact = True
                continue
            if was_contact and first_phase_only:
                break
            # termination: link can never catch up with the human again
            vcm = (m_m * x[3] + m_l * x[4]) / (m_m + m_l)
            Erel = 0.5 * mr * (x[3] - x[4]) ** 2 + 0.5 * K_c * (x[0] - x[1]) ** 2
            vosc = np.sqrt(2 * Erel / mr) * m_m / (m_m + m_l)
            if x[5] >= vcm + vosc:
                break
            # free flight: large exact steps while the gap stays open with
            # margin; fall back to small steps close to re-contact
            xb = Pf_big @ x
            gap_now, gap_next = x[2] - x[1], xb[2] - xb[1]
            closing = max(x[4] - x[5], xb[4] - xb[5], 0.0)
            if gap_next > 0 and gap_now > 2 * closing * dt_big:
                x = xb
                t += dt_big
            else:
                x = Pf @ x
                t += dt
            was_contact = False
        self._cache[key] = Fmax
        return Fmax

    def rigid_gain(self, m_R: float, k_H: float, m_H: float) -> float:
        mu = m_R * m_H / (m_R + m_H)
        return float(np.sqrt(mu * k_H))


def cartesian_quantities(arm, q, point, frame_idx, u, K_q, Ts=None, Ml_inv=None, Mr_inv=None):
    """Project joint-space quantities on the unit direction u at ``point``.

    Returns (m_l, m_m, K_c): link-side mass, motor-side mass, stiffness.
    m_m is defined so that m_l + m_m equals the rigid-robot effective mass
    computed with reflected rotor inertia (Khatib's operational-space
    inertia along u)."""
    Jp = arm.jacobian(q, Ts=Ts, point=point, upto=frame_idx)[:3]
    a = Jp.T @ u
    if np.linalg.norm(a) < 1e-9:
        return np.inf, np.inf, np.inf
    if Ml_inv is None:
        Ml_inv = np.linalg.inv(arm.mass_matrix(q, Ts=Ts))
    if Mr_inv is None:
        Mr_inv = np.linalg.inv(arm.mass_matrix(q, include_rotor=True, Ts=Ts))
    m_l = 1.0 / float(a @ Ml_inv @ a)
    m_rig = 1.0 / float(a @ Mr_inv @ a)
    m_m = max(m_rig - m_l, 1e-3)
    K_c = 1.0 / float(a @ (a / np.asarray(K_q)))
    return m_l, m_m, K_c


def vmax_contact(region: str, m_l, m_m, K_c, model: ImpactModel | None = None) -> float:
    """Largest admissible pre-impact speed (m/s) along u for the compliant
    robot so that the peak transient force stays below the ISO limit."""
    F = transient_limit(region)
    if F is None:
        return 0.0
    _, _, k, mH = ISO_BODY[region]
    model = model or ImpactModel()
    return F / model.gain(m_m, m_l, K_c, k, mH)


# ------------------------------------------------------------ clamping
@dataclass
class ClampModel:
    """Constrained (clamping) contact: the body part is backed by a fixed
    surface (m_H -> infinity).  The motor side keeps moving at v for the
    detection delay t_d (collision detected from the spring torque), then
    brakes at a_b and holds position (no retraction, conservative).

    Two ISO/TS 15066 checks:
      peak force in the first 0.5 s  <= transient limit (2 x quasi-static)
      residual clamping force        <= quasi-static limit
    The residual force is K_series * s with s = v t_d + v^2/(2 a_b) the
    motor-side over-travel and K_series = (1/K_c + 1/k_H)^-1, so a softer
    joint directly lowers it.
    """
    t_d: float = 0.03       # s, detection + reaction delay
    a_b: float = 3.0        # m/s^2, Cartesian braking deceleration
    zeta: float = 0.05

    def forces(self, v, m_l, K_c, k_H, T=0.5):
        """Return (peak force over T, residual clamping force)."""
        from scipy.linalg import expm
        K_c = min(K_c, 1e9)
        c = 2 * self.zeta * np.sqrt(K_c * m_l)
        s_tot = v * self.t_d + v * v / (2 * self.a_b)
        t_stop = self.t_d + v / self.a_b
        Ks = K_c * k_H / (K_c + k_H)
        F_res = Ks * s_tot
        # link dynamics with prescribed motor motion x_m(t):
        # m_l x'' = K_c (x_m - x) + c (v_m - x') - k_H max(x, 0)
        Tc = np.pi * np.sqrt(m_l / (K_c + k_H))
        dt = min(Tc / 60, 1e-3)
        A = np.array([[0, 1], [-(K_c + k_H) / m_l, -c / m_l]])
        B = np.array([[0, 0], [K_c / m_l, c / m_l]])
        n = int(T / dt) + 1
        # exact discretisation with zero-order hold on the input [x_m, v_m]
        Mbig = np.zeros((4, 4)); Mbig[:2, :2] = A; Mbig[:2, 2:] = B
        E = expm(Mbig * dt)
        Ad, Bd = E[:2, :2], E[:2, 2:]
        x = np.array([0.0, v])
        Fmax = 0.0
        for i in range(n):
            t = i * dt
            if t < self.t_d:
                xm, vm = v * t, v
            elif t < t_stop:
                tb = t - self.t_d
                xm, vm = v * self.t_d + v * tb - 0.5 * self.a_b * tb * tb, v - self.a_b * tb
            else:
                xm, vm = s_tot, 0.0
            Fmax = max(Fmax, k_H * max(x[0], 0.0))
            x = Ad @ x + Bd @ np.array([xm, vm])
            # unilateral wall: if the link separates the contact spring is inactive;
            # with the motor pushing forward this does not occur after first contact
        return Fmax, F_res

    def vmax(self, region, m_l, K_c, v_hi=2.0, iters=22):
        Fqs, mult, k_H, _ = ISO_BODY[region]
        if mult is None:
            return 0.0
        FT = Fqs * mult
        lo, hi = 0.0, v_hi
        for _ in range(iters):
            v = 0.5 * (lo + hi)
            Fp, Fr = self.forces(v, m_l, K_c, k_H)
            if Fp <= FT and Fr <= Fqs:
                lo = v
            else:
                hi = v
        return lo
