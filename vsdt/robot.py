"""Kinematics and rigid-body model of the 6-actuated-joint arm.

The kinematic chain reproduces Table III of the original manuscript
(9 DH frames, 6 actuated joints, all a_i = 0).  Only the joint
variables theta_1..theta_6 are actuated; the remaining frames are
fixed offsets.  Angles in radians, lengths in metres.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

G = np.array([0.0, 0.0, -9.81])


def cross(a, b):
    """Fast cross product along the last axis (np.cross is slow for small arrays)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim == 1 and b.ndim == 1:
        return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])
    a, b = np.broadcast_arrays(a, b)
    out = np.empty(a.shape)
    out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return out


def dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


@dataclass
class Frame:
    d: float
    alpha: float
    theta0: float
    joint: int | None  # index of actuated joint (0..5) or None if fixed
    a: float = 0.0


def paper_frames(scale: float = 1.0) -> list[Frame]:
    """DH frames exactly as in the original manuscript (Table III).

    ``scale`` multiplies all link offsets (used for the re-dimensioned arm).
    """
    d = np.array([0.150, 0.240, 0.170, 0.240, 0.760, 0.410, 0.760, 0.410, 0.180]) * scale
    r = np.deg2rad
    return [
        Frame(d[0], 0.0, r(90), 0),
        Frame(d[1], r(90), 0.0, None),
        Frame(d[2], 0.0, 0.0, None),
        Frame(d[3], r(90), 0.0, 1),
        Frame(d[4], r(-90), 0.0, 2),
        Frame(d[5], r(-90), 0.0, 3),
        Frame(d[6], r(90), 0.0, 4),
        Frame(d[7], r(90), 0.0, 5),
        Frame(d[8], 0.0, 0.0, None),
    ]


@dataclass
class Arm:
    frames: list[Frame]
    q_min: np.ndarray = field(default_factory=lambda: np.full(6, -np.pi))
    q_max: np.ndarray = field(default_factory=lambda: np.full(6, np.pi))
    actuator_mass: float = 2.8          # kg per joint module (housing+motors+VSAM)
    link_lin_density: float = 1.2       # kg/m, aluminium tube links
    payload: float = 2.5                # kg
    link_radius: float = 0.045          # collision capsule radius (m)
    rotor_inertia: float = 1.2e-4       # kg m^2, M1 rotor
    gear_ratio: float = 100.0
    q_home: np.ndarray = field(default_factory=lambda: np.array([0.0, 2.997, 2.682, 2.682, -0.563, 0.224]))

    n: int = 6

    # ------------------------------------------------------------------ FK
    def frames_fk(self, q: np.ndarray) -> list[np.ndarray]:
        """Return list of 10 homogeneous transforms T_0..T_9 (T_0 = I).
        A_i = Rz(theta_i) * C_i with the constant C_i = Tz(d_i) Rx(alpha_i)."""
        if not hasattr(self, "_C"):
            self._C = [dh(0.0, f.d, f.a, f.alpha) for f in self.frames]
        T = np.eye(4)
        out = [T]
        Rz = np.eye(4)
        for f, C in zip(self.frames, self._C):
            th = f.theta0 + (q[f.joint] if f.joint is not None else 0.0)
            c, s = np.cos(th), np.sin(th)
            Rz[0, 0] = c; Rz[0, 1] = -s; Rz[1, 0] = s; Rz[1, 1] = c
            T = T @ (Rz @ C)
            out.append(T)
        return out

    def fk(self, q: np.ndarray) -> np.ndarray:
        return self.frames_fk(q)[-1]

    def joint_frame_index(self) -> list[int]:
        """Index i such that joint j rotates about z of T_i (T before the frame)."""
        idx = [0] * 6
        for i, f in enumerate(self.frames):
            if f.joint is not None:
                idx[f.joint] = i
        return idx

    # ------------------------------------------------------------ Jacobian
    def jacobian(self, q: np.ndarray, Ts: list[np.ndarray] | None = None,
                 point: np.ndarray | None = None, upto: int | None = None) -> np.ndarray:
        """Analytic geometric Jacobian (6x6) of ``point`` (default: flange).

        ``upto``: only joints whose frame index < upto contribute (for
        points on intermediate links).
        """
        Ts = self.frames_fk(q) if Ts is None else Ts
        p = Ts[-1][:3, 3] if point is None else point
        J = np.zeros((6, 6))
        for j, i in enumerate(self.joint_frame_index()):
            if upto is not None and i >= upto:
                continue
            z = Ts[i][:3, 2]
            o = Ts[i][:3, 3]
            J[:3, j] = cross(z, p - o)
            J[3:, j] = z
        return J

    # ------------------------------------------------------ link geometry
    def segments(self, q: np.ndarray, Ts=None) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """Straight link segments between consecutive frame origins.

        Returns (p_start, p_end, frame_index_of_end)."""
        Ts = self.frames_fk(q) if Ts is None else Ts
        segs = []
        for i in range(1, len(Ts)):
            a, b = Ts[i - 1][:3, 3], Ts[i][:3, 3]
            if np.linalg.norm(b - a) > 1e-9:
                segs.append((a, b, i))
        return segs

    # ------------------------------------------------------- mass model
    def point_masses(self, q: np.ndarray, Ts=None, payload: float | None = None):
        """Lumped mass model: each segment's mass at its midpoint, an actuator
        module at every actuated joint after the base joint, and the payload
        at the flange.  Returns list of (mass, position, frame_index)."""
        Ts = self.frames_fk(q) if Ts is None else Ts
        payload = self.payload if payload is None else payload
        pm = []
        for a, b, i in self.segments(q, Ts):
            pm.append((self.link_lin_density * np.linalg.norm(b - a), 0.5 * (a + b), i))
        for j, i in enumerate(self.joint_frame_index()):
            if j == 0:
                continue  # base actuator does not move with the arm
            pm.append((self.actuator_mass, Ts[i][:3, 3], i))
        pm.append((payload, Ts[-1][:3, 3], len(Ts) - 1))
        return pm

    def _mass_points(self, q, Ts, payload):
        """Arrays (m[k], p[k,3], upto[k]) of the lumped-mass model."""
        payload = self.payload if payload is None else payload
        O = np.array([T[:3, 3] for T in Ts])
        ms, ps, up = [], [], []
        for i in range(1, len(Ts)):
            a, b = O[i - 1], O[i]
            L = np.linalg.norm(b - a)
            if L > 1e-9:
                ms.append(self.link_lin_density * L); ps.append(0.5 * (a + b)); up.append(i)
        for j, i in enumerate(self.joint_frame_index()):
            if j == 0:
                continue
            ms.append(self.actuator_mass); ps.append(O[i]); up.append(i)
        ms.append(payload); ps.append(O[-1]); up.append(len(Ts) - 1)
        return np.array(ms), np.array(ps), np.array(up)

    def _axes(self, Ts):
        jidx = self.joint_frame_index()
        Z = np.array([Ts[i][:3, 2] for i in jidx])
        O = np.array([Ts[i][:3, 3] for i in jidx])
        return np.array(jidx), Z, O

    def gravity_torque(self, q: np.ndarray, payload: float | None = None, Ts=None) -> np.ndarray:
        """Joint torques needed to hold the arm against gravity (vectorised)."""
        Ts = self.frames_fk(q) if Ts is None else Ts
        m, p, up = self._mass_points(q, Ts, payload)
        jidx, Z, O = self._axes(Ts)
        F = -m[:, None] * G[None, :]                   # holding force at each mass
        r = p[None, :, :] - O[:, None, :]              # [6, k, 3]
        mom = cross(r, F[None, :, :])               # [6, k, 3]
        tz = np.einsum("jkc,jc->jk", mom, Z)
        mask = jidx[:, None] < up[None, :]
        return (tz * mask).sum(1)

    def mass_matrix(self, q: np.ndarray, payload: float | None = None,
                    include_rotor: bool = False, Ts=None) -> np.ndarray:
        """Link-side joint-space inertia of the lumped-mass model (point
        masses; 0.01 kg m^2 added per joint for module rotational inertia).
        ``include_rotor`` adds the reflected rotor inertia N^2 J_m."""
        Ts = self.frames_fk(q) if Ts is None else Ts
        m, p, up = self._mass_points(q, Ts, payload)
        jidx, Z, O = self._axes(Ts)
        r = p[None, :, :] - O[:, None, :]
        Jv = cross(Z[:, None, :], r)                # [6, k, 3] column j for mass k
        Jv = Jv * (jidx[:, None] < up[None, :])[:, :, None]
        M = np.einsum("k,jkc,ikc->ji", m, Jv, Jv)
        M += np.eye(6) * 0.01
        if include_rotor:
            M += np.eye(6) * self.rotor_inertia * self.gear_ratio ** 2
        return M


def paper_arm() -> Arm:
    return Arm(paper_frames(1.0))


def resized_arm(scale: float = 0.4) -> Arm:
    """Same kinematic topology, link offsets scaled."""
    return Arm(paper_frames(scale))


def ur5e_frames() -> list[Frame]:
    """Standard DH of the UR5e kinematic structure (Universal Robots,
    'DH parameters for calculations of kinematics and dynamics')."""
    h = np.pi / 2
    return [Frame(0.1625, h, 0.0, 0),
            Frame(0.0, 0.0, 0.0, 1, a=-0.425),
            Frame(0.0, 0.0, 0.0, 2, a=-0.3922),
            Frame(0.1333, h, 0.0, 3),
            Frame(0.0997, -h, 0.0, 4),
            Frame(0.0996, 0.0, 0.0, 5),
            Frame(0.15, 0.0, 0.0, None)]          # gripper: TCP 0.15 m beyond the flange


def ur5e_class_arm() -> Arm:
    """VSSEA modules on a UR5e-class kinematic structure (the arm of the
    revised study).  Mass model: 2.8 kg module per moving joint, 1.2 kg/m
    links, 2.5 kg payload; tool frame = flange."""
    a = Arm(ur5e_frames())
    a.link_radius = 0.045
    a.q_home = np.array([0.0, -1.9, 1.9, -1.571, -1.571, 0.0])
    a.q_min, a.q_max = np.full(6, -2 * np.pi), np.full(6, 2 * np.pi)   # +-360 deg, as UR
    return a
