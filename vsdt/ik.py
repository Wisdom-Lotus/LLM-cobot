"""Damped-least-squares inverse kinematics.

Task: flange position p* and tool axis z* (5 constraints; the rotation
about the tool axis is free).  Two Jacobian variants are implemented so
that the original manuscript's finite-difference scheme can be compared
with the analytic geometric Jacobian.
"""
from __future__ import annotations

import numpy as np

from .robot import Arm, cross


def task_error(arm: Arm, q, p_star, z_star, w_orient=0.5):
    T = arm.fk(q)
    ep = p_star - T[:3, 3]
    z = T[:3, 2]
    eo = cross(z, z_star)            # small-angle axis error
    return np.concatenate([ep, w_orient * eo]), T


def _jac_analytic(arm, q, z_star, w_orient, Ts=None):
    Ts = arm.frames_fk(q) if Ts is None else Ts
    J = arm.jacobian(q, Ts)
    z = Ts[-1][:3, 2]
    # error e_o = z x z*;  de_o/dq_j = (omega_j x z) x z*.  The DLS step
    # uses the Jacobian of the task map f (e = target - f), hence the sign.
    Jz = cross(J[3:].T, z).T                 # 3x6, columns omega_j x z
    Jo = -cross(Jz.T, z_star).T
    return np.vstack([J[:3], w_orient * Jo])


def _jac_fd(arm, q, p_star, z_star, w_orient, eps):
    e0, _ = task_error(arm, q, p_star, z_star, w_orient)
    J = np.zeros((6, 6))
    for j in range(6):
        dq = np.zeros(6); dq[j] = eps
        e1, _ = task_error(arm, q + dq, p_star, z_star, w_orient)
        J[:, j] = -(e1 - e0) / eps          # Jacobian of f(q), error = target - f
    return J


def solve(arm: Arm, p_star, z_star=np.array([0, 0, -1.0]), q0=None, *,
          jac: str = "analytic", lam: float = 0.1, max_iter: int = 500,
          tol: float = 1e-5, w_orient: float = 0.5, eps_fd: float = 1e-7,
          max_step: float = 0.3, restarts: int = 8, rng=None):
    """Returns (q, converged, iterations, final_error_norm)."""
    rng = np.random.default_rng(0) if rng is None else rng
    p_star = np.asarray(p_star, float)
    z_star = np.asarray(z_star, float) / np.linalg.norm(z_star)
    seeds = [np.zeros(6) if q0 is None else np.asarray(q0, float)]
    seeds += [rng.uniform(arm.q_min, arm.q_max) for _ in range(restarts)]
    best = None
    total_it = 0
    for qs in seeds:
        q = qs.copy()
        for it in range(max_iter):
            Ts = arm.frames_fk(q)
            T = Ts[-1]
            e = np.concatenate([p_star - T[:3, 3], w_orient * cross(T[:3, 2], z_star)])
            en = np.linalg.norm(e)
            if en < tol:
                break
            if jac == "analytic":
                J = _jac_analytic(arm, q, z_star, w_orient, Ts)
            else:
                J = _jac_fd(arm, q, p_star, z_star, w_orient, eps_fd)
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
            s = np.max(np.abs(dq))
            if s > max_step:
                dq *= max_step / s
            q = q + dq
            q = np.clip(q, arm.q_min, arm.q_max)
        total_it += it + 1
        e, _ = task_error(arm, q, p_star, z_star, w_orient)
        en = np.linalg.norm(e)
        if best is None or en < best[1]:
            best = (q.copy(), en)
        if en < tol:
            return q, True, total_it, en
    return best[0], False, total_it, best[1]


def solve_all(arm: Arm, p_star, z_star=np.array([0, 0, -1.0]), seeds=(), tol=1e-5, **kw):
    """Run DLS from each seed (no random restarts) and return the distinct
    converged solutions."""
    sols = []
    for s in seeds:
        q, ok, _, _ = solve(arm, p_star, z_star, q0=s, restarts=0, tol=tol, max_iter=kw.pop("max_iter", 200), **kw)
        if ok and not any(np.max(np.abs(q - x)) < 1e-3 for x in sols):
            sols.append(q)
    return sols
