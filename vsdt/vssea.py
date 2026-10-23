"""Leaf-spring variable-stiffness mechanism (VSAM) model.

Mechanism (after Sariyildiz et al., IEEE/ASME T-Mech 2023): ``n`` leaf
springs are arranged axially at radius ``r`` around the joint axis and
clamped at one end.  A roller carriage driven by a stiffness motor M2
through a ball screw contacts every spring at axial distance ``x`` from the
clamp.  A relative rotation ``q_d`` between the M1 output and the link
deflects every spring tangentially by ``delta = r * q_d`` at the roller.

Four models of increasing fidelity are provided:
  * ``k_eb``      Euler-Bernoulli cantilever (linear)
  * ``k_timo``    Timoshenko cantilever (linear, bending + shear)
  * ``fe_plane``  independent 2-D plane-stress finite-element model (Q8)
  * ``elastica``  geometrically exact large-deflection beam with a
                  frictionless roller (force normal to the deformed beam)
SI units throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


@dataclass
class VSAMParams:
    n: int = 8                 # number of leaf springs
    E: float = 206e9           # Young's modulus (spring steel)
    nu: float = 0.29
    b: float = 0.020           # spring width (m)
    h: float = 0.002           # spring thickness (m)
    r: float = 0.040           # roller radius from joint axis (m)
    L: float = 0.180           # free spring length (m)
    x_min: float = 0.030       # closest roller position to clamp (m)
    x_max: float = 0.170       # farthest roller position (m)
    sigma_allow: float = 800e6 # allowable bending stress (Pa) (yield/1.6)
    K_hd: float = 3.1e4        # harmonic-drive torsional stiffness (Nm/rad), series
    lead: float = 0.002        # ball-screw lead (m/rev)
    w_m2_max: float = 419.0    # M2 max speed (rad/s) ~ 4000 rpm
    a_m2_max: float = 4000.0   # M2 max acceleration (rad/s^2)
    m_carriage: float = 0.15   # roller carriage mass (kg)
    mu_roll: float = 0.005     # rolling-friction coefficient
    preload: float = 5.0       # roller preload per spring (N)
    screw_eff: float = 0.9     # ball-screw efficiency

    @property
    def G(self) -> float:
        return self.E / (2 * (1 + self.nu))

    @property
    def I(self) -> float:
        return self.b * self.h ** 3 / 12

    @property
    def A(self) -> float:
        return self.b * self.h

    kappa: float = 5.0 / 6.0   # shear coefficient, rectangle


# ---------------------------------------------------------------- linear
def kb_eb(p: VSAMParams, x):
    """Transverse stiffness of one spring at the roller (N/m), Euler-Bernoulli."""
    return 3 * p.E * p.I / np.asarray(x) ** 3


def kb_timo(p: VSAMParams, x):
    x = np.asarray(x)
    return 1.0 / (x ** 3 / (3 * p.E * p.I) + x / (p.kappa * p.G * p.A))


def K_vsam(p: VSAMParams, x, model: str = "timo"):
    """Rotational stiffness of the VSAM (Nm/rad) at roller position x."""
    kb = kb_timo(p, x) if model == "timo" else kb_eb(p, x)
    return p.n * kb * p.r ** 2


def K_joint(p: VSAMParams, x, model: str = "timo"):
    """Joint stiffness: VSAM in series with harmonic-drive compliance."""
    Kv = K_vsam(p, x, model)
    return 1.0 / (1.0 / Kv + 1.0 / p.K_hd)


def x_from_K(p: VSAMParams, K_target: float) -> float:
    """Roller position giving joint stiffness K_target (monotone bisection)."""
    lo, hi = p.x_min, p.x_max
    if K_target >= K_joint(p, lo):
        return lo
    if K_target <= K_joint(p, hi):
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if K_joint(p, mid) > K_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def K_range(p: VSAMParams) -> tuple[float, float]:
    return float(K_joint(p, p.x_max)), float(K_joint(p, p.x_min))


def tau_max_stress(p: VSAMParams, x):
    """Largest transmissible torque before the root bending stress reaches
    sigma_allow (linear theory): sigma = F x (h/2)/I, tau = n F r."""
    F = p.sigma_allow * p.I / (np.asarray(x) * p.h / 2)
    return p.n * F * p.r


def qd_max_stress(p: VSAMParams, x):
    """Allowable spring deflection angle (rad) at roller position x."""
    return tau_max_stress(p, x) / K_vsam(p, x)


# ---------------------------------------------------- plane-stress FE (Q8)
def _q8_shape(xi, eta):
    N = np.array([
        -0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta),
        -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta),
        -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta),
        -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta),
        0.5 * (1 - xi ** 2) * (1 - eta),
        0.5 * (1 + xi) * (1 - eta ** 2),
        0.5 * (1 - xi ** 2) * (1 + eta),
        0.5 * (1 - xi) * (1 - eta ** 2)])
    dxi = np.array([
        0.25 * (1 - eta) * (2 * xi + eta),
        0.25 * (1 - eta) * (2 * xi - eta),
        0.25 * (1 + eta) * (2 * xi + eta),
        0.25 * (1 + eta) * (2 * xi - eta),
        -xi * (1 - eta),
        0.5 * (1 - eta ** 2),
        -xi * (1 + eta),
        -0.5 * (1 - eta ** 2)])
    deta = np.array([
        0.25 * (1 - xi) * (xi + 2 * eta),
        0.25 * (1 + xi) * (-xi + 2 * eta),
        0.25 * (1 + xi) * (xi + 2 * eta),
        0.25 * (1 - xi) * (-xi + 2 * eta),
        -0.5 * (1 - xi ** 2),
        -(1 + xi) * eta,
        0.5 * (1 - xi ** 2),
        -(1 - xi) * eta])
    return N, dxi, deta


def fe_plane(p: VSAMParams, x: float, nx: int = 120, ny: int = 4,
             extend_to_L: bool = False) -> float:
    """Transverse stiffness of one spring (N/m) from a 2-D plane-stress Q8
    model of the strip (length along X, thickness along Y, width b out of
    plane).  Fully clamped at X=0.  A unit transverse load is spread
    parabolically (as a shear traction) over the cross-section at X=x.
    Returns load / mean transverse displacement of that section."""
    Lm = p.L if extend_to_L else x
    # nodes on a (2nx+1) x (2ny+1) grid; Q8 uses all except element centres
    NX, NY = 2 * nx + 1, 2 * ny + 1
    X = np.linspace(0, Lm, NX)
    if extend_to_L:  # make sure x is a grid line
        nx1 = max(2, int(round(nx * x / Lm)))
        X = np.concatenate([np.linspace(0, x, 2 * nx1 + 1),
                            np.linspace(x, Lm, 2 * (nx - nx1) + 1)[1:]])
    Y = np.linspace(-p.h / 2, p.h / 2, NY)
    nid = -np.ones((NX, NY), dtype=int)
    coords = []
    for i in range(NX):
        for j in range(NY):
            if i % 2 == 1 and j % 2 == 1:
                continue
            nid[i, j] = len(coords)
            coords.append((X[i], Y[j]))
    coords = np.array(coords)
    ndof = 2 * len(coords)
    E, nu = p.E, p.nu
    D = E / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]) * p.b
    gp = np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])
    gw = np.array([5 / 9, 8 / 9, 5 / 9])
    K = lil_matrix((ndof, ndof))
    for ei in range(nx):
        for ej in range(ny):
            i0, j0 = 2 * ei, 2 * ej
            conn = [nid[i0, j0], nid[i0 + 2, j0], nid[i0 + 2, j0 + 2], nid[i0, j0 + 2],
                    nid[i0 + 1, j0], nid[i0 + 2, j0 + 1], nid[i0 + 1, j0 + 2], nid[i0, j0 + 1]]
            xy = coords[conn]
            ke = np.zeros((16, 16))
            for a, wa in zip(gp, gw):
                for c, wc in zip(gp, gw):
                    _, dxi, deta = _q8_shape(a, c)
                    Jm = np.array([[dxi @ xy[:, 0], dxi @ xy[:, 1]],
                                   [deta @ xy[:, 0], deta @ xy[:, 1]]])
                    dN = np.linalg.solve(Jm, np.vstack([dxi, deta]))
                    B = np.zeros((3, 16))
                    B[0, 0::2] = dN[0]
                    B[1, 1::2] = dN[1]
                    B[2, 0::2] = dN[1]
                    B[2, 1::2] = dN[0]
                    ke += B.T @ D @ B * np.linalg.det(Jm) * wa * wc
            dofs = np.ravel([[2 * n, 2 * n + 1] for n in conn])
            for r_, gr in enumerate(dofs):
                for c_, gc in enumerate(dofs):
                    K[gr, gc] += ke[r_, c_]
    f = np.zeros(ndof)
    ix = int(np.argmin(np.abs(X - x)))
    sec = [nid[ix, j] for j in range(NY)]
    # consistent nodal loads of a parabolic shear traction (unit resultant)
    ys = Y
    w = 1.5 / p.h * (1 - (2 * ys / p.h) ** 2)  # traction density, integrates to 1
    # Simpson-like lumping along the section using quadratic edges
    lump = np.zeros(NY)
    for e in range(ny):
        yy = ys[2 * e:2 * e + 3]
        lump[2 * e:2 * e + 3] += (yy[2] - yy[0]) / 6 * np.array([1, 4, 1]) * w[2 * e:2 * e + 3]
    lump /= lump.sum()
    for n_, l_ in zip(sec, lump):
        f[2 * n_ + 1] += l_
    fixed = []
    for j in range(NY):
        n_ = nid[0, j]
        fixed += [2 * n_, 2 * n_ + 1]
    free = np.setdiff1d(np.arange(ndof), fixed)
    Kc = K.tocsr()[free][:, free]
    u = np.zeros(ndof)
    u[free] = spsolve(Kc, f[free])
    v = np.array([u[2 * n_ + 1] for n_ in sec])
    return 1.0 / float(lump @ v)


# ----------------------------------------------------- large deflection
def elastica(p: VSAMParams, x: float, delta: float, guess=None):
    """Geometrically exact clamped beam pushed by a frictionless roller
    located at axial coordinate X = x, transverse position Y = delta.

    The contact force acts normal to the deformed beam at the contact
    point (arc length s_c).  Unknowns: root moment M0, force magnitude P,
    contact arc length s_c, beam angle at contact phi_c.
    Returns dict(P, phi_c, s_c, F_tan = P cos phi_c, F_axial = P sin phi_c,
    sigma_root).
    """
    EI = p.E * p.I

    def shoot(z):
        M0, P, sc, phic = z
        fx, fy = -P * np.sin(phic), P * np.cos(phic)

        def rhs(s, y):
            X_, Y_, ph, M = y
            return [np.cos(ph), np.sin(ph), M / EI, -np.cos(ph) * fy + np.sin(ph) * fx]
        sol = solve_ivp(rhs, (0, sc), [0, 0, 0, M0], rtol=1e-10, atol=1e-12)
        X_, Y_, ph, M = sol.y[:, -1]
        return [M / (abs(M0) + 1e-9), (X_ - x) / x, (Y_ - delta) / x, ph - phic], (M0, P, sc, phic)

    if guess is None:
        P0 = 3 * EI * delta / x ** 3
        guess = [P0 * x, P0, x * (1 + 0.5 * (1.5 * delta / x) ** 2 / 3), 1.5 * delta / x]
    z, info, ier, msg = fsolve(lambda z: shoot(z)[0], guess, full_output=True, xtol=1e-11)
    res = shoot(z)[0]
    M0, P, sc, phic = z
    return dict(M0=M0, P=P, s_c=sc, phi_c=phic, F_tan=P * np.cos(phic),
                F_axial=P * np.sin(phic), sigma_root=abs(M0) * (p.h / 2) / p.I,
                ok=bool(ier == 1 and np.max(np.abs(res)) < 1e-6), z=z)


def torque_curve_elastica(p: VSAMParams, x: float, qd: np.ndarray):
    """Nonlinear torque-deflection curve from the elastica model.
    Returns arrays tau (Nm), axial carriage force (N), root stress (Pa)."""
    tau, fax, sig = [], [], []
    guess = None
    for q in qd:
        if abs(q) < 1e-9:
            tau.append(0.0); fax.append(0.0); sig.append(0.0); guess = None
            continue
        out = elastica(p, x, p.r * q, guess)
        guess = out["z"]
        tau.append(p.n * out["F_tan"] * p.r)
        fax.append(p.n * out["F_axial"])
        sig.append(out["sigma_root"])
    return np.array(tau), np.array(fax), np.array(sig)


# ------------------------------------------------- stiffness modulation
def modulation_time(p: VSAMParams, x0: float, x1: float) -> float:
    """Minimum time to move the roller carriage from x0 to x1 with a
    trapezoidal M2 speed profile."""
    eta = p.lead / (2 * np.pi)
    dist = abs(x1 - x0) / eta          # M2 rotation (rad)
    w, a = p.w_m2_max, p.a_m2_max
    if dist <= w ** 2 / a:
        return 2 * np.sqrt(dist / a)
    return dist / w + w / a


def modulation_energy(p: VSAMParams, x0: float, x1: float, qd: float = 0.0,
                      npts: int = 40) -> float:
    """Mechanical work (J) delivered by M2 to move the carriage from x0 to
    x1 while the joint holds deflection qd.  Includes rolling friction
    (preload + spring load) and the axial spring reaction from the
    elastica model; kinetic energy of the carriage is assumed recovered
    or dissipated in braking, so it is excluded."""
    xs = np.linspace(x0, x1, npts)
    F = []
    for x in xs:
        if abs(qd) > 1e-9:
            e = elastica(p, x, p.r * qd)
            Fn = p.n * e["P"]
            Fa = p.n * e["F_axial"]
        else:
            Fn, Fa = 0.0, 0.0
        Ffr = p.mu_roll * (p.n * p.preload + Fn)
        # the axial reaction pushes the carriage toward the free end
        # (softening); moving toward the clamp works against it
        F.append(Ffr + Fa * np.sign(x0 - x1))
    F = np.array(F)
    W = np.trapezoid(np.maximum(F, 0), xs) * np.sign(x1 - x0)
    return abs(W) / p.screw_eff


# ------------------------------------------- capacity from the elastica
def tau_max_elastica(p: VSAMParams, x: float, tol: float = 1e-4) -> float:
    """Torque at which the root stress of the geometrically exact model
    reaches sigma_allow (bisection on the deflection angle)."""
    q_lo, q_hi = 0.0, float(qd_max_stress(p, x)) * 1.6
    guess = None
    for _ in range(40):
        q = 0.5 * (q_lo + q_hi)
        e = elastica(p, x, p.r * q, guess)
        if not e["ok"]:
            q_hi = q
            continue
        guess = e["z"]
        if e["sigma_root"] > p.sigma_allow:
            q_hi = q
        else:
            q_lo = q
        if q_hi - q_lo < tol * q_hi:
            break
    e = elastica(p, x, p.r * q_lo)
    return p.n * e["F_tan"] * p.r


_CAP_TABLE = {}


def tau_cap_table(p: VSAMParams, n: int = 40):
    """Interpolation table x -> min(linear, elastica) stress-limited torque."""
    key = (p.n, p.E, p.b, p.h, p.r, p.x_min, p.x_max, p.sigma_allow)
    if key not in _CAP_TABLE:
        xs = np.linspace(p.x_min, p.x_max, n)
        cap = np.array([min(float(tau_max_stress(p, x)), tau_max_elastica(p, x)) for x in xs])
        _CAP_TABLE[key] = (xs, cap)
    return _CAP_TABLE[key]


def tau_capacity(p: VSAMParams, x):
    xs, cap = tau_cap_table(p)
    # capacity decreases with x; linear interpolation then 0.5 % margin
    return np.interp(x, xs, cap) * 0.995
