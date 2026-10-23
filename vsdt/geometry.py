"""Exact distance queries between capsules and axis-aligned boxes."""
from __future__ import annotations

import numpy as np


def closest_seg_seg(p1, q1, p2, q2):
    """Closest points between segments [p1,q1] and [p2,q2] (Ericson 2004,
    Sec. 5.1.9).  Returns (dist, c1, c2)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(r)), p1, p2
    if a <= eps:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e <= eps:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > eps else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    c1, c2 = p1 + d1 * s, p2 + d2 * t
    return float(np.linalg.norm(c1 - c2)), c1, c2


def point_box(p, lo, hi):
    c = np.clip(p, lo, hi)
    return float(np.linalg.norm(p - c)), c


def closest_seg_box(p, q, lo, hi, n: int = 33, iters: int = 25):
    """Segment to AABB distance.  The point-to-box distance is convex, so
    its restriction to the segment is convex in the parameter s: a
    vectorised scan brackets the minimiser and a golden-section search
    refines it.  Returns (dist, point_on_segment, point_on_box); dist = 0
    if they intersect."""
    s = np.linspace(0.0, 1.0, n)
    P = p[None, :] + (q - p)[None, :] * s[:, None]
    D = np.linalg.norm(P - np.clip(P, lo, hi), axis=1)
    k = int(np.argmin(D))
    if D[k] == 0.0:
        return 0.0, P[k], P[k]
    a, b = s[max(k - 1, 0)], s[min(k + 1, n - 1)]
    g = (np.sqrt(5) - 1) / 2
    f = lambda t: point_box(p + (q - p) * t, lo, hi)[0]
    x1, x2 = b - g * (b - a), a + g * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(iters):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - g * (b - a); f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + g * (b - a); f2 = f(x2)
    t = min((s[k], 0.5 * (a + b)), key=f)
    ps = p + (q - p) * t
    d, pb = point_box(ps, lo, hi)
    return d, ps, pb


# ------------------------------------------------------------ batched
def batch_seg_box(P0, P1, LO, HI, n: int = 17, iters: int = 30):
    """Distances between every segment (rows of P0,P1: [ns,3]) and every
    AABB (LO,HI: [nb,3]).  Same convexity argument as closest_seg_box:
    a coarse scan brackets the minimiser, then a vectorised golden-section
    search refines all pairs at once.  Returns D[ns,nb] and the parameter
    S[ns,nb] of the closest point on each segment."""
    ns, nb = len(P0), len(LO)
    if ns == 0 or nb == 0:
        return np.zeros((ns, nb)), np.zeros((ns, nb))
    d = (P1 - P0)[:, None, :]                       # [ns,1,3]
    p0 = P0[:, None, :]
    lo, hi = LO[None, :, :], HI[None, :, :]

    def f(s):                                        # s: [ns,nb]
        X = p0 + d * s[..., None]
        return np.linalg.norm(X - np.clip(X, lo, hi), axis=-1)

    grid = np.linspace(0.0, 1.0, n)
    X = P0[:, None, :] + (P1 - P0)[:, None, :] * grid[None, :, None]       # [ns,n,3]
    Dg = np.linalg.norm(X[:, None] - np.clip(X[:, None], LO[None, :, None], HI[None, :, None]), axis=-1)  # [ns,nb,n]
    k = Dg.argmin(-1)
    a = grid[np.clip(k - 1, 0, n - 1)]
    b = grid[np.clip(k + 1, 0, n - 1)]
    g = (np.sqrt(5) - 1) / 2
    x1, x2 = b - g * (b - a), a + g * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(iters):
        m = f1 < f2
        b = np.where(m, x2, b)
        a = np.where(m, a, x1)
        x1, x2 = b - g * (b - a), a + g * (b - a)
        f1, f2 = f(x1), f(x2)
    s_ref = 0.5 * (a + b)
    s_grid = grid[k]
    D1, D2 = f(s_ref), Dg.min(-1)
    S = np.where(D1 < D2, s_ref, s_grid)
    return np.minimum(D1, D2), S


def batch_seg_seg(A0, A1, B0, B1):
    """Closest distances between all segment pairs (Ericson 5.1.9,
    vectorised).  Returns D[n,m], C1[n,m,3], C2[n,m,3]."""
    eps = 1e-12
    d1 = (A1 - A0)[:, None, :]
    d2 = (B1 - B0)[None, :, :]
    r = A0[:, None, :] - B0[None, :, :]
    a = np.sum(d1 * d1, -1) + 0 * r[..., 0]
    e = np.sum(d2 * d2, -1) + 0 * r[..., 0]
    f = np.sum(d2 * r, -1)
    c = np.sum(d1 * r, -1)
    b = np.sum(d1 * d2, -1)
    den = a * e - b * b
    s = np.where(den > eps, np.clip((b * f - c * e) / np.where(den > eps, den, 1), 0, 1), 0.0)
    t = np.where(e > eps, (b * s + f) / np.where(e > eps, e, 1), 0.0)
    # clamp t and recompute s
    s = np.where(t < 0, np.clip(-c / np.where(a > eps, a, 1), 0, 1), s)
    s = np.where(t > 1, np.clip((b - c) / np.where(a > eps, a, 1), 0, 1), s)
    t = np.clip(t, 0, 1)
    # degenerate first segment (point)
    s = np.where(a <= eps, 0.0, s)
    t = np.where(a <= eps, np.where(e > eps, np.clip(f / np.where(e > eps, e, 1), 0, 1), 0.0), t)
    s = np.where((e <= eps) & (a > eps), np.clip(-c / np.where(a > eps, a, 1), 0, 1), s)
    C1 = A0[:, None, :] + d1 * s[..., None]
    C2 = B0[None, :, :] + d2 * t[..., None]
    return np.linalg.norm(C1 - C2, axis=-1), C1, C2
