"""Pre-computed impact-gain tables for fast, conservative contact checks.

gain(m_l, m_m, K_c) = peak force per unit impact speed (N s/m) from the
unilateral three-mass model in ``safety.ImpactModel``.  Tables are built
on a log grid per (k_H, m_H) pair; queries return the maximum over the 8
corners of the enclosing cell (conservative if the gain is monotone in
the masses; the residual error is measured in tests/scripts).
"""
from __future__ import annotations

import os
from functools import lru_cache
import numpy as np

from .safety import ISO_BODY, ImpactModel, transient_limit

ML = np.logspace(np.log10(0.3), 4, 18)
MM = np.logspace(-1, 4, 18)
KC = np.logspace(1, np.log10(3e7), 28)
TABLE_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "gain_table.npz")


def _pairs():
    return sorted({(v[2], v[3]) for v in ISO_BODY.values() if v[1] is not None})


def _row(args):
    k, mH, i = args
    M = ImpactModel()
    out = np.zeros((len(MM), len(KC)))
    for j, mm in enumerate(MM):
        for l, K in enumerate(KC):
            out[j, l] = M.gain(mm, ML[i], K, k, mH)
    return out


def build(path: str = TABLE_PATH, procs: int = 2):
    from multiprocessing import Pool
    tables = {}
    with Pool(procs) as pool:
        for (k, mH) in _pairs():
            rows = pool.map(_row, [(k, mH, i) for i in range(len(ML))])
            tables[f"{k:.0f}_{mH:.2f}"] = np.stack(rows)        # [ml, mm, K]
    np.savez_compressed(path, ML=ML, MM=MM, KC=KC, **tables)
    return path


@lru_cache(maxsize=1)
def _load(path: str = TABLE_PATH):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def gain_lookup(region: str, m_l: float, m_m: float, K_c: float) -> float:
    _, _, k, mH = ISO_BODY[region]
    T = _load()
    tab = T[f"{k:.0f}_{mH:.2f}"]
    if m_l > ML[-1] or m_m > MM[-1]:
        return float(np.sqrt(mH * k) * 1.02)      # asymptotic bound (rigid, m_R -> inf)
    idx = []
    for val, grid in ((m_l, ML), (m_m, MM), (K_c, KC)):
        v = np.clip(val, grid[0], grid[-1])
        i = int(np.clip(np.searchsorted(grid, v) - 1, 0, len(grid) - 2))
        idx.append((i, i + 1))
    cell = tab[np.ix_(idx[0], idx[1], idx[2])]
    return float(cell.max()) * 1.01     # 1 % margin for non-monotone cells (see tests)


def vmax_lookup(region: str, m_l: float, m_m: float, K_c: float) -> float:
    F = transient_limit(region)
    if F is None:
        return 0.0
    return F / gain_lookup(region, m_l, m_m, K_c)


# ------------------------------------------------------- clamping tables
CML = np.logspace(np.log10(0.3), 4, 14)
CKC = np.logspace(1, 8, 20)
CLAMP_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "clamp_table.npz")


def _clamp_row(args):
    from .safety import ClampModel
    region, i = args
    C = ClampModel()
    return [C.vmax(region, CML[i], K) for K in CKC]


def clamp_regions():
    return [r for r, v in ISO_BODY.items() if v[1] is not None]


def build_clamp(path: str = CLAMP_PATH, procs: int = 2):
    from multiprocessing import Pool
    tabs = {}
    with Pool(procs) as pool:
        for r in clamp_regions():
            tabs[r] = np.array(pool.map(_clamp_row, [(r, i) for i in range(len(CML))]))
    np.savez_compressed(path, CML=CML, CKC=CKC, **tabs)
    return path


@lru_cache(maxsize=1)
def _load_clamp(path: str = CLAMP_PATH):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def vmax_clamp_lookup(region: str, m_l: float, K_c: float) -> float:
    """Conservative (min over the enclosing cell corners; v_max decreases
    with mass and stiffness)."""
    if transient_limit(region) is None:
        return 0.0
    T = _load_clamp()[region]
    idx = []
    for val, grid in ((m_l, CML), (K_c, CKC)):
        v = np.clip(val, grid[0], grid[-1])      # beyond the grid: last (most conservative) node
        i = int(np.clip(np.searchsorted(grid, v) - 1, 0, len(grid) - 2))
        idx.append((i, i + 1))
    return float(T[np.ix_(idx[0], idx[1])].min())
