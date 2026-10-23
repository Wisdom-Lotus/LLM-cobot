"""Unit tests: each model is checked against an independent computation."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vsdt.robot import ur5e_class_arm
from vsdt import ik
from vsdt.geometry import closest_seg_box, closest_seg_seg
from vsdt.vssea import VSAMParams, kb_timo, kb_eb, fe_plane, elastica, tau_max_elastica, tau_max_stress
from vsdt.safety import ImpactModel, ISO_BODY, ClampModel
from vsdt.plan import extract_json, validate, PlanFormatError

ARM = ur5e_class_arm()
RNG = np.random.default_rng(0)


def test_jacobian_matches_finite_difference():
    for _ in range(10):
        q = RNG.uniform(-np.pi, np.pi, 6)
        J = ARM.jacobian(q)[:3]
        Jn = np.zeros((3, 6))
        for j in range(6):
            d = np.zeros(6); d[j] = 1e-6
            Jn[:, j] = (ARM.fk(q + d)[:3, 3] - ARM.fk(q - d)[:3, 3]) / 2e-6
        assert np.abs(J - Jn).max() < 1e-7


def test_gravity_is_gradient_of_potential_energy():
    for _ in range(10):
        q = RNG.uniform(-np.pi, np.pi, 6)
        def U(q):
            m, p, _ = ARM._mass_points(q, ARM.frames_fk(q), None)
            return float((m * 9.81 * p[:, 2]).sum())
        g = np.array([(U(q + e * 1e-6) - U(q - e * 1e-6)) / 2e-6 for e in np.eye(6)])
        assert np.abs(g - ARM.gravity_torque(q)).max() < 1e-5


def test_mass_matrix_symmetric_positive_definite():
    q = RNG.uniform(-np.pi, np.pi, 6)
    M = ARM.mass_matrix(q)
    assert np.allclose(M, M.T)
    assert np.linalg.eigvalsh(M).min() > 0


def test_ik_analytic_jacobian():
    q = RNG.uniform(-3, 3, 6)
    z = np.array([0.3, -0.2, -0.9]); z /= np.linalg.norm(z)
    Ja = ik._jac_analytic(ARM, q, z, 0.5)
    Jf = ik._jac_fd(ARM, q, np.zeros(3), z, 0.5, 1e-6)
    assert np.abs(Ja - Jf).max() < 1e-5


def test_ik_recovers_reachable_pose():
    q = RNG.uniform(-np.pi, np.pi, 6)
    T = ARM.fk(q)
    qs, ok, _, err = ik.solve(ARM, T[:3, 3], T[:3, 2])
    assert ok and np.linalg.norm(ARM.fk(qs)[:3, 3] - T[:3, 3]) < 1e-4


def test_segment_box_distance_against_dense_sampling():
    for _ in range(200):
        p, q = RNG.normal(size=3), RNG.normal(size=3)
        lo = RNG.normal(size=3) * 0.5; hi = lo + RNG.uniform(0.05, 1, 3)
        d, _, _ = closest_seg_box(p, q, lo, hi)
        P = p + (q - p) * np.linspace(0, 1, 5001)[:, None]
        ref = np.linalg.norm(P - np.clip(P, lo, hi), axis=1).min()
        assert d <= ref + 1e-6 and d >= ref - 1e-3


def test_segment_segment_distance():
    d, _, _ = closest_seg_seg(np.zeros(3), np.array([1.0, 0, 0]), np.array([0.5, 1, -1]), np.array([0.5, 1, 1]))
    assert abs(d - 1.0) < 1e-12


def test_fe_agrees_with_timoshenko():
    p = VSAMParams()
    for x in (0.03, 0.1):
        assert abs(fe_plane(p, x, nx=60, ny=2) / kb_timo(p, x) - 1) < 0.005


def test_elastica_reduces_to_linear_for_small_deflection():
    p = VSAMParams()
    e = elastica(p, 0.1, 1e-4)
    assert e["ok"] and abs(e["F_tan"] / (kb_eb(p, 0.1) * 1e-4) - 1) < 1e-3


def test_elastica_capacity_not_above_linear_soft_end():
    p = VSAMParams()
    assert tau_max_elastica(p, p.x_max) <= tau_max_stress(p, p.x_max)


def test_impact_gain_limits_and_ode():
    from scipy.integrate import solve_ivp
    M = ImpactModel()
    k, mH = ISO_BODY["chest"][2], ISO_BODY["chest"][3]
    mm, ml, K = 8.0, 3.0, 300.0
    g = M.gain(mm, ml, K, k, mH)
    c = 2 * 0.05 * np.sqrt(K * mm * ml / (mm + ml))
    def f(t, y):
        xm, xl, xh, vm, vl, vh = y
        Fc = k * max(0, xl - xh); Fs = K * (xm - xl) + c * (vm - vl)
        return [vm, vl, vh, -Fs / mm, (Fs - Fc) / ml, Fc / mH]
    s = solve_ivp(f, (0, 8), [0, 0, 0, 1, 1, 0], max_step=5e-5, rtol=1e-9, atol=1e-12)
    ref = k * np.maximum(0, s.y[1] - s.y[2]).max()
    assert abs(g / ref - 1) < 1e-3
    # rigid limit equals the ISO two-body formula
    gr = M.gain(mm, ml, 1e9, k, mH)
    mu = (mm + ml) * mH / (mm + ml + mH)
    assert abs(gr / np.sqrt(mu * k) - 1) < 1e-6


def test_clamp_vmax_decreases_with_stiffness():
    C = ClampModel()
    v = [C.vmax("hand", 5.0, K) for K in (100, 1000, 1e4, 1e5)]
    assert all(a >= b for a, b in zip(v, v[1:]))


def test_plan_parsing():
    txt = 'Sure.\n```json\n{"feasible": true, "steps": [{"op": "release"}]}\n```'
    p = extract_json(txt)
    validate(p)
    with pytest.raises(PlanFormatError):
        validate({"feasible": True, "steps": [{"op": "move"}]})
