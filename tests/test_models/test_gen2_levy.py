"""Unit tests for Generation-2 exponential Lévy models."""

import numpy as np
import pytest
from volsurf.models.gen2_levy import NIGModel, VGModel, BGMModel


# ---------------------------------------------------------------------------
# NIG tests
# ---------------------------------------------------------------------------

def test_nig_cf_at_zero():
    nig = NIGModel()
    phi = nig.characteristic_function(0.0, 1.0, nig.params)
    assert np.isclose(phi, 1.0, atol=1e-10)


def test_nig_cgf_consistency():
    nig = NIGModel()
    u = 1.0
    t = 2.0
    phi = nig.characteristic_function(u, t, nig.params)
    psi = np.log(phi)
    cgf = nig.cumulant_generating_function(1j * u, nig.params)
    assert np.isclose(psi, t * cgf, atol=1e-10)


def test_nig_esscher_solution():
    nig = NIGModel()
    theta = nig.solve_esscher(0.05, 0.0, nig.params)
    lhs = (
        nig.cumulant_generating_function(theta + 1, nig.params)
        - nig.cumulant_generating_function(theta, nig.params)
    )
    assert np.isclose(lhs, 0.05, atol=1e-6)


def test_nig_rn_martingale():
    nig = NIGModel()
    r, q = 0.05, 0.0
    T = 1.0
    phi = nig.characteristic_function_rn(-1j, T, r, q, nig.params)
    assert np.isclose(phi, np.exp((r - q) * T), atol=1e-6)


def test_nig_invalid_params_rejected():
    nig = NIGModel()
    bad = {"alpha": 1.0, "beta": 2.0, "delta": 0.5, "mu": 0.0}  # |beta| > alpha
    assert not nig._validate_params(bad)


# ---------------------------------------------------------------------------
# VG tests
# ---------------------------------------------------------------------------

def test_vg_cf_at_zero():
    vg = VGModel()
    phi = vg.characteristic_function(0.0, 1.0, vg.params)
    assert np.isclose(phi, 1.0, atol=1e-10)


def test_vg_esscher_solution():
    vg = VGModel()
    theta = vg.solve_esscher(0.05, 0.0, vg.params)
    lhs = (
        vg.cumulant_generating_function(theta + 1, vg.params)
        - vg.cumulant_generating_function(theta, vg.params)
    )
    assert np.isfinite(theta)
    assert np.isclose(lhs, 0.05, atol=1e-6)


def test_vg_rn_martingale():
    vg = VGModel()
    r, q = 0.05, 0.0
    T = 1.0
    phi = vg.characteristic_function_rn(-1j, T, r, q, vg.params)
    assert np.isclose(phi, np.exp((r - q) * T), atol=1e-6)


# ---------------------------------------------------------------------------
# BGM tests
# ---------------------------------------------------------------------------

def test_bgm_cf_at_zero():
    bgm = BGMModel()
    phi = bgm.characteristic_function(0.0, 1.0, bgm.params)
    assert np.isclose(phi, 1.0, atol=1e-10)


def test_bgm_esscher_solution():
    bgm = BGMModel()
    theta = bgm.solve_esscher(0.05, 0.0, bgm.params)
    lhs = (
        bgm.cumulant_generating_function(theta + 1, bgm.params)
        - bgm.cumulant_generating_function(theta, bgm.params)
    )
    assert np.isclose(lhs, 0.05, atol=1e-6)


def test_bgm_rn_martingale():
    bgm = BGMModel()
    r, q = 0.05, 0.0
    T = 1.0
    phi = bgm.characteristic_function_rn(-1j, T, r, q, bgm.params)
    assert np.isclose(phi, np.exp((r - q) * T), atol=1e-6)


# ---------------------------------------------------------------------------
# Pricing sanity test
# ---------------------------------------------------------------------------

def test_nig_prices_positive():
    nig = NIGModel()
    strikes = np.array([280.0, 300.0, 320.0])
    mats = np.array([30.0 / 365.0] * 3)
    prices = nig.price_points(strikes, mats, 307.59, 0.04, 0.004)
    assert prices is not None
    assert np.all(np.isfinite(prices))
    assert np.all(prices >= 0.0)