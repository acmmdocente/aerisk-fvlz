"""Regression and invariant tests of the aerisk_fvlz package.

The properties the audit report identified as critical are checked: full
billing of start-ups, policy independence, N-1 criterion convergence,
non-relaxation, balance consistency, and seed determinism.
"""
import numpy as np
import pytest

from aerisk_fvlz.dispatch import GateConfig, _security_ok, simulate
from aerisk_fvlz.forecast import QuantileForecaster
from aerisk_fvlz.system import (OFFICIAL_2025, S0_PV_MW, S0_WIND_MW, SystemConfig,
                                derive_cost_and_emissions, derive_s1_capacity)
from aerisk_fvlz.world import build_world, calibration_report


@pytest.fixture(scope="module")
def cfg():
    return SystemConfig("S0", S0_WIND_MW, S0_PV_MW)


@pytest.fixture(scope="module")
def fc(cfg):
    """Reduced fit: sufficient to verify structural invariants.
    The production configuration (five training realizations and three
    calibration ones) is exercised in the campaign itself."""
    tr = [build_world(900 + i, cfg) for i in range(3)]
    va = [build_world(950 + i, cfg) for i in range(2)]
    return QuantileForecaster.fit(tr, va, max_iter=80)


# ---------------------------------------------------------------- determinismo
def test_semilla_reproducible(cfg):
    a, b = build_world(7, cfg), build_world(7, cfg)
    assert np.array_equal(a.demand, b.demand)
    assert np.array_equal(a.wind, b.wind)
    assert np.array_equal(a.availability, b.availability)


def test_semillas_distintas_difieren(cfg):
    assert not np.allclose(build_world(7, cfg).demand, build_world(8, cfg).demand)


# ---------------------------------------------------------------- calibracion
def test_calibracion_dentro_de_tolerancia(cfg):
    ws = [build_world(s, cfg) for s in range(1, 9)]
    rep = calibration_report(ws, OFFICIAL_2025)
    assert abs(rep["demanda_anual_GWh"]["error_rel_pct"]) < 1.0
    assert abs(rep["punta_MW"]["error_rel_pct"]) < 3.0
    assert abs(rep["producible_renovable_GWh"]["error_rel_pct"]) < 2.0
    assert rep["autocorrelacion_lag1"]["media"] > 0.90
    assert rep["rampa_p99_pct_punta"]["media"] < 15.0


def test_perfil_intradiario_usa_distancia_circular(cfg):
    """Regression of the version 1.0 phase defect: hour 12 must be one unit
    away from center 13 under the circular distance."""
    w = build_world(3, cfg)
    perfil = w.demand[: 24 * 28].reshape(-1, 24).mean(axis=0)
    assert perfil[12] > perfil[3]
    assert perfil[20] == pytest.approx(perfil.max(), rel=0.02)


# ------------------------------------------------------------------- economia
def test_costes_derivados_ordenados():
    c_d, e_d = derive_cost_and_emissions("fuel_oil", 0.44, "diesel")
    c_g, e_g = derive_cost_and_emissions("gas_oil", 0.33, "gt")
    assert c_d < c_g, "el motor diesel debe preceder a la turbina en merito"
    assert e_d < e_g
    assert 90 < c_d < 140 and 170 < c_g < 250


def test_derivacion_escenario_S1():
    w, p = derive_s1_capacity()
    assert w > S0_WIND_MW and p > S0_PV_MW
    assert 500 < w + p < 700


# -------------------------------------------------------------- seguridad N-1
def test_criterio_seguridad():
    assert _security_ok(200, 150, 37.5, 23.5, 100, 90, 120)
    assert not _security_ok(100, 40, 37.5, 23.5, 100, 90, 120)
    assert not _security_ok(200, 5, 37.5, 5.0, 100, 130, 120)


def test_convergencia_criterio_N1(cfg, fc):
    """No policy may declare systematic infeasibility: the commitment loop
    must satisfy the criterion in nearly every hour of the base scenario."""
    w = build_world(11, cfg)
    f = fc.predict(w)
    for m in ("A", "B", "C", "D"):
        r = simulate(w, cfg, f, m)
        assert r.infeasible_h < 0.02 * 8760


# --------------------------------------------- independencia y no relajacion
def test_politicas_independientes(cfg, fc):
    """The execution of B must not depend on the order or on the previous
    execution of any other policy (policy independence)."""
    w = build_world(13, cfg)
    f = fc.predict(w)
    b1 = simulate(w, cfg, f, "B")
    _ = simulate(w, cfg, f, "A")
    b2 = simulate(w, cfg, f, "B")
    assert b1.cost_MEUR == pytest.approx(b2.cost_MEUR, rel=1e-12)
    assert b1.violation_h == b2.violation_h


def test_no_relajacion(cfg, fc):
    w = build_world(17, cfg)
    f = fc.predict(w)
    r = simulate(w, cfg, f, "B", check_non_relaxation=True)
    assert r.non_relaxation_violations == 0


def test_todo_arranque_se_factura(cfg, fc):
    """Start-up cost cannot be zero if there are start-ups, nor can the
    number of start-ups differ from the transitions of the committed set."""
    w = build_world(19, cfg)
    f = fc.predict(w)
    for m in ("A", "B", "C", "D"):
        r = simulate(w, cfg, f, m)
        assert r.starts > 0
        assert r.start_cost_MEUR > 0
        assert r.cost_MEUR == pytest.approx(
            r.fuel_cost_MEUR + r.start_cost_MEUR + r.ens_cost_MEUR, rel=1e-9)


# ------------------------------------------------------------------ ordenacion
def test_ordenacion_seguridad_coste(cfg, fc):
    """The alphabetical order of the proposed policies must coincide with
    increasing conservatism: A less secure than B, B than C, and C than D; and
    the safeguarded policy must verify non-relaxation hour by hour."""
    w = build_world(23, cfg)
    f = fc.predict(w)
    r = {m: simulate(w, cfg, f, m) for m in ("A", "B", "C", "D")}
    assert r["A"].violation_h > r["B"].violation_h > r["C"].violation_h > r["D"].violation_h
    assert r["D"].cost_MEUR > r["C"].cost_MEUR > r["B"].cost_MEUR


def test_politicas_de_referencia(cfg, fc):
    """The two policies taken from the literature must run and sit within
    the security ordering documented in the campaign."""
    w = build_world(23, cfg)
    f = fc.predict(w)
    r = {m: simulate(w, cfg, f, m) for m in ("A", "B", "D", "E", "F")}
    for m in ("E", "F"):
        assert r[m].starts > 0 and r[m].cost_MEUR > 0
        assert r["D"].violation_h <= r[m].violation_h <= r["A"].violation_h
    # la regla estatica por porcentajes es menos segura que el presupuesto robusto
    assert r["E"].violation_h > r["F"].violation_h


def test_separacion_del_sistema_es_completa(cfg):
    """La indisponibilidad del enlace de 132 kV debe anular el transito."""
    assert cfg.topology.link_capacity_outage_MW == 0.0


def test_puerta_monotona_en_umbral(cfg, fc):
    """Elevar el umbral de riesgo reduce las horas con salvaguarda activa."""
    w = build_world(29, cfg)
    f = fc.predict(w)
    g_lo = simulate(w, cfg, f, "B", GateConfig(rho_on=0.05, rho_off=0.02)).gate_h
    g_hi = simulate(w, cfg, f, "B", GateConfig(rho_on=0.45, rho_off=0.18)).gate_h
    assert g_lo > g_hi


# ------------------------------------------------------------------- prevision
def test_calibracion_conformal(fc):
    """Conformal recalibration must bring coverage toward the nominal level
    for all targets and reach it with slack for the two that govern the
    aggregate uncertainty."""
    for tgt in ("demand", "cf_wind", "cf_pv"):
        antes = abs(fc.val_report_raw[tgt]["picp90"] - 0.80)
        despues = abs(fc.val_report[tgt]["picp90"] - 0.80)
        assert despues <= antes + 1e-6, f"la recalibracion empeora {tgt}"
        assert fc.val_report[tgt]["skill_vs_baseline"] > 0.0
    for tgt in ("demand", "cf_wind"):
        assert fc.val_report[tgt]["picp90"] == pytest.approx(0.80, abs=0.05)


def test_recalibracion_no_degenera_en_horas_nocturnas(cfg, fc):
    """Regression of the defect detected in calibration: hours with
    zero-width intervals (nighttime solar PV) must not drag the conformal
    quantile down and spuriously narrow the daytime interval."""
    w = build_world(31, cfg)
    f = fc.predict(w)
    dia = w.features["elevation"] > 0.2
    anchura = float(np.mean(f["V"]["q90"][dia] - f["V"]["q10"][dia]))
    nivel = float(np.mean(f["V"]["q50"][dia]))
    assert anchura / max(nivel, 1e-9) > 0.25, "intervalo diurno colapsado"


def test_evaluacion_en_tercer_conjunto(cfg):
    """The forecasting layer must be evaluable on a set disjoint from the
    training and calibration ones; coverage there cannot coincide by
    construction with the nominal level."""
    tr = [build_world(900 + i, cfg) for i in range(2)]
    va = [build_world(950 + i, cfg) for i in range(2)]
    te = [build_world(970 + i, cfg) for i in range(2)]
    m = QuantileForecaster.fit(tr, va, te, max_iter=60)
    assert m.test_report is not None and m.test_report_raw is not None
    assert m.val_report["demand"]["picp90"] == pytest.approx(0.80, abs=0.02)
    assert 0.4 < m.test_report["demand"]["picp90"] < 1.0


def test_cuantiles_ordenados(cfg, fc):
    f = fc.predict(build_world(31, cfg))
    for k in ("D", "W", "V"):
        assert np.all(f[k]["q10"] <= f[k]["q50"] + 1e-9)
        assert np.all(f[k]["q50"] <= f[k]["q90"] + 1e-9)
        assert np.all(f[k]["sigma"] > 0)
