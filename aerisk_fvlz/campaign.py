"""
campaign.py — Orchestration of the experimental campaign.

Blocks:
  1. Calibration of the generator against official magnitudes.
  2. Out-of-sample validation of the forecasting layer.
  3. Main multi-seed campaign in S0 and S1 with the four policies.
  4. Renewable penetration sweep (seven levels derived from the official 2030
     target).
  5. Forecast error sensitivity: total scaling and partial per-source scaling
     (demand, wind, solar PV).
  6. Gate trade-off curve (risk threshold) and detector quality.
  7. Contingency battery, including the separation of the system into two
     islands.
  8. Ablations: without the risk gate, without the regime flag, without the
     non-relaxation ratchet, with oracle uncertainty, and with uncalibrated
     uncertainty.

Every figure published in the manuscript comes from the results file this
module generates; none is transcribed by hand.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .dispatch import GateConfig, simulate
from .forecast import QuantileForecaster
from .stats import convergence_curve, holm_bonferroni, mean_ci, paired_summary
from .system import (OFFICIAL_2025, PTECAN_2030_TARGET, S0_PV_MW, S0_WIND_MW,
                     SystemConfig, build_fleet, derive_s1_capacity)
from .world import build_world, calibration_report

METRICS = ("cost_MEUR", "fuel_cost_MEUR", "start_cost_MEUR", "ens_cost_MEUR",
           "violation_h", "ens_MWh", "curtailment_GWh", "co2_kt", "starts",
           "gate_h", "res_share_pct", "link_binding_h", "infeasible_h")
POLICIES = ("A", "B", "C", "D", "E", "F")


class _Cache:
    """Reuse worlds and forecasts across policies and scenarios.

    Capacity factors are independent of the structural scenario (only the
    installed capacity scaling them changes), so the quantile prediction of
    capacity factors is computed once per seed and rescaled.
    """

    def __init__(self, forecaster: QuantileForecaster):
        self.fc = forecaster
        self._cf: dict = {}

    def world_and_forecast(self, seed: int, cfg: SystemConfig, **kw):
        w = build_world(seed, cfg, **kw)
        key = (seed, kw.get("error_scale", 1.0),
               kw.get("error_scale_demand"), kw.get("error_scale_wind"),
               kw.get("error_scale_pv"), kw.get("detector_recall", 0.80),
               kw.get("detector_fpr", 0.05), kw.get("n_events", 18),
               kw.get("event_depth", 0.45))
        if key not in self._cf:
            self._cf[key] = self.fc.predict(w)
            self._cf[key]["_wcap"] = cfg.wind_MW
            self._cf[key]["_pcap"] = cfg.pv_MW
            if len(self._cf) > 240:
                self._cf.pop(next(iter(self._cf)))
        base = self._cf[key]
        kw_w = cfg.wind_MW / max(base["_wcap"], 1e-9)
        kp_v = cfg.pv_MW / max(base["_pcap"], 1e-9)
        f = {
            "D": base["D"],
            "W": {k: v * kw_w for k, v in base["W"].items()},
            "V": {k: v * kp_v for k, v in base["V"].items()},
        }
        return w, f


def _run_policies(cache, seeds, cfg, gate, policies=POLICIES,
                  world_kw=None, sim_kw=None):
    """Run a set of policies on the same realizations."""
    world_kw = world_kw or {}
    sim_kw = sim_kw or {}
    out = {m: {k: [] for k in METRICS} for m in policies}
    extra = {m: {"avoided": [], "false_alarm": [], "precision": [],
                 "nr_viol": [], "ens_h": [], "infeas": []} for m in policies}
    for s in seeds:
        w, f = cache.world_and_forecast(s, cfg, **world_kw)
        rA = simulate(w, cfg, f, "A", gate, keep_traces=True, **sim_kw)
        adv = rA.traces["adverse"]
        for m in policies:
            r = rA if m == "A" else simulate(
                w, cfg, f, m, gate,
                reference_adverse=(adv if m == "B" else None),
                check_non_relaxation=(m == "B"), **sim_kw)
            for k in METRICS:
                out[m][k].append(getattr(r, k))
            extra[m]["avoided"].append(r.avoided_adverse_h)
            extra[m]["false_alarm"].append(r.false_alarm_h)
            extra[m]["precision"].append(r.gate_precision_pct)
            extra[m]["nr_viol"].append(r.non_relaxation_violations)
            extra[m]["ens_h"].append(r.ens_h)
            extra[m]["infeas"].append(r.infeasible_h)
    return out, extra


def _summarise(out, extra, policies) -> dict:
    res = {}
    for m in policies:
        res[m] = {k: mean_ci(out[m][k]) for k in METRICS}
        res[m]["_extra"] = {k: mean_ci(v) for k, v in extra[m].items()
                            if not all(np.isnan(np.asarray(v, dtype=float)))}
    return res


ALL_PARTS = ("calibracion", "prevision", "principal", "penetracion",
             "sensibilidad", "factores", "puerta", "contingencias", "ablaciones")


def run_campaign(outdir: Path, *, n_seeds_main: int = 50, n_seeds_sweep: int = 25,
                 rho_on: float = 0.30, verbose: bool = True,
                 parts: tuple = ALL_PARTS) -> dict:
    """Run the campaign.  `parts` allows the execution to be split and the
    results accumulated in the same file, so that the complete campaign can be
    rebuilt in blocks without losing traceability."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    resfile = outdir / "campaign_results.json"
    previous = {}
    if resfile.exists():
        with open(resfile, encoding="utf-8") as fh:
            previous = json.load(fh)

    gate = GateConfig(rho_on=rho_on, rho_off=rho_on * 0.4)
    s1_wind, s1_pv = derive_s1_capacity()
    cfg0 = SystemConfig("S0", S0_WIND_MW, S0_PV_MW)
    cfg1 = SystemConfig("S1", s1_wind, s1_pv)

    results: dict = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_seeds_main": n_seeds_main,
            "n_seeds_sweep": n_seeds_sweep,
            "gate": asdict(gate),
            "scenario_S0": {"wind_MW": S0_WIND_MW, "pv_MW": S0_PV_MW,
                            "fuente": "REE CTSOC 2026, lamina 12 (a 31/12/2025)"},
            "scenario_S1": {"wind_MW": s1_wind, "pv_MW": s1_pv,
                            "objetivo": PTECAN_2030_TARGET,
                            "derivacion": "system.derive_s1_capacity"},
            "official_2025": OFFICIAL_2025,
        }
    }

    results.update({k: v for k, v in previous.items() if k != "meta"})

    # --- 1. generator calibration -------------------------------------
    if "calibracion" in parts:
        log("[1/8] Calibracion del generador ...")
        worlds_cal = [build_world(s, cfg0) for s in range(1, 21)]
        results["calibracion_generador"] = calibration_report(worlds_cal, OFFICIAL_2025)
        del worlds_cal

    # --- 2. forecasting layer ---------------------------------------------
    log("[2/8] Capa de prevision ...")
    fpath = outdir / "forecaster.pkl"
    if fpath.exists() and "prevision" not in parts:
        fc = QuantileForecaster.load(fpath)
    else:
        tr = [build_world(900 + i, cfg0) for i in range(5)]
        va = [build_world(950 + i, cfg0) for i in range(3)]
        te = [build_world(970 + i, cfg0) for i in range(4)]
        fc = QuantileForecaster.fit(tr, va, te)
        fc.save(fpath)
    # empirical error correlation: verifies the independence assumption
    va = [build_world(950 + i, cfg0) for i in range(3)]
    pv = fc.predict(va[0])
    e = np.column_stack([va[0].demand - pv["D"]["q50"],
                         va[0].wind - pv["W"]["q50"],
                         va[0].pv - pv["V"]["q50"]])
    results["prevision"] = {
        "conjunto_calibracion": fc.val_report,
        "conjunto_calibracion_sin_conformal": fc.val_report_raw,
        "conjunto_prueba": fc.test_report,
        "conjunto_prueba_sin_conformal": fc.test_report_raw,
        "semillas_prueba": [970 + i for i in range(4)],
        "validacion_fuera_de_muestra": fc.test_report,
        "sin_recalibracion_conformal": fc.test_report_raw,
        "lambda_conformal": fc.conformal,
        "semillas_entrenamiento": list(fc.train_seeds),
        "semillas_validacion": [w.seed for w in va],
        "correlacion_errores": np.corrcoef(e.T).round(4).tolist(),
    }
    del va
    cache = _Cache(fc)

    seeds_main = list(range(1, n_seeds_main + 1))
    seeds_sweep = list(range(1, n_seeds_sweep + 1))

    _do_principal = "principal" in parts
    if _do_principal:
        # --- 3. main campaign ---------------------------------------------
        log("[3/8] Campana principal multisemilla ...")
        results["principal"] = {}
        raw_main = {}
        for name, cfg in (("S0", cfg0), ("S1", cfg1)):
            t0 = time.time()
            out, extra = _run_policies(cache, seeds_main, cfg, gate)
            raw_main[name] = out
            results["principal"][name] = _summarise(out, extra, POLICIES)
            # sweep of the value of lost load, reconstructed from the cost
            # decomposition without repeating the simulation
            results["principal"][name]["_voll_sweep"] = {
                str(v): {m: mean_ci([f + st + v * e / 1e6 for f, st, e in
                                     zip(out[m]["fuel_cost_MEUR"],
                                         out[m]["start_cost_MEUR"],
                                         out[m]["ens_MWh"])])
                         for m in POLICIES}
                for v in (0, 1000, 2500, 5000, 10000, 20000)}
            # paired tests
            comps, praw = {}, {}
            for a, b in (("B", "A"), ("B", "C"), ("B", "D"), ("C", "A"), ("D", "A"),
                         ("B", "E"), ("B", "F"), ("E", "A"), ("F", "A"), ("C", "F")):
                for k in ("cost_MEUR", "violation_h", "ens_MWh", "co2_kt", "curtailment_GWh"):
                    key = f"{a}_vs_{b}|{k}"
                    comps[key] = paired_summary(np.array(out[a][k]), np.array(out[b][k]))
                    praw[key] = comps[key]["p_wilcoxon"]
            results["principal"][name]["_comparaciones"] = comps
            results["principal"][name]["_holm"] = holm_bonferroni(praw)
            results["principal"][name]["_convergencia"] = {
                m: convergence_curve(out[m]["violation_h"]) for m in ("A", "B", "C")
            }
            log(f"      {name} completado en {time.time()-t0:.0f} s")

    _do_penetracion = "penetracion" in parts
    if _do_penetracion:
        # --- 4. penetration sweep ----------------------------------------
        log("[4/8] Barrido de penetracion renovable ...")
        total0 = S0_WIND_MW + S0_PV_MW
        total1 = s1_wind + s1_pv
        fw = S0_WIND_MW / total0
        levels = np.linspace(total0, total1, 7)
        results["penetracion"] = []
        for tot in levels:
            cfg = SystemConfig(f"P{tot:.0f}", round(tot * fw, 1), round(tot * (1 - fw), 1))
            out, extra = _run_policies(cache, seeds_sweep, cfg, gate)
            row = {"total_MW": float(tot), "wind_MW": cfg.wind_MW, "pv_MW": cfg.pv_MW,
                   "politicas": _summarise(out, extra, POLICIES),
                   "B_vs_A_violation": paired_summary(np.array(out["B"]["violation_h"]),
                                                      np.array(out["A"]["violation_h"])),
                   "B_vs_C_cost": paired_summary(np.array(out["B"]["cost_MEUR"]),
                                                 np.array(out["C"]["cost_MEUR"]))}
            results["penetracion"].append(row)
            log(f"      {tot:6.1f} MW renovables")

    _do_sensibilidad = "sensibilidad" in parts
    if _do_sensibilidad:
        # --- 5. error sensitivity -----------------------------------------
        log("[5/8] Sensibilidad al error de prevision ...")
        results["sensibilidad_error"] = {"total": [], "parcial": []}
        for k in (0.5, 1.0, 1.5, 2.0):
            out, extra = _run_policies(cache, seeds_sweep, cfg0, gate,
                                       world_kw=dict(error_scale=k))
            results["sensibilidad_error"]["total"].append(
                {"k_sigma": k, "politicas": _summarise(out, extra, POLICIES)})
            log(f"      k_sigma = {k}")
        for src, kw in (("demanda", dict(error_scale_demand=2.0)),
                        ("eolica", dict(error_scale_wind=2.0)),
                        ("fotovoltaica", dict(error_scale_pv=2.0))):
            out, extra = _run_policies(cache, seeds_sweep, cfg0, gate, world_kw=kw)
            results["sensibilidad_error"]["parcial"].append(
                {"fuente": src, "k_sigma": 2.0,
                 "politicas": _summarise(out, extra, POLICIES)})
            log(f"      fuente escalada: {src}")

    _do_factores = "factores" in parts
    if _do_factores:
        log("[6b/9] Sensibilidad del escenario 2030 a los factores de carga ...")
        from .system import CF_VARIANTS, derive_capacity_factors
        base_cf = derive_capacity_factors()
        results["factores_de_carga"] = []
        for vname, pair in CF_VARIANTS.items():
            cfw, cfv = base_cf if pair is None else pair
            w1, p1 = derive_s1_capacity(cf_wind=cfw, cf_pv=cfv)
            cfgv = SystemConfig(f"S1-{vname}", w1, p1)
            out, extra = _run_policies(
                cache, seeds_sweep, cfgv, gate,
                world_kw=dict(cf_wind_target=cfw, cf_pv_target=cfv))
            results["factores_de_carga"].append(
                {"variante": vname, "cf_wind": cfw, "cf_pv": cfv,
                 "wind_MW": w1, "pv_MW": p1, "total_MW": round(w1 + p1, 1),
                 "politicas": _summarise(out, extra, POLICIES)})
            log(f"      {vname}: cf=({cfw}, {cfv}) -> {w1 + p1:.1f} MW")

    _do_puerta = "puerta" in parts
    if _do_puerta:
        # --- 6. gate trade-off curve -------------------------------
        log("[6/8] Curva de compromiso de la puerta ...")
        results["puerta"] = {"umbral": [], "detector": []}
        for name, cfg in (("S0", cfg0), ("S1", cfg1)):
            for ron in (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
                g = GateConfig(rho_on=ron, rho_off=ron * 0.4)
                out, extra = _run_policies(cache, seeds_sweep, cfg, g,
                                           policies=("A", "B"))
                results["puerta"]["umbral"].append(
                    {"escenario": name, "rho_on": ron,
                     "politicas": _summarise(out, extra, ("A", "B"))})
            log(f"      umbrales barridos en {name}")
        # (i) raw sweep, same threshold and different activation rates
        for rec, fpr in ((0.60, 0.02), (0.80, 0.05), (0.95, 0.15), (1.00, 0.00)):
            out, extra = _run_policies(cache, seeds_sweep, cfg0, gate,
                                       policies=("A", "B"),
                                       world_kw=dict(detector_recall=rec, detector_fpr=fpr))
            results["puerta"]["detector"].append(
                {"recall": rec, "fpr": fpr,
                 "politicas": _summarise(out, extra, ("A", "B"))})
        # (ii) controlled sweep: rho_on is adjusted by bisection to equalize the
        # activation rate, so that the comparison isolates the quality of the
        # detector and not the level of conservatism
        results["puerta"]["detector_controlado"] = []
        target_rate = 45.0
        seeds_cal = seeds_sweep[:6]
        for rec, fpr in ((0.60, 0.02), (0.80, 0.05), (0.95, 0.15), (1.00, 0.00)):
            lo, hi = 0.01, 0.95
            for _ in range(9):
                mid = 0.5 * (lo + hi)
                g = GateConfig(rho_on=mid, rho_off=mid * 0.4)
                o, _e = _run_policies(cache, seeds_cal, cfg0, g, policies=("B",),
                                      world_kw=dict(detector_recall=rec,
                                                    detector_fpr=fpr))
                rate = float(np.mean(o["B"]["gate_h"])) / 87.60
                if rate > target_rate:
                    lo = mid
                else:
                    hi = mid
            rho_star = 0.5 * (lo + hi)
            g = GateConfig(rho_on=rho_star, rho_off=rho_star * 0.4)
            out, extra = _run_policies(cache, seeds_sweep, cfg0, g,
                                       policies=("A", "B"),
                                       world_kw=dict(detector_recall=rec,
                                                     detector_fpr=fpr))
            results["puerta"]["detector_controlado"].append(
                {"recall": rec, "fpr": fpr, "rho_on": round(rho_star, 4),
                 "politicas": _summarise(out, extra, ("A", "B"))})
            log(f"      detector controlado S={rec} FP={fpr} -> rho_on={rho_star:.3f}")
        log("      calidad del detector barrida")

    _do_contingencias = "contingencias" in parts
    if _do_contingencias:
        # --- 7. contingencies --------------------------------------------------
        log("[7/8] Bateria de contingencias ...")
        rng = np.random.default_rng(4242)
        unit_win = tuple((int(s), int(s) + 48, 6) for s in rng.choice(np.arange(500, 8200), 8, replace=False))
        link_win = tuple((int(s), int(s) + 24) for s in rng.choice(np.arange(500, 8200), 6, replace=False))
        link_win_long = tuple((int(s), int(s) + 72) for s in
                              rng.choice(np.arange(500, 8100), 8, replace=False))
        scenarios_cont = {
            "N-1 prolongada del mayor grupo": dict(unit_outage_windows=unit_win),
            "separacion del sistema en dos islas": dict(link_outage_windows=link_win),
            "meteorologia adversa correlacionada": dict(n_events=40, event_depth=0.60),
            "indisponibilidad estructural elevada": dict(outage_rate=0.08),
        }
        scenarios_cont["separacion prolongada en dos islas"] = dict(
            link_outage_windows=link_win_long)
        results["contingencias"] = {}
        for name, kw in scenarios_cont.items():
            blk = {}
            for sname, cfg in (("S0", cfg0), ("S1", cfg1)):
                out, extra = _run_policies(cache, seeds_sweep, cfg, gate, world_kw=kw)
                blk[sname] = _summarise(out, extra, POLICIES)
                blk[sname]["_B_vs_A_violation"] = paired_summary(
                    np.array(out["B"]["violation_h"]), np.array(out["A"]["violation_h"]))
            results["contingencias"][name] = blk
            log(f"      {name}")

    _do_ablaciones = "ablaciones" in parts
    if _do_ablaciones:
        # --- 8. ablations -----------------------------------------------------
        log("[8/8] Ablaciones ...")
        results["ablaciones"] = {}
        variants = {
            "completa": dict(gate=gate, sim_kw={}),
            "sin condicion de riesgo": dict(gate=GateConfig(rho_on=rho_on, rho_off=rho_on * 0.4,
                                                            use_risk=False), sim_kw={}),
            "sin bandera de regimen": dict(gate=GateConfig(rho_on=rho_on, rho_off=rho_on * 0.4,
                                                           use_flag=False), sim_kw={}),
            "sin histeresis ni permanencia": dict(gate=GateConfig(rho_on=rho_on, rho_off=rho_on,
                                                                  dwell_h=0), sim_kw={}),
            "incertidumbre oraculo": dict(gate=gate, sim_kw=dict(oracle_sigma=True)),
        }
        for name, v in variants.items():
            blk = {}
            for sname, cfg in (("S0", cfg0), ("S1", cfg1)):
                out, extra = _run_policies(cache, seeds_sweep, cfg, v["gate"],
                                           policies=("A", "B"), sim_kw=v["sim_kw"])
                blk[sname] = _summarise(out, extra, ("A", "B"))
            results["ablaciones"][name] = blk
            log(f"      {name}")

        # --- uncertainty without conformal recalibration ----------------------------
        class _Raw:
            def __init__(self, fc):
                self.fc = fc
            def predict(self, w):
                return self.fc.predict(w, conformal=False)
        cache_raw = _Cache(_Raw(fc))
        blk = {}
        for sname, cfg in (("S0", cfg0), ("S1", cfg1)):
            out, extra = _run_policies(cache_raw, seeds_sweep, cfg, gate, policies=("A", "B"))
            blk[sname] = _summarise(out, extra, ("A", "B"))
        results["ablaciones"]["incertidumbre sin recalibrar"] = blk

    results["meta"]["runtime_s"] = round(time.time() - t_start, 1)
    with open(outdir / "campaign_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False, default=float)
    log(f"Campana completada en {results['meta']['runtime_s']} s")
    return results
