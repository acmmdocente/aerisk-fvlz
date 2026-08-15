"""
world.py — Stochastic generator of hourly realizations of the FV-LZ system.

CENTRAL METHODOLOGICAL NOTICE
-----------------------------
The hourly series are SYNTHETIC.  As of the time of writing, no public
downloadable hourly series of demand and generation exists for the joint
Fuerteventura-Lanzarote system.  The generator is calibrated against the
annual, peak, and curtailment magnitudes published by the system operator, and
its goodness of fit is audited in `calibration_report` (energy, peak, load
factor, load duration curve, autocorrelation, ramps, and available renewable
energy).  No conclusion rests on unaudited absolute levels: the tests are
always paired over identical realizations (common random numbers).

CAUSAL STRUCTURE OF THE ERROR
-----------------------------
The generator produces (i) the physical realization and (ii) a set of
PREDICTORS actually observable at decision time: calendar, demand persistence,
meteorological forecast proxies, and the flag of an imperfect regime detector.
Tail episodes (unanticipated wind-drop spells) are applied ON THE REALIZATION
after building the forecast proxy, so that they are genuinely unanticipable:
the error is not injected artificially into the forecast; it emerges from the
difference between the foreseeable and the realized.  The uncertainty consumed
by the policies is the one estimated out of sample by the model in
`forecast.py`, not the generator's parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

H = 8760

# --- Calibrated parameters --------------------------------------------------
# Demand level (MW) reproducing 1,592 GWh per year and a mean peak of 272 MW.
_DEMAND_LEVEL_MW = 211.357
# Target capacity factors, derived from the official 2025 data:
#   integrated = 17.1% x 1,592 GWh = 272.2 GWh;
#   available  = 272.2 / (1 - 0.1567) = 322.8 GWh
#   with 105.6 MW of wind and 61.2 MW of solar PV.
# Capacity factors derived from the official data; see
# `system.derive_capacity_factors`.  They are not free parameters.
_WIND_CF_TARGET = 0.2550
_PV_CF_TARGET = 0.1622
# Interannual variability of resource and demand.  Without it, the annual
# available energy would be identical in every realization and the campaign
# would measure only the effect of the intraday shape, of the outages, and
# of the forecast error.  The adopted dispersions — 6% for the wind index,
# 3% for the solar one, and 2% for demand — match the order of magnitude
# usually documented for annual resource indices at subtropical latitudes
# and are subjected to sensitivity analysis.
_WIND_INDEX_SD = 0.060
_PV_INDEX_SD = 0.030
_DEMAND_INDEX_SD = 0.020
# Noise of the forecast proxies, calibrated so that the error of the trained
# model sits in the range published for intraday and day-ahead horizons:
# demand ~2% of demand, wind ~12% of installed capacity,
# solar PV ~9% of installed capacity.
_NWP_WIND_NOISE = 0.130
_NWP_PV_NOISE = 0.095
_DEMAND_NOISE_SD = 0.030
_DEMAND_NOISE_PHI = 0.985


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """Vectorized stationary AR(1) with marginal variance sigma^2."""
    eps = rng.normal(0.0, sigma * np.sqrt(1.0 - phi ** 2), n)
    return lfilter([1.0], [1.0, -phi], eps)


def _circ(h: np.ndarray, c: float) -> np.ndarray:
    """Circular distance in hours between `h` and the center `c`.

    Audit note: version 1.0 of the simulator used `(hod - c) % 24`, which is
    not a circular distance and phase-shifted the intraday profile (hour 12
    was 23 units away from center 13).  The defect produced hourly ramps on
    the order of 25% of the peak and a lag-1 autocorrelation of 0.78, both
    incompatible with the audited temporal structure of the system.
    """
    d = np.abs(h - c) % 24.0
    return np.minimum(d, 24.0 - d)


def _rescale_to_mean(x: np.ndarray, target: float, hi: float):
    """Scale `x` to mean `target` while respecting the cap `hi`.

    Naive scaling followed by clipping biases the mean downward; the factor is
    solved by bisection on the already-clipped mean.
    """
    lo_k, hi_k = 1e-6, 80.0
    for _ in range(70):
        k = 0.5 * (lo_k + hi_k)
        if np.clip(x * k, 0.0, hi).mean() < target:
            lo_k = k
        else:
            hi_k = k
    k = 0.5 * (lo_k + hi_k)
    return np.clip(x * k, 0.0, hi), k


def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    ker = np.ones(k) / k
    pad = np.r_[x[:k][::-1], x, x[-k:][::-1]]
    return np.convolve(pad, ker, mode="same")[k:k + len(x)]


@dataclass
class World:
    """Complete annual realization, shared by all policies."""
    seed: int
    demand: np.ndarray
    demand_node: np.ndarray
    wind: np.ndarray
    pv: np.ndarray
    wind_node: np.ndarray
    pv_node: np.ndarray
    features: dict
    availability: np.ndarray
    regime_flag: np.ndarray
    true_event: np.ndarray
    link_available: np.ndarray
    meta: dict


def build_world(seed: int, cfg, *, error_scale: float = 1.0,
                error_scale_demand: float | None = None,
                error_scale_wind: float | None = None,
                error_scale_pv: float | None = None,
                detector_recall: float = 0.80,
                detector_fpr: float = 0.05,
                n_events: int = 18,
                event_depth: float = 0.45,
                link_outage_windows: tuple = (),
                unit_outage_windows: tuple = (),
                outage_rate: float = 0.04,
                cf_wind_target: float | None = None,
                cf_pv_target: float | None = None) -> World:
    """Generate one annual realization of the system.

    The `error_scale_*` factors scale the uncertainty of each source
    independently, a requirement of the experimental design (partial versus
    total sensitivity).
    """
    rng = np.random.default_rng(seed)
    # annual indices: introduce variability across realizations
    idx_w = float(np.exp(rng.normal(0.0, _WIND_INDEX_SD) - 0.5 * _WIND_INDEX_SD ** 2))
    idx_v = float(np.exp(rng.normal(0.0, _PV_INDEX_SD) - 0.5 * _PV_INDEX_SD ** 2))
    idx_d = float(1.0 + rng.normal(0.0, _DEMAND_INDEX_SD))
    t = np.arange(H)
    hod = (t % 24).astype(np.float64)
    doy = (t // 24).astype(np.float64)
    dow = (doy % 7).astype(np.float64)

    ks_d = error_scale if error_scale_demand is None else error_scale_demand
    ks_w = error_scale if error_scale_wind is None else error_scale_wind
    ks_p = error_scale if error_scale_pv is None else error_scale_pv

    # ------------------------------------------------------------------ demand
    season = 1.0 + 0.100 * np.sin(2 * np.pi * (doy - 205) / 365.0)
    daily = (0.800
             + 0.140 * np.exp(-(_circ(hod, 13.0) ** 2) / 18.0)
             + 0.300 * np.exp(-(_circ(hod, 20.0) ** 2) / 7.5)
             - 0.130 * np.exp(-(_circ(hod, 4.0) ** 2) / 14.0))
    weekend = np.where((dow == 5) | (dow == 6), 0.962, 1.0)
    noise_d = _ar1(rng, H, _DEMAND_NOISE_PHI, _DEMAND_NOISE_SD * ks_d)
    demand = np.clip(_DEMAND_LEVEL_MW * idx_d * season * daily * weekend * (1.0 + noise_d),
                     60.0, None)

    # ---------------------------------------------------- base wind resource
    lat_w = _ar1(rng, H, 0.9865, 1.0)
    cf_w_base = 1.0 / (1.0 + np.exp(-1.35 * lat_w - 0.10))
    cf_w_base = np.clip(cf_w_base, 0.0, 0.97)
    cf_w_base *= 1.0 + 0.10 * np.sin(2 * np.pi * (doy - 190) / 365.0)
    cf_w_base = np.clip(cf_w_base, 0.0, 0.97)

    # ------------------------------------------- unanticipated tail episodes
    true_event = np.zeros(H, bool)
    if n_events > 0:
        for s in rng.choice(H - 12, n_events, replace=False):
            true_event[s:s + 8] = True
    drop = np.where(true_event, 1.0 - float(np.clip(event_depth * ks_w, 0.0, 0.95)), 1.0)
    tgt_w = _WIND_CF_TARGET if cf_wind_target is None else cf_wind_target
    tgt_v = _PV_CF_TARGET if cf_pv_target is None else cf_pv_target
    cf_wind, k_w = _rescale_to_mean(cf_w_base * drop, tgt_w * idx_w, 0.97)
    cf_w_expect = np.clip(cf_w_base * k_w, 0.0, 0.97)   # event-free trajectory

    # ----------------------------------------------------- solar PV resource
    elev = np.clip(np.sin(np.pi * (hod - 6.75) / 12.5), 0.0, None)
    seas_pv = 0.90 + 0.16 * np.sin(2 * np.pi * (doy - 172) / 365.0)
    clear = np.clip(1.0 - np.abs(rng.normal(0.0, 0.16 * ks_p, H))
                    - 0.28 * (rng.random(H) < 0.075), 0.10, 1.0)
    cf_pv, _ = _rescale_to_mean(np.clip(elev * seas_pv * clear, 0.0, 1.0),
                                tgt_v * idx_v, 1.0)

    wind = cfg.wind_MW * cf_wind
    pv = cfg.pv_MW * cf_pv

    # ------------------------------------------------- observable predictors
    # The wind forecast proxy is built on the EXPECTED (event-free) trajectory:
    # the tail episode is, by definition, unanticipated.
    nwp_wind = np.clip(_smooth(cf_w_expect)
                       + rng.normal(0.0, _NWP_WIND_NOISE * ks_w, H), 0.0, 1.0)
    nwp_pv = np.clip(_smooth(cf_pv)
                     + rng.normal(0.0, _NWP_PV_NOISE * ks_p, H), 0.0, 1.0)
    nwp_temp = (20.0 + 5.0 * np.sin(2 * np.pi * (doy - 205) / 365.0)
                + 3.0 * np.sin(2 * np.pi * (hod - 15) / 24.0)
                + rng.normal(0.0, 1.1, H))

    lag24_d = np.r_[demand[:24], demand[:-24]]
    lag168_d = np.r_[demand[:168], demand[:-168]]
    lag24_w = np.r_[cf_wind[:24], cf_wind[:-24]]

    flag = ((true_event & (rng.random(H) < detector_recall))
            | (~true_event & (rng.random(H) < detector_fpr)))

    features = {
        "hod": hod, "doy": doy, "dow": dow,
        "sin_hod": np.sin(2 * np.pi * hod / 24.0),
        "cos_hod": np.cos(2 * np.pi * hod / 24.0),
        "sin_doy": np.sin(2 * np.pi * doy / 365.0),
        "cos_doy": np.cos(2 * np.pi * doy / 365.0),
        "weekend": (weekend < 1.0).astype(np.float64),
        "lag24_demand": lag24_d, "lag168_demand": lag168_d,
        "lag24_cf_wind": lag24_w,
        "nwp_wind": nwp_wind, "nwp_pv": nwp_pv, "nwp_temp": nwp_temp,
        "elevation": elev,
        "regime_flag": flag.astype(np.float64),
        "wind_MW": np.full(H, cfg.wind_MW),
        "pv_MW": np.full(H, cfg.pv_MW),
    }

    # ---------------------------------------------------- per-island split
    from .system import DEMAND_SHARE, RENEWABLE_SPLIT_2025
    demand_node = np.column_stack([demand * DEMAND_SHARE["LZ"],
                                   demand * DEMAND_SHARE["FV"]])
    w = RENEWABLE_SPLIT_2025["wind"]
    p = RENEWABLE_SPLIT_2025["pv"]
    fw_lz = w["LZ"] / (w["LZ"] + w["FV"])
    fp_lz = p["LZ"] / (p["LZ"] + p["FV"])
    wind_node = np.column_stack([wind * fw_lz, wind * (1.0 - fw_lz)])
    pv_node = np.column_stack([pv * fp_lz, pv * (1.0 - fp_lz)])

    # --------------------------------------------------- forced outages
    nu = len(cfg.fleet)
    avail = np.ones((H, nu), bool)
    for u in range(nu):
        remaining = outage_rate * H * rng.gamma(4.0, 0.25)
        while remaining > 0:
            dur = int(np.clip(rng.gamma(2.0, 20.0), 6, 400))
            s = int(rng.integers(0, H - 1))
            avail[s:min(s + dur, H), u] = False
            remaining -= dur
    for (a, b, idx) in unit_outage_windows:
        avail[a:b, idx] = False

    link_available = np.ones(H, bool)
    for (a, b) in link_outage_windows:
        link_available[a:b] = False

    meta = dict(demand_GWh=float(demand.sum() / 1e3),
                peak_MW=float(demand.max()),
                wind_GWh=float(wind.sum() / 1e3),
                pv_GWh=float(pv.sum() / 1e3),
                res_avail_GWh=float((wind + pv).sum() / 1e3),
                cf_wind=float(cf_wind.mean()), cf_pv=float(cf_pv.mean()),
                event_hours=int(true_event.sum()),
                wind_index=idx_w, pv_index=idx_v, demand_index=idx_d,
                outage_rate=float(1.0 - avail.mean()),
                error_scale=(ks_d, ks_w, ks_p))

    return World(seed=seed, demand=demand, demand_node=demand_node,
                 wind=wind, pv=pv, wind_node=wind_node, pv_node=pv_node,
                 features=features, availability=avail, regime_flag=flag,
                 true_event=true_event, link_available=link_available, meta=meta)


def calibration_report(worlds, official: dict) -> dict:
    """Goodness of fit of the generator against the official magnitudes.

    Accepts one realization or a list; with several it reports mean and
    standard deviation across realizations.
    """
    if not isinstance(worlds, (list, tuple)):
        worlds = [worlds]
    acc: dict = {}

    def push(k, v):
        acc.setdefault(k, []).append(float(v))

    for w in worlds:
        d = w.demand
        ldc = np.sort(d)[::-1]
        ramps = np.abs(np.diff(d))
        res = w.wind + w.pv
        push("demanda_anual_GWh", d.sum() / 1e3)
        push("punta_MW", d.max())
        push("minimo_MW", d.min())
        push("factor_de_carga", d.mean() / d.max())
        push("P10_curva_monotona_MW", ldc[int(0.10 * H)])
        push("P50_curva_monotona_MW", ldc[int(0.50 * H)])
        push("P90_curva_monotona_MW", ldc[int(0.90 * H)])
        push("rampa_horaria_p99_MW", np.percentile(ramps, 99))
        push("rampa_p99_pct_punta", 100 * np.percentile(ramps, 99) / d.max())
        push("autocorrelacion_lag1", np.corrcoef(d[:-1], d[1:])[0, 1])
        push("autocorrelacion_lag24", np.corrcoef(d[:-24], d[24:])[0, 1])
        push("producible_renovable_GWh", res.sum() / 1e3)
        push("cf_eolico", w.meta["cf_wind"])
        push("cf_fotovoltaico", w.meta["cf_pv"])
        push("indisponibilidad_media", w.meta["outage_rate"])
        push("cuota_renovable_max_pct", 100.0 * res.sum() / d.sum())

    targets = {
        "demanda_anual_GWh": official["demand_GWh"],
        "punta_MW": official["peak_MW"],
        # theoretical maximum renewable share of the model: the generator does not
        # represent the internal-grid or stability constraints that cause the
        # observed curtailment, so its share exceeds the official one
        "cuota_renovable_max_pct": official["renewable_share_pct"],
        "producible_renovable_GWh": (official["renewable_share_pct"] / 100.0
                                     * official["demand_GWh"]
                                     / (1.0 - official["curtailment_pct"] / 100.0)),
    }
    out = {}
    for k, v in acc.items():
        arr = np.asarray(v)
        tgt = targets.get(k)
        out[k] = dict(media=float(arr.mean()),
                      desv=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                      objetivo=tgt,
                      error_rel_pct=(None if tgt is None
                                     else float(100.0 * (arr.mean() - tgt) / tgt)))
    return out
