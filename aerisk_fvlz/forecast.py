"""
forecast.py — Probabilistic forecasting layer, learned and validated out of sample.

Methodological motivation
-------------------------
In an evaluation of dispatch policies it is tempting to feed the policy with
the very standard deviation the generator used to construct the error.  That
shortcut is circular: it grants the operator an exact knowledge of its own
uncertainty that does not exist in operation and systematically overstates the
performance of any robust policy.

Here the uncertainty consumed by the policies comes from a quantile regression
model trained on realizations DIFFERENT from those used in the experimental
campaign (a per-realization split, analogous to a temporal split) and
evaluated out of sample through pinball loss, empirical coverage (PICP), and
normalized width (PINAW), against a climatology-persistence reference.  The
predicted quantiles, not the generator's parameters, define the robust
setpoint and the gate risk.

The component is, therefore, a learned, trained, and validated statistical
model; it is described as such and no capability is attributed to it that has
not been measured.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

QUANTILES = (0.10, 0.50, 0.90)

# Width threshold below which an hour is considered DEGENERATE and is
# excluded from the conformal calibration and from the coverage computation.
# Solar PV output is identically zero at night: including those hours
# would make the conformal score percentile fall to values that
# spuriously narrow the daytime interval.  The coverage guarantee is
# therefore stated conditionally on the hours with possible production.
_ACTIVE_WIDTH = {"demand": 1e-6, "cf_wind": 1e-4, "cf_pv": 5e-3}
Z90 = 1.2815515655446004  # 0.90 quantile of the standard normal

FEATURE_NAMES = [
    "sin_hod", "cos_hod", "sin_doy", "cos_doy", "weekend", "hod",
    "lag24_demand", "lag168_demand", "lag24_cf_wind",
    "nwp_wind", "nwp_pv", "nwp_temp", "elevation", "regime_flag",
]


def _design_matrix(features: dict) -> np.ndarray:
    return np.column_stack([features[k] for k in FEATURE_NAMES])


@dataclass
class QuantileForecaster:
    """Three targets (demand MW, wind capacity factor, solar PV capacity
    factor) x three quantiles, with conformal recalibration."""
    models: dict
    train_seeds: tuple
    val_report: dict
    conformal: dict = None            # multiplicative factor per target
    val_report_raw: dict = None
    test_report: dict = None          # evaluation on a THIRD set
    test_report_raw: dict = None

    # -- fit ------------------------------------------------------------
    @classmethod
    def fit(cls, worlds_train, worlds_val, worlds_test=None, *, max_iter: int = 220,
            learning_rate: float = 0.08, max_depth: int | None = 6,
            random_state: int = 0) -> "QuantileForecaster":
        Xtr = np.vstack([_design_matrix(w.features) for w in worlds_train])
        targets_tr = {
            "demand": np.concatenate([w.demand for w in worlds_train]),
            "cf_wind": np.concatenate([w.wind / max(w.features["wind_MW"][0], 1e-9)
                                       for w in worlds_train]),
            "cf_pv": np.concatenate([w.pv / max(w.features["pv_MW"][0], 1e-9)
                                     for w in worlds_train]),
        }
        models = {}
        for tgt, y in targets_tr.items():
            for q in QUANTILES:
                m = HistGradientBoostingRegressor(
                    loss="quantile", quantile=q, max_iter=max_iter,
                    learning_rate=learning_rate, max_depth=max_depth,
                    min_samples_leaf=40, l2_regularization=1.0,
                    early_stopping=False, random_state=random_state)
                m.fit(Xtr, y)
                models[(tgt, q)] = m
        obj = cls(models=models,
                  train_seeds=tuple(w.seed for w in worlds_train),
                  val_report={},
                  conformal={t: 1.0 for t in ("demand", "cf_wind", "cf_pv")})
        obj.val_report_raw = obj.evaluate(worlds_val)
        obj.calibrate(worlds_val)
        obj.val_report = obj.evaluate(worlds_val, conformal=True)
        # Coverage on the calibration set is not a measurement: the conformal
        # procedure fixes it at the nominal level by construction.
        # The only informative coverage is the one obtained on a third set,
        # disjoint from the training and the calibration ones.
        if worlds_test:
            obj.test_report = obj.evaluate(worlds_test, conformal=True)
            obj.test_report_raw = obj.evaluate(worlds_test, conformal=False)
        return obj

    # -- split conformal recalibration (Romano et al., 2019) --------------
    def calibrate(self, worlds_cal, nominal: float = 0.80) -> None:
        """Fit a per-target factor lambda so that the interval
        [q50 - lambda (q50-q10), q50 + lambda (q90-q50)] reaches the nominal
        coverage on a calibration set disjoint from the training one.

        This is conformalized quantile regression in its multiplicative
        version: the marginal coverage guarantee ceases to depend on the model
        being well specified.
        """
        for tgt, getter in _TARGETS:
            scores = []
            for w in worlds_cal:
                X = _design_matrix(w.features)
                y = getter(w)
                q10 = self.models[(tgt, 0.10)].predict(X)
                q50 = self.models[(tgt, 0.50)].predict(X)
                q90 = self.models[(tgt, 0.90)].predict(X)
                q10 = np.minimum(q10, q50)
                q90 = np.maximum(q90, q50)
                width = q90 - q10
                active = width > _ACTIVE_WIDTH.get(tgt, 1e-9)
                lo_w = np.maximum(q50 - q10, 1e-9)
                hi_w = np.maximum(q90 - q50, 1e-9)
                sc = np.maximum((q50 - y) / lo_w, (y - q50) / hi_w)
                scores.append(sc[active])
            s = np.concatenate(scores)
            if s.size < 50:      # without enough active hours, no recalibration
                self.conformal[tgt] = 1.0
                continue
            n = s.size
            k = min(n - 1, int(np.ceil((n + 1) * nominal)) - 1)
            self.conformal[tgt] = float(max(np.sort(s)[k], 1e-6))

    # -- prediction --------------------------------------------------------
    def predict(self, world, conformal: bool = True) -> dict:
        """Return recalibrated quantiles and implied standard deviation."""
        X = _design_matrix(world.features)
        wcap = float(world.features["wind_MW"][0])
        pcap = float(world.features["pv_MW"][0])
        out = {}
        scales = {"demand": 1.0, "cf_wind": wcap, "cf_pv": pcap}
        for tgt, scale in scales.items():
            q10 = self.models[(tgt, 0.10)].predict(X)
            q50 = self.models[(tgt, 0.50)].predict(X)
            q90 = self.models[(tgt, 0.90)].predict(X)
            q10 = np.minimum(q10, q50)
            q90 = np.maximum(q90, q50)
            if conformal and self.conformal:
                lam = self.conformal.get(tgt, 1.0)
                q10 = q50 - lam * (q50 - q10)
                q90 = q50 + lam * (q90 - q50)
            q10 = np.maximum(q10, 0.0) * scale
            q50 = np.maximum(q50, 0.0) * scale
            q90 = np.maximum(q90, 0.0) * scale
            sigma = np.maximum((q90 - q10) / (2.0 * Z90), 1e-6)
            key = {"demand": "D", "cf_wind": "W", "cf_pv": "V"}[tgt]
            out[key] = dict(q10=q10, q50=q50, q90=q90, sigma=sigma)
        return out

    # -- out-of-sample evaluation ---------------------------------------
    def evaluate(self, worlds, conformal: bool = False) -> dict:
        rep = {}
        for tgt, getter in _TARGETS:
            pin, cov, wid, base_pin, ys = [], [], [], [], []
            for w in worlds:
                X = _design_matrix(w.features)
                y = getter(w)
                qs = {q: self.models[(tgt, q)].predict(X) for q in QUANTILES}
                q10 = np.minimum(qs[0.10], qs[0.50])
                q90 = np.maximum(qs[0.90], qs[0.50])
                if conformal and self.conformal:
                    lam = self.conformal.get(tgt, 1.0)
                    q10 = qs[0.50] - lam * (qs[0.50] - q10)
                    q90 = qs[0.50] + lam * (q90 - qs[0.50])
                    qs = {0.10: q10, 0.50: qs[0.50], 0.90: q90}
                pin.append(np.mean([_pinball(y, qs[q], q) for q in QUANTILES]))
                act = (q90 - q10) > _ACTIVE_WIDTH.get(tgt, 1e-9)
                if act.sum() < 50:
                    act = np.ones_like(act, dtype=bool)
                cov.append(float(np.mean((y[act] >= q10[act]) & (y[act] <= q90[act]))))
                rng_y = np.ptp(y) if np.ptp(y) > 0 else 1.0
                wid.append(float(np.mean(q90 - q10) / rng_y))
                base_pin.append(_baseline_pinball(y, w))
                ys.append(y)
            rep[tgt] = dict(
                lambda_conformal=float(self.conformal.get(tgt, 1.0)) if self.conformal else 1.0,
                pinball=float(np.mean(pin)),
                pinball_baseline=float(np.mean(base_pin)),
                skill_vs_baseline=float(1.0 - np.mean(pin) / np.mean(base_pin)),
                picp90=float(np.mean(cov)),
                pinaw=float(np.mean(wid)),
                n_worlds=len(worlds),
            )
        return rep

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> "QuantileForecaster":
        with open(path, "rb") as fh:
            return pickle.load(fh)


_TARGETS = (
    ("demand", lambda w: w.demand),
    ("cf_wind", lambda w: w.wind / max(w.features["wind_MW"][0], 1e-9)),
    ("cf_pv", lambda w: w.pv / max(w.features["pv_MW"][0], 1e-9)),
)


def _pinball(y: np.ndarray, yhat: np.ndarray, q: float) -> float:
    d = y - yhat
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


def _baseline_pinball(y: np.ndarray, world) -> float:
    """Climatology-persistence reference: per-hour-of-day mean plus Gaussian
    quantiles with the empirical standard deviation of the residual."""
    hod = world.features["hod"].astype(int)
    clim = np.zeros_like(y)
    for h in range(24):
        m = hod == h
        clim[m] = y[m].mean()
    resid = y - clim
    s = resid.std()
    vals = []
    for q in QUANTILES:
        z = {0.10: -Z90, 0.50: 0.0, 0.90: Z90}[q]
        vals.append(_pinball(y, clim + z * s, q))
    return float(np.mean(vals))
