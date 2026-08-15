"""
benchmark.py — Validation of the commitment heuristic against the optimal commitment.

Commitment is solved in `dispatch.py` by a greedy heuristic over the merit
order.  The pertinent objection is that the asymmetric ratchet could be
compensating for the myopia of that heuristic with respect to minimum down
times, rather than reflecting a genuine operational phenomenon.  To rule this
out, the same unit commitment problem is solved EXACTLY, as a mixed-integer
program, over representative weeks, and the resulting cost is compared.

Formulation (per week, 168 h horizon, 16 units):

    min  sum_t sum_g [ c_g p_gt + k_g s_gt ]

    s.t. p_gt <= Pmax_g u_gt                          upper limit
         p_gt >= Pmin_g u_gt                          technical minimum
         sum_g p_gt >= N_t                            balance with admissible curtailment
         sum_g Pmax_g u_gt - Pmax_k u_kt >= N_t   ∀k  linearized N-1 criterion
         p_gt - p_g,t-1 <= R_g ;  p_g,t-1 - p_gt <= R_g   ramps
         s_gt >= u_gt - u_g,t-1                       start-up
         sum_{tau=t-MU+1}^{t} s_g,tau <= u_gt         minimum up time
         sum_{tau=t-MD+1}^{t} (u_g,tau-1 - u_g,tau) <= 1 - u_gt   minimum down time

The linearization of the N-1 criterion (one constraint per unit and hour) is
exact, not relaxed: it requires that, after losing any committed unit, the
remaining capacity cover the setpoint.  The heuristic imposes the same
condition.

The comparison is carried out on a single-node system, with the transfer limit
relaxed in the heuristic as well, so that the only difference between the two
is the temporal decision procedure.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import lil_matrix

from .dispatch import Z90


def solve_week(cfg, n_target: np.ndarray, avail: np.ndarray, *,
               time_limit: float = 120.0, mip_gap: float = 0.005):
    """Solve the optimal commitment for a window of `T` hours.

    Parameters
    ----------
    n_target : hourly net setpoint (MW) that the commitment must cover.
    avail    : (T, G) matrix of unit availability.
    """
    p = cfg.unit_params()
    G = len(p["pmax"])
    T = len(n_target)
    pmax, pmin, cvar = p["pmax"], p["pmin"], p["cvar"]
    ramp, mu, md, sc = p["ramp"], p["min_up"], p["min_down"], p["start_cost"]

    nU, nP, nS = G * T, G * T, G * T
    N = nU + nP + nS
    iU = lambda g, t: g * T + t
    iP = lambda g, t: nU + g * T + t
    iS = lambda g, t: nU + nP + g * T + t

    c = np.zeros(N)
    for g in range(G):
        for t in range(T):
            c[iP(g, t)] = cvar[g]
            c[iS(g, t)] = sc[g]

    integrality = np.zeros(N)
    integrality[:nU] = 1
    lb, ub = np.zeros(N), np.zeros(N)
    for g in range(G):
        for t in range(T):
            ok = bool(avail[t, g])
            lb[iU(g, t)], ub[iU(g, t)] = 0.0, (1.0 if ok else 0.0)
            lb[iP(g, t)], ub[iP(g, t)] = 0.0, (pmax[g] if ok else 0.0)
            lb[iS(g, t)], ub[iS(g, t)] = 0.0, 1.0

    rows, lo, hi = [], [], []

    def add(coeffs, l, h):
        rows.append(coeffs)
        lo.append(l)
        hi.append(h)

    for t in range(T):
        for g in range(G):
            add({iP(g, t): 1.0, iU(g, t): -pmax[g]}, -np.inf, 0.0)   # p <= Pmax u
            add({iP(g, t): 1.0, iU(g, t): -pmin[g]}, 0.0, np.inf)    # p >= Pmin u
        # the balance is imposed as an inequality: when the technical minima of
        # the units the N-1 criterion forces to commit exceed the setpoint,
        # the excess is absorbed by curtailing renewable generation, exactly as
        # the hourly simulator does.  With strict equality the program becomes
        # infeasible in the high-penetration, low-net-load hours.
        add({iP(g, t): 1.0 for g in range(G)}, n_target[t], np.inf)
        for k in range(G):                                            # N-1
            co = {iU(g, t): pmax[g] for g in range(G)}
            co[iU(k, t)] = pmax[k] - pmax[k]
            add(co, n_target[t], np.inf)
        if t > 0:
            for g in range(G):
                add({iP(g, t): 1.0, iP(g, t - 1): -1.0}, -ramp[g], ramp[g])
                add({iS(g, t): 1.0, iU(g, t): -1.0, iU(g, t - 1): 1.0}, 0.0, np.inf)
        else:
            for g in range(G):
                add({iS(g, t): 1.0, iU(g, t): -1.0}, 0.0, np.inf)
    for g in range(G):                                                # minimum up time
        for t in range(T):
            span = range(max(0, t - mu[g] + 1), t + 1)
            co = {iS(g, tau): 1.0 for tau in span}
            co[iU(g, t)] = co.get(iU(g, t), 0.0) - 1.0
            add(co, -np.inf, 0.0)
    for g in range(G):                                                # minimum down time
        for t in range(1, T):
            span = range(max(1, t - md[g] + 1), t + 1)
            co = {}
            for tau in span:
                co[iU(g, tau - 1)] = co.get(iU(g, tau - 1), 0.0) + 1.0
                co[iU(g, tau)] = co.get(iU(g, tau), 0.0) - 1.0
            co[iU(g, t)] = co.get(iU(g, t), 0.0) + 1.0
            add(co, -np.inf, 1.0)

    A = lil_matrix((len(rows), N))
    for i, co in enumerate(rows):
        for j, v in co.items():
            A[i, j] = v
    con = LinearConstraint(A.tocsr(), np.array(lo), np.array(hi))

    res = milp(c=c, constraints=con, integrality=integrality,
               bounds=(lb, ub),
               options={"time_limit": time_limit, "mip_rel_gap": mip_gap,
                        "presolve": True})
    if res.x is None:
        return None
    x = res.x
    fuel = sum(cvar[g] * x[iP(g, t)] for g in range(G) for t in range(T))
    start = sum(sc[g] * x[iS(g, t)] for g in range(G) for t in range(T))
    return dict(status=int(res.status), cost=float(fuel + start),
                fuel=float(fuel), start=float(start),
                gap=float(getattr(res, "mip_gap", np.nan)),
                commitments=int(round(sum(x[iU(g, t)] for g in range(G)
                                          for t in range(T)))))


def heuristic_week(cfg, n_target: np.ndarray, avail: np.ndarray):
    """Cost of the greedy heuristic on the same window and setpoint.

    Reproduces the commitment and release loop of `dispatch.simulate` without
    the real-time layer, so that the comparison isolates exclusively the
    quality of the commitment decision.
    """
    p = cfg.unit_params()
    G = len(p["pmax"])
    T = len(n_target)
    pmax, pmin, cvar = p["pmax"], p["pmin"], p["cvar"]
    ramp, mu, md, sc = p["ramp"], p["min_up"], p["min_down"], p["start_cost"]
    merit = sorted(range(G), key=lambda g: cvar[g])

    on = [False] * G
    up_t, dn_t = [0] * G, [10] * G
    p_prev = [0.0] * G
    fuel = start = 0.0
    for t in range(T):
        av = avail[t]
        need = float(n_target[t])
        # commitment with N-1 criterion recomputed to a fixed point
        guard = 0
        while True:
            cap = sum(pmax[g] for g in range(G) if on[g] and av[g])
            big = max((pmax[g] for g in range(G) if on[g] and av[g]), default=0.0)
            if cap - big >= need - 1e-6:
                break
            guard += 1
            if guard > G + 2:
                break
            cand = next((g for g in merit
                         if not on[g] and av[g] and dn_t[g] >= md[g]), -1)
            if cand < 0:
                break
            on[cand] = True
            up_t[cand] = 0
            dn_t[cand] = 0
            start += sc[cand]
        # release
        for g in reversed(merit):
            if not (on[g] and av[g]) or up_t[g] < mu[g]:
                continue
            if sum(1 for k in range(G) if on[k] and av[k]) <= 2:
                break
            cap2 = sum(pmax[k] for k in range(G) if on[k] and av[k] and k != g)
            big2 = max((pmax[k] for k in range(G)
                        if on[k] and av[k] and k != g), default=0.0)
            if cap2 - big2 >= need - 1e-6 and cap2 >= need:
                on[g] = False
                up_t[g] = 0
                dn_t[g] = 0
        # economic dispatch with technical minima and ramps
        out = [0.0] * G
        rem = need
        for g in range(G):
            if on[g] and av[g]:
                out[g] = max(pmin[g], p_prev[g] - ramp[g])
                rem -= out[g]
        if rem > 0:
            for g in merit:
                if rem <= 1e-9 or not (on[g] and av[g]):
                    continue
                hi = min(pmax[g], p_prev[g] + ramp[g])
                add = max(0.0, min(hi - out[g], rem))
                out[g] += add
                rem -= add
        elif rem < 0:
            for g in reversed(merit):
                if rem >= -1e-9 or not (on[g] and av[g]):
                    continue
                red = min(out[g] - pmin[g], -rem)
                if red > 0:
                    out[g] -= red
                    rem += red
        for g in range(G):
            fuel += cvar[g] * out[g]
            if on[g] and av[g]:
                up_t[g] += 1
                dn_t[g] = 0
                p_prev[g] = out[g]
            else:
                dn_t[g] += 1
                up_t[g] = 0
                p_prev[g] = 0.0
    return dict(cost=float(fuel + start), fuel=float(fuel), start=float(start))


def run_benchmark(cfg, forecaster, seeds=(1,), weeks=(6, 19, 32, 45),
                  beta: float = Z90, time_limit: float = 120.0) -> list:
    """Compare heuristic and optimum over representative weeks of the year."""
    from .world import build_world
    rows = []
    for seed in seeds:
        w = build_world(seed, cfg)
        f = forecaster.predict(w)
        nhat = np.maximum(f["D"]["q50"] - f["W"]["q50"] - f["V"]["q50"], 0.0)
        sg = np.sqrt(f["D"]["sigma"] ** 2 + f["W"]["sigma"] ** 2 + f["V"]["sigma"] ** 2)
        for wk in weeks:
            a, b = wk * 168, wk * 168 + 168
            for label, tgt in (("consigna determinista", nhat[a:b]),
                               ("consigna robusta", (nhat + beta * sg)[a:b])):
                av = w.availability[a:b]
                h = heuristic_week(cfg, tgt, av)
                o = solve_week(cfg, tgt, av, time_limit=time_limit)
                if o is None:
                    continue
                rows.append(dict(seed=seed, semana=wk, consigna=label,
                                 coste_heuristico=round(h["cost"] / 1e3, 2),
                                 coste_optimo=round(o["cost"] / 1e3, 2),
                                 brecha_pct=round(100 * (h["cost"] - o["cost"])
                                                  / o["cost"], 3),
                                 estado=o["status"]))
    return rows
