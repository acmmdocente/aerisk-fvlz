"""
dispatch.py — Hourly commitment and dispatch simulator of the FV-LZ system.

Substantive corrections with respect to version 1.0 of the simulator
--------------------------------------------------------------------
(1) INDEPENDENT POLICIES.  Each policy builds its own commitment trajectory
    from the same state and the same realizations (common random numbers).
    The `floor_on` mechanism has been removed; it forced the supervised
    policy's commitment to contain that of the previously computed automatic
    policy and imputed zero start-up cost to the units so forced.

(2) EVERY START-UP IS BILLED.  Start-up cost is accrued by comparing the
    resulting committed set with the hour's starting set; no free commitment
    path exists.

(3) N-1 CRITERION RECOMPUTED TO CONVERGENCE.  The reserve requirement depends
    on the largest committed unit; since every start-up can change it, it is
    recomputed at each iteration of the commitment loop until a fixed point is
    reached or infeasibility is declared.

(4) NON-RELAXATION, STRUCTURAL.  The gate audits the AUTOMATIC RECOMMENDATION:
    from the same state, the automatic commitment U_A(t) is built first; if
    the residual risk of that recommendation exceeds the threshold, the
    commitment is rebuilt with the robust requirement.  Since the commitment
    procedure is monotone non-decreasing in the net requirement, U_B(t)
    contains U_A(t) by construction, not by external imposition.  Moreover,
    the RELEASE of capacity is always evaluated with the robust requirement:
    the safeguard does not give back to the system capacity that the robust
    criterion still demands.  Without this asymmetry, minimum down times
    prevent the released capacity from being recovered and the selective
    safeguard ends up dominated by permanent robustness.

(5) CONSTRAINTS DECLARED AND IMPLEMENTED.  Technical minima, ramps, minimum up
    and down times, link transfer limit, and reserve deliverability after
    contingency are implemented; no constraint is stated that the code does
    not apply.

Policies
--------
    A  automatic deterministic     requirement = N_hat
    B  selectively safeguarded     requirement = N_hat + beta sigma  if gamma_t = 1
    C  permanently robust          requirement = N_hat + beta sigma  always
    D  permanently conservative    requirement = N_hat + beta sigma + 5% D_hat

    Alphabetical order coincides with increasing conservatism.

    A versus B  measures the effect of the safeguard;
    B versus C  isolates the value of SELECTIVITY (same robustness, applied
                only where the residual risk justifies it);
    C versus D  isolates the permanent additional margin.

Reference policies taken from the literature
--------------------------------------------
    E  static reserve rule         requirement = N_hat + a D_hat + b W_hat
       Deterministic percentage criterion, widely used in island system
       operation: largest committed unit plus one percentage of forecast
       demand and another of forecast renewable generation.
    F  robust with uncertainty budget (Bertsimas et al.)
       Linear aggregation with budget Gamma instead of quadratic composition:
       the worst case assigns the budget to the sources of largest standard
       deviation, without assuming independence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

Z90 = 1.2815515655446004
SQRT2 = 1.4142135623730951
H = 8760
MODES = ("A", "B", "C", "D", "E", "F")


@dataclass
class GateConfig:
    """Parameters of the rule-based risk gate."""
    rho_on: float = 0.30      # activation threshold on the residual risk
    rho_off: float = 0.12     # deactivation threshold (Schmitt trigger)
    dwell_h: int = 4          # minimum dwell after activation (h)
    use_risk: bool = True     # quantitative condition (risk equation)
    use_flag: bool = True     # regime-detector flag
    beta: float = Z90         # robustness level of the setpoint
    conservative_extra: float = 0.05   # permanent additional margin of mode D
    static_load_frac: float = 0.03     # mode E: percentage of demand
    static_wind_frac: float = 0.10     # mode E: percentage of renewables
    budget_gamma: float = 2.0          # mode F: uncertainty budget


@dataclass
class RunResult:
    mode: str
    cost_MEUR: float = 0.0
    fuel_cost_MEUR: float = 0.0
    start_cost_MEUR: float = 0.0
    ens_cost_MEUR: float = 0.0
    ens_MWh: float = 0.0
    curtailment_GWh: float = 0.0
    co2_kt: float = 0.0
    violation_h: int = 0
    ens_h: int = 0
    gate_h: int = 0
    starts: int = 0
    res_share_pct: float = 0.0
    infeasible_h: int = 0
    link_binding_h: int = 0
    avoided_adverse_h: int = 0
    false_alarm_h: int = 0
    gate_precision_pct: float = float("nan")
    non_relaxation_violations: int = 0
    traces: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "traces"}


def _security_ok(cap_l: float, cap_f: float, big_l: float, big_f: float,
                 net_l: float, net_f: float, flim: float) -> bool:
    """Single-contingency coverage with deliverability at two nodes.

    Requires (i) per-node balance coverage within the transfer limit;
    (ii) aggregate coverage after the loss of the largest committed unit; and
    (iii) per-node coverage after the loss of that node's largest unit,
    admitting support from the other node only up to the link limit.
    """
    eps = 1e-6
    if cap_l + flim + eps < net_l:
        return False
    if cap_f + flim + eps < net_f:
        return False
    big = big_l if big_l >= big_f else big_f
    if cap_l + cap_f - big + eps < net_l + net_f:
        return False
    if cap_l - big_l + flim + eps < net_l:
        return False
    if cap_f - big_f + flim + eps < net_f:
        return False
    return True


def simulate(world, cfg, forecast, mode: str, gate: GateConfig | None = None,
             *, tau_rescue: float = 0.30, keep_traces: bool = False,
             min_online: int = 2, oracle_sigma: bool = False,
             check_non_relaxation: bool = False,
             reference_adverse: np.ndarray | None = None) -> RunResult:
    """Run one dispatch policy on one realization.

    `reference_adverse` is the hourly indicator of adverse hours under the
    automatic policy; it is used EXCLUSIVELY a posteriori to classify gate
    activations into hits and false alarms, and it takes part in no decision.
    """
    if gate is None:
        gate = GateConfig()
    if mode not in MODES:
        raise ValueError(f"modo desconocido: {mode}")

    p = cfg.unit_params()
    nu = len(p["pmax"])
    pmax, pmin, cvar, ef = p["pmax"], p["pmin"], p["cvar"], p["ef"]
    ramp, mnup, mndn = p["ramp"], p["min_up"], p["min_down"]
    scost, fast, node = p["start_cost"], p["fast"], p["node"]
    merit = sorted(range(nu), key=lambda u: cvar[u])
    merit_rev = merit[::-1]
    voll = cfg.voll_EUR_MWh
    flim_nom = cfg.topology.link_capacity_MW
    flim_bkp = cfg.topology.link_capacity_outage_MW

    D, Wr, Vr = world.demand, world.wind, world.pv
    Dn, Wn, Vn = world.demand_node, world.wind_node, world.pv_node

    fD, fW, fV = forecast["D"], forecast["W"], forecast["V"]
    if oracle_sigma:
        sD = np.abs(D - fD["q50"]) * 1.2533
        sW = np.abs(Wr - fW["q50"]) * 1.2533
        sV = np.abs(Vr - fV["q50"]) * 1.2533
    else:
        sD, sW, sV = fD["sigma"], fW["sigma"], fV["sigma"]
    sigma_agg = np.sqrt(sD ** 2 + sW ** 2 + sV ** 2)
    nhat_arr = fD["q50"] - fW["q50"] - fV["q50"]

    # flat lists: in loops of 8,760 iterations over 16 units, numpy
    # operations are dominated by call overhead.
    D_l = D.tolist()
    Dl_l, Df_l = Dn[:, 0].tolist(), Dn[:, 1].tolist()
    Wl_l, Wf_l = Wn[:, 0].tolist(), Wn[:, 1].tolist()
    Vl_l, Vf_l = Vn[:, 0].tolist(), Vn[:, 1].tolist()
    nhat_l, sag_l = nhat_arr.tolist(), sigma_agg.tolist()
    hatD_l = fD["q50"].tolist()
    hatR_l = (fW["q50"] + fV["q50"]).tolist()
    sD_l, sW_l, sV_l = sD.tolist(), sW.tolist(), sV.tolist()
    flag_l = world.regime_flag.tolist()
    link_l = world.link_available.tolist()
    avail_l = world.availability.tolist()
    share_l = (Dn[:, 0] / np.maximum(D, 1e-9)).tolist()

    # ------------------------------------------------------------- initial state
    on = [False] * nu
    for u in merit[:5]:
        on[u] = True
    up_t = [10 if on[u] else 0 for u in range(nu)]
    dn_t = [0 if on[u] else 10 for u in range(nu)]
    p_prev = [pmin[u] if on[u] else 0.0 for u in range(nu)]

    fuel_c = start_c = ens_c = co2 = 0.0
    ens_tot = curt = 0.0
    viol = ens_h = gate_h = starts = infeas = link_bind = nr_viol = 0
    gate_tr = np.zeros(H, bool)
    adverse_tr = np.zeros(H, bool)
    rho_tr = np.zeros(H)
    fail_kind = [0, 0, 0]

    beta_mode = gate.beta
    hold = 0
    armed = False

    # ---------------------------------------------------------------------
    def commit(base_on, av, n_tot, s_l, flim):
        """Minimal merit-order commitment satisfying the security criterion
        for requirement `n_tot`.  Recomputes the largest unit at each
        iteration (fixed point of the N-1 criterion).  Returns the resulting
        set and an infeasibility indicator.

        Monotonicity: if n1 <= n2, the set returned for n1 is contained in the
        one returned for n2 starting from the same `base_on`, because the loop
        always traverses the same merit order and only adds units.
        """
        res_on = base_on[:]
        cl = cf = bl = bf = 0.0
        for u in range(nu):
            if res_on[u] and av[u]:
                if node[u] == 0:
                    cl += pmax[u]
                    if pmax[u] > bl:
                        bl = pmax[u]
                else:
                    cf += pmax[u]
                    if pmax[u] > bf:
                        bf = pmax[u]
        nl = n_tot * s_l
        nf = n_tot - nl
        bad = 0
        guard = 0
        while not _security_ok(cl, cf, bl, bf, nl, nf, flim):
            guard += 1
            if guard > nu + 2:
                bad = 1
                break
            def_l = nl - (cl - bl + flim)
            def_f = nf - (cf - bf + flim)
            prefer = 0 if def_l >= def_f else 1
            cand = -1
            for u in merit:
                if (not res_on[u]) and av[u] and dn_t[u] >= mndn[u] and node[u] == prefer:
                    cand = u
                    break
            if cand < 0:
                for u in merit:
                    if (not res_on[u]) and av[u] and dn_t[u] >= mndn[u]:
                        cand = u
                        break
            if cand < 0:
                bad = 1
                break
            res_on[cand] = True
            if node[cand] == 0:
                cl += pmax[cand]
                if pmax[cand] > bl:
                    bl = pmax[cand]
            else:
                cf += pmax[cand]
                if pmax[cand] > bf:
                    bf = pmax[cand]
        return res_on, cl, cf, bl, bf, bad

    # ---------------------------------------------------------------------
    for i in range(H):
        av = avail_l[i]
        flim = flim_nom if link_l[i] else flim_bkp
        s_l = share_l[i]
        nhat = nhat_l[i]
        if nhat < 0.0:
            nhat = 0.0
        sg = sag_l[i]
        base_on = on[:]

        # ---- 1. automatic recommendation (common to all policies) ----------
        auto_on, cl_a, cf_a, bl_a, bf_a, bad_a = commit(base_on, av, nhat, s_l, flim)

        # ---- 2. residual risk of the automatic recommendation --------------
        big_a = bl_a if bl_a >= bf_a else bf_a
        if sg > 1e-9:
            z = (cl_a + cf_a - big_a - nhat) / sg
            rho = 0.5 * math.erfc(z / SQRT2)
        else:
            rho = 0.0
        rho_tr[i] = rho

        # ---- 3. requirement selection per policy ---------------------------
        n_rob = nhat + beta_mode * sg
        if mode == "A":
            chosen, cl, cf, bl, bf, bad = auto_on, cl_a, cf_a, bl_a, bf_a, bad_a
            fired = False
            n_release = nhat
        elif mode == "B":
            trigger = ((gate.use_risk and rho > gate.rho_on)
                       or (gate.use_flag and flag_l[i]))
            if trigger:
                armed = True
                hold = gate.dwell_h
            elif armed and hold <= 0 and (not gate.use_risk or rho < gate.rho_off):
                armed = False
            if hold > 0:
                hold -= 1
            fired = armed
            if fired:
                chosen, cl, cf, bl, bf, bad = commit(base_on, av, n_rob, s_l, flim)
                if check_non_relaxation and any(auto_on[u] and not chosen[u]
                                                for u in range(nu)):
                    nr_viol += 1
            else:
                chosen, cl, cf, bl, bf, bad = auto_on, cl_a, cf_a, bl_a, bf_a, bad_a
            n_release = n_rob
        elif mode == "C":
            chosen, cl, cf, bl, bf, bad = commit(base_on, av, n_rob, s_l, flim)
            fired = False
            n_release = n_rob
        elif mode == "D":
            n_d = n_rob + gate.conservative_extra * hatD_l[i]
            chosen, cl, cf, bl, bf, bad = commit(base_on, av, n_d, s_l, flim)
            fired = False
            n_release = n_d
        elif mode == "E":
            n_e = (nhat + gate.static_load_frac * hatD_l[i]
                   + gate.static_wind_frac * hatR_l[i])
            chosen, cl, cf, bl, bf, bad = commit(base_on, av, n_e, s_l, flim)
            fired = False
            n_release = n_e
        else:  # F
            a, b, c = sD_l[i], sW_l[i], sV_l[i]
            srt = sorted((a, b, c), reverse=True)
            g_left, dev = gate.budget_gamma, 0.0
            for sk in srt:
                take = 1.0 if g_left >= 1.0 else max(g_left, 0.0)
                dev += take * sk
                g_left -= take
            n_f = nhat + dev
            chosen, cl, cf, bl, bf, bad = commit(base_on, av, n_f, s_l, flim)
            fired = False
            n_release = n_f

        infeas += bad
        if fired:
            gate_h += 1
            gate_tr[i] = True

        # ---- 4. full billing of start-ups ----------------------------------
        for u in range(nu):
            if chosen[u] and not base_on[u]:
                start_c += scost[u]
                starts += 1
                up_t[u] = 0
                dn_t[u] = 0
        on = chosen

        # ---- 5. release under the robust criterion (non-relaxation) --------
        nl_x = n_release * s_l
        nf_x = n_release - nl_x
        n_on = sum(1 for u in range(nu) if on[u] and av[u])
        for u in merit_rev:
            if n_on <= min_online:
                break
            if not (on[u] and av[u]):
                continue
            if up_t[u] < mnup[u]:
                continue
            if node[u] == 0:
                c_l, c_f = cl - pmax[u], cf
            else:
                c_l, c_f = cl, cf - pmax[u]
            b_l = b_f = 0.0
            for v in range(nu):
                if on[v] and av[v] and v != u:
                    if node[v] == 0:
                        if pmax[v] > b_l:
                            b_l = pmax[v]
                    elif pmax[v] > b_f:
                        b_f = pmax[v]
            if _security_ok(c_l, c_f, b_l, b_f, nl_x, nf_x, flim):
                on[u] = False
                up_t[u] = 0
                dn_t[u] = 0
                cl, cf, bl, bf = c_l, c_f, b_l, b_f
                n_on -= 1

        # ---- 6. real time ---------------------------------------------------
        dl, df = Dl_l[i], Df_l[i]
        res_l = Wl_l[i] + Vl_l[i]
        res_f = Wf_l[i] + Vf_l[i]

        pmin_l = pmin_f = pmax_l = pmax_f = 0.0
        for u in range(nu):
            if on[u] and av[u]:
                if node[u] == 0:
                    pmin_l += pmin[u]
                    pmax_l += pmax[u]
                else:
                    pmin_f += pmin[u]
                    pmax_f += pmax[u]

        # renewable uptake with local technical minima and link limit
        room_l = dl - pmin_l + flim
        room_f = df - pmin_f + flim
        use_l = res_l if res_l < room_l else max(room_l, 0.0)
        use_f = res_f if res_f < room_f else max(room_f, 0.0)
        if use_l < 0.0:
            use_l = 0.0
        if use_f < 0.0:
            use_f = 0.0
        room_sys = dl + df - pmin_l - pmin_f
        if room_sys < 0.0:
            room_sys = 0.0
        if use_l + use_f > room_sys:
            tot = use_l + use_f
            k = room_sys / tot if tot > 0 else 0.0
            use_l *= k
            use_f *= k
        curt += (res_l + res_f) - use_l - use_f
        if res_l > room_l or res_f > room_f:
            link_bind += 1

        net_l_r = dl - use_l
        net_f_r = df - use_f
        net_tot = net_l_r + net_f_r

        # rescue with fast-start units
        ens_i = 0.0
        short = net_tot - (pmax_l + pmax_f)
        if short > 1e-6:
            resc = short
            for u in merit:
                if resc <= 1e-6:
                    break
                if (not on[u]) and av[u] and fast[u] and dn_t[u] >= mndn[u]:
                    on[u] = True
                    up_t[u] = 0
                    dn_t[u] = 0
                    start_c += scost[u]
                    starts += 1
                    if node[u] == 0:
                        pmax_l += pmax[u]
                        pmin_l += pmin[u]
                        cl += pmax[u]
                        if pmax[u] > bl:
                            bl = pmax[u]
                    else:
                        pmax_f += pmax[u]
                        pmin_f += pmin[u]
                        cf += pmax[u]
                        if pmax[u] > bf:
                            bf = pmax[u]
                    resc = net_tot - (pmax_l + pmax_f)
                    if resc < 0.0:
                        resc = 0.0
            ens_i = (short - resc) * tau_rescue + resc
            if ens_i < 0.0:
                ens_i = 0.0
        served = net_tot - ens_i

        # economic dispatch with technical minima and ramps
        out = [0.0] * nu
        rem = served
        for u in range(nu):
            if on[u] and av[u]:
                lo = pmin[u]
                lo_r = p_prev[u] - ramp[u]
                if lo_r > lo:
                    lo = lo_r
                if lo > pmax[u]:
                    lo = pmax[u]
                out[u] = lo
                rem -= lo
        if rem > 0.0:
            for u in merit:
                if rem <= 1e-9:
                    break
                if not (on[u] and av[u]):
                    continue
                hi = pmax[u]
                hi_r = p_prev[u] + ramp[u]
                if hi_r < hi:
                    hi = hi_r
                if hi < out[u]:
                    hi = out[u]
                add = hi - out[u]
                if add > rem:
                    add = rem
                out[u] += add
                rem -= add
        elif rem < 0.0:
            for u in merit_rev:
                if rem >= -1e-9:
                    break
                if not (on[u] and av[u]):
                    continue
                red = out[u] - pmin[u]
                if red > -rem:
                    red = -rem
                if red > 0.0:
                    out[u] -= red
                    rem += red

        # transfer-limit correction
        gl = 0.0
        for u in range(nu):
            if node[u] == 0:
                gl += out[u]
        flow = gl - net_l_r
        if flow > flim + 1e-6 or flow < -flim - 1e-6:
            link_bind += 1
            excess = abs(flow) - flim
            src, dst = (0, 1) if flow > 0 else (1, 0)
            left = excess
            for u in merit_rev:
                if left <= 1e-9:
                    break
                if on[u] and av[u] and node[u] == src:
                    red = out[u] - pmin[u]
                    if red > left:
                        red = left
                    if red > 0.0:
                        out[u] -= red
                        left -= red
            inc_left = excess - left
            for u in merit:
                if inc_left <= 1e-9:
                    break
                if on[u] and av[u] and node[u] == dst:
                    hi = pmax[u]
                    hi_r = p_prev[u] + ramp[u]
                    if hi_r < hi:
                        hi = hi_r
                    inc = hi - out[u]
                    if inc > inc_left:
                        inc = inc_left
                    if inc > 0.0:
                        out[u] += inc
                        inc_left -= inc
            if inc_left > 1e-6:
                ens_i += inc_left
        ens_tot += ens_i
        if ens_i > 1e-9:
            ens_h += 1

        for u in range(nu):
            q = out[u]
            if q > 0.0:
                fuel_c += cvar[u] * q
                co2 += ef[u] * q
        ens_c += voll * ens_i

        # ---- 7. violation indicator on the realized state -------------------
        rl = rf = b1 = b2 = 0.0
        for u in range(nu):
            if on[u] and av[u]:
                if node[u] == 0:
                    rl += pmax[u]
                    if pmax[u] > b1:
                        b1 = pmax[u]
                else:
                    rf += pmax[u]
                    if pmax[u] > b2:
                        b2 = pmax[u]
        secure = _security_ok(rl, rf, b1, b2, net_l_r, net_f_r, flim)
        if not secure:
            bg = b1 if b1 >= b2 else b2
            if (rl + flim < net_l_r) or (rf + flim < net_f_r):
                fail_kind[0] += 1
            elif rl + rf - bg < net_l_r + net_f_r:
                fail_kind[1] += 1
            else:
                fail_kind[2] += 1
        if (not secure) or ens_i > 1e-9:
            viol += 1
            adverse_tr[i] = True

        # ---- 8. state update -------------------------------------------------
        for u in range(nu):
            if on[u] and av[u]:
                up_t[u] += 1
                dn_t[u] = 0
                p_prev[u] = out[u]
            else:
                dn_t[u] += 1
                up_t[u] = 0
                p_prev[u] = 0.0

    res_int = float(Wr.sum() + Vr.sum()) - curt
    r = RunResult(
        mode=mode,
        cost_MEUR=(fuel_c + start_c + ens_c) / 1e6,
        fuel_cost_MEUR=fuel_c / 1e6,
        start_cost_MEUR=start_c / 1e6,
        ens_cost_MEUR=ens_c / 1e6,
        ens_MWh=ens_tot,
        curtailment_GWh=curt / 1e3,
        co2_kt=co2 / 1e3,
        violation_h=viol,
        ens_h=ens_h,
        gate_h=gate_h,
        starts=starts,
        res_share_pct=100.0 * res_int / float(D.sum()),
        infeasible_h=infeas,
        link_binding_h=link_bind,
        non_relaxation_violations=nr_viol,
    )
    if reference_adverse is not None and mode == "B":
        g = gate_tr
        r.avoided_adverse_h = int((g & reference_adverse & ~adverse_tr).sum())
        r.false_alarm_h = int((g & ~reference_adverse).sum())
        n = int(g.sum())
        r.gate_precision_pct = 100.0 * int((g & reference_adverse).sum()) / n if n else float("nan")
    r.traces = dict(fail_kind=tuple(fail_kind))
    if keep_traces:
        r.traces.update(gate=gate_tr, adverse=adverse_tr, rho=rho_tr)
    return r
