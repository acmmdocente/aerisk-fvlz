"""
system.py — Physical definition of the Fuerteventura-Lanzarote (FV-LZ) power system.

All structural parameters come from official primary sources and are declared
with their traceability in `PARAMETER_PROVENANCE`.  Economic and emission
parameters are NOT postulated: they are derived from fuel price, lower heating
value (LHV), net efficiency, and IPCC default emission factors (2006, vol. 2,
table 1.4), so that any reviewer can reconstruct them and subject them to
sensitivity analysis.

Sources:
  [REE-CTSOC-2026] Red Electrica. Comite Tecnico de Seguimiento de la Operacion
      del Sistema Electrico de Canarias. January-December 2025. March 2026.
  [BOC-190-2024]   Official Gazette of the Canary Islands no. 190/2024,
      announcement 3047 (system operator's report on the closure proceedings of
      the Punta Grande and Las Salinas units).
  [REE-2022]       REE press release, 17/10/2022: 132 kV submarine link
      Playa Blanca-La Oliva, single circuit, 120 MVA, 14.5 km submarine route.
  [REE-2024]       REE press release, 13/12/2024: 132 kV double-circuit backbone
      La Oliva-Matas Blancas (140 M EUR, five GIS substations).
  [IPCC-2006]      IPCC 2006 Guidelines, vol. 2, table 1.4 (default factors).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. Official reference magnitudes (year 2025) — calibration targets
# ---------------------------------------------------------------------------

OFFICIAL_2025 = {
    # [REE-CTSOC-2026], slide 4
    "demand_GWh": 1592.0,
    # [REE-CTSOC-2026], slide 8 (maximum instantaneous power 11/08/2025 19:54)
    "peak_MW": 272.3,
    # [REE-CTSOC-2026], slide 8 (maximum hourly energy)
    "peak_hourly_MWh": 266.1,
    # [REE-CTSOC-2026], slide 13 (origin of cumulative 2025 production)
    "renewable_share_pct": 17.1,
    # [REE-CTSOC-2026], slide 16 (total FV-LZ renewable curtailment, cumulative 2025)
    "curtailment_pct": 15.67,
    "curtailment_wind_pct": 17.33,
    "curtailment_pv_pct": 12.13,
    # [REE-CTSOC-2026], slide 15 (% of time with renewable limitation)
    "time_limited_pct": 42.9,
    # [REE-CTSOC-2026], slide 14 (instantaneous maximum, 23/03/2025 13:55)
    "max_instant_renewable_pct": 45.97,
    # [REE-CTSOC-2026], slide 11: installed capacity, category A
    "conventional_installed_MW": 205.0 + 159.0,
    # [REE-CTSOC-2026], slide 30: 2026 coverage report
    "available_forecast_2026_MW": 335.0,
    "peak_forecast_2026_MW": 259.0,
    "coverage_index_2026": 1.29,
    # [REE-CTSOC-2026], slide 12: installed renewable capacity per island
    "wind_installed_MW": 40.7 + 64.9,
    "pv_installed_MW": 11.1 + 50.1,
    "other_renewable_MW": 2.1,
    # [REE-CTSOC-2026], slide 27 (frequency outside +-150 mHz for over 5 min)
    "hours_freq_out_150mHz": 20.5,
    # [REE-CTSOC-2026], slide 22
    "supply_loss_incidents_2025": 2,
}

# Regional planning target (PTECan-2030): 60% of electricity demand
# from renewable sources in 2030.
PTECAN_2030_TARGET = 0.60

# ---------------------------------------------------------------------------
# Capacity factors: derivation from official data
# ---------------------------------------------------------------------------
# The joint FV-LZ available renewable energy (322.8 GWh) does not by itself
# identify the split between wind and solar capacity factors: infinitely many
# pairs reproduce it.  The split is pinned down by a second official datum:
# generation and installed capacity by technology for the Canary Islands as a
# whole, which are disaggregated [REE-CTSOC-2026, slides 10 and 12].
#
#   wind:      1,373.41 GWh / (653.2 MW x 8,760 h) = 0.2400  (integrated)
#   solar PV:    460.00 GWh / (325.0 MW x 8,760 h) = 0.1616  (integrated)
#
# Correcting for the regional curtailment of each technology (16.25% and
# 11.35%, slide 16) yields the capacity factor of the AVAILABLE resource:
#
#   wind       0.2400 / (1 - 0.1625) = 0.2866
#   solar PV   0.1616 / (1 - 0.1135) = 0.1823
#
# Applied to the FV-LZ capacity, those factors would yield 362.8 GWh against
# the official 322.8 GWh.  The discrepancy is attributed to 18.4 MW — 10.9%
# of the renewable capacity — entering service over the course of 2025
# [REE-CTSOC-2026, slide 12], so that the mean available capacity was below
# the year-end one.  A reconciliation factor k = 322.8/362.8 = 0.890 is
# accordingly applied, common to both technologies so as not to introduce
# an arbitrary split.
CANARIAS_2025 = {
    "wind_energy_GWh": 1373.41, "wind_capacity_MW": 653.2, "wind_curtailment": 0.1625,
    "pv_energy_GWh": 460.00, "pv_capacity_MW": 325.0, "pv_curtailment": 0.1135,
}


def derive_capacity_factors(reconcile: bool = True) -> tuple[float, float]:
    """Capacity factors of the available resource derived from the official data."""
    c = CANARIAS_2025
    cf_w = c["wind_energy_GWh"] * 1e3 / (c["wind_capacity_MW"] * 8760.0)
    cf_v = c["pv_energy_GWh"] * 1e3 / (c["pv_capacity_MW"] * 8760.0)
    cf_w /= (1.0 - c["wind_curtailment"])
    cf_v /= (1.0 - c["pv_curtailment"])
    if not reconcile:
        return round(cf_w, 4), round(cf_v, 4)
    target = (OFFICIAL_2025["renewable_share_pct"] / 100.0
              * OFFICIAL_2025["demand_GWh"]
              / (1.0 - OFFICIAL_2025["curtailment_pct"] / 100.0))
    raw = (S0_WIND_MW * 8760.0 * cf_w + S0_PV_MW * 8760.0 * cf_v) / 1e3
    k = target / raw
    return round(cf_w * k, 4), round(cf_v * k, 4)


# Alternative sets used in the sensitivity analysis of the 2030 scenario
# (objection M2 of the editorial assessment).
CF_VARIANTS = {
    "reconciliado": None,          # solved by derive_capacity_factors()
    "recurso pleno": (0.2866, 0.1823),
    "conservador": (0.2300, 0.1450),
}

# ---------------------------------------------------------------------------
# 2. Transparent derivation of variable costs and emission factors
# ---------------------------------------------------------------------------
# Variable cost [EUR/MWh_e] = fuel_price/LHV/efficiency + variable O&M
# Emission factor [tCO2/MWh_e] = EF_IPCC / efficiency
# Net efficiency degrades with unit age, which reproduces the real merit
# order within each technology.

FUELS = {
    # LHV in MWh_th/t; EF in tCO2/MWh_th (IPCC 2006, vol. 2, table 1.4)
    #   residual fuel oil: 77.4 tCO2/TJ -> 0.27864 tCO2/MWh_th
    #   gas/diesel oil: 74.1 tCO2/TJ -> 0.26676 tCO2/MWh_th
    "fuel_oil": {"price_EUR_t": 480.0, "LHV_MWh_t": 11.25, "EF_tCO2_MWh_th": 0.27864},
    "gas_oil": {"price_EUR_t": 750.0, "LHV_MWh_t": 11.86, "EF_tCO2_MWh_th": 0.26676},
}

VARIABLE_OM = {"diesel": 14.0, "gt": 11.0}  # EUR/MWh_e


def derive_cost_and_emissions(fuel: str, efficiency: float, tech: str,
                              fuel_price_scale: float = 1.0) -> tuple[float, float]:
    """Return (variable cost EUR/MWh_e, emission factor tCO2/MWh_e)."""
    f = FUELS[fuel]
    fuel_cost_th = f["price_EUR_t"] * fuel_price_scale / f["LHV_MWh_t"]  # EUR/MWh_th
    cvar = fuel_cost_th / efficiency + VARIABLE_OM[tech]
    ef = f["EF_tCO2_MWh_th"] / efficiency
    return cvar, ef


# ---------------------------------------------------------------------------
# 3. Aggregated thermal fleet at two nodes
# ---------------------------------------------------------------------------
# Split of the available capacity foreseen for 2026 (335 MW, [REE-CTSOC-2026]
# slide 30) between the islands in proportion to the published per-island
# available capacities of the same slide (Lanzarote 205 MW, Fuerteventura
# 148 MW):  335 x 205/353 = 194.6 MW (LZ) and 335 x 148/353 = 140.4 MW (FV).
# The Punta Grande gas turbines (37.5 and 23.45 MW) are the only unit
# capacities documented nominally and are preserved as such.

NODES = ("LZ", "FV")


@dataclass(frozen=True)
class Unit:
    name: str
    node: str
    pmax: float          # MW
    pmin_frac: float     # fraction of pmax
    efficiency: float    # net efficiency at full load
    fuel: str
    tech: str            # 'diesel' | 'gt'
    ramp_frac: float     # MW/h as a fraction of pmax
    min_up: int          # h
    min_down: int        # h
    start_cost: float    # EUR
    fast_start: bool     # capable of starting in < 15 min

    @property
    def pmin(self) -> float:
        return self.pmax * self.pmin_frac

    @property
    def ramp(self) -> float:
        return self.pmax * self.ramp_frac


def _diesel(name: str, node: str, pmax: float, eff: float) -> Unit:
    return Unit(name=name, node=node, pmax=pmax, pmin_frac=0.35, efficiency=eff,
                fuel="fuel_oil", tech="diesel", ramp_frac=0.50,
                min_up=4, min_down=3, start_cost=2800.0, fast_start=False)


def _gt(name: str, node: str, pmax: float, eff: float) -> Unit:
    return Unit(name=name, node=node, pmax=pmax, pmin_frac=0.30, efficiency=eff,
                fuel="gas_oil", tech="gt", ramp_frac=1.00,
                min_up=1, min_down=1, start_cost=1500.0, fast_start=True)


def build_fleet(extra_flexible_MW: float = 0.0) -> list[Unit]:
    """Aggregated thermal fleet of 16 units, 334.85 MW.

    `extra_flexible_MW` allows the capacity awarded in the competitive tender
    resolved on 03/02/2026 ([REE-CTSOC-2026], slide 36: 215 MW for FV-LZ) to be
    instantiated as a structural sensitivity analysis.
    """
    fleet: list[Unit] = []
    # --- Punta Grande (Lanzarote): 6 diesel engines + 2 gas turbines ---
    for i, eff in enumerate((0.440, 0.435, 0.430, 0.425, 0.418, 0.410), start=1):
        fleet.append(_diesel(f"LZ-D{i}", "LZ", 22.30, eff))
    fleet.append(_gt("LZ-GT1", "LZ", 37.50, 0.330))
    fleet.append(_gt("LZ-GT2", "LZ", 23.45, 0.315))
    # --- Las Salinas (Fuerteventura): 6 diesel engines + 2 gas turbines ---
    for i, eff in enumerate((0.438, 0.432, 0.427, 0.421, 0.414, 0.406), start=1):
        fleet.append(_diesel(f"FV-D{i}", "FV", 17.40, eff))
    fleet.append(_gt("FV-GT1", "FV", 23.45, 0.322))
    fleet.append(_gt("FV-GT2", "FV", 12.00, 0.300))

    if extra_flexible_MW > 0:
        n = max(1, int(round(extra_flexible_MW / 20.0)))
        size = extra_flexible_MW / n
        for i in range(1, n + 1):
            node = "LZ" if i % 2 else "FV"
            fleet.append(
                Unit(name=f"NEW-{i}", node=node, pmax=size, pmin_frac=0.25,
                     efficiency=0.455, fuel="fuel_oil", tech="diesel",
                     ramp_frac=0.80, min_up=2, min_down=2,
                     start_cost=2200.0, fast_start=True))
    return fleet


# ---------------------------------------------------------------------------
# 4. Two-node topology
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Topology:
    """FV-LZ link.

    The 132 kV Playa Blanca-La Oliva link is SINGLE circuit, 120 MVA
    [REE-2022]; it coexists with the historical 66 kV link in service since
    2005.  The 2026 coverage report [REE-CTSOC-2026, slide 30] expressly warns
    that the single-circuit contingency separates the system into two
    electrical islands, with the coverage index of Fuerteventura falling from
    1.19 to 0.96.
    """
    link_capacity_MW: float = 120.0   # 120 MVA, unity power factor
    # The unavailability of the single 132 kV circuit is modeled as COMPLETE
    # SEPARATION (zero transfer capacity), which is the assumption the operator
    # identifies in its coverage report and the only one that produces two
    # electrical islands.  No residual capacity is postulated: the historical
    # 66 kV link supplies the Playa Blanca substation from Las Salinas, which
    # shifts load between islands but does not restore exchange capacity
    # between the two thermal fleets.
    link_capacity_outage_MW: float = 0.0


# Per-island demand split.  The aggregate report publishes the joint demand
# (1,592 GWh); the split between islands is anchored in the 2024 island
# balances (Lanzarote 817 GWh, Fuerteventura 766 GWh) and is consistent with
# the 2025 island peaks (152.8 MW in Lanzarote, 123.3 MW in Fuerteventura).
# It is expressly declared as a split hypothesis.
DEMAND_SHARE = {"LZ": 0.516, "FV": 0.484}

# Per-island split of the renewable resource, with the official December
# 2025 capacities [REE-CTSOC-2026, slide 12].
RENEWABLE_SPLIT_2025 = {
    "wind": {"LZ": 40.7, "FV": 64.9},
    "pv": {"LZ": 11.1, "FV": 50.1},
}


@dataclass
class SystemConfig:
    """Complete configuration of one structural scenario."""
    name: str
    wind_MW: float
    pv_MW: float
    fleet: list[Unit] = field(default_factory=build_fleet)
    topology: Topology = field(default_factory=Topology)
    fuel_price_scale: float = 1.0
    voll_EUR_MWh: float = 10000.0   # value of lost load

    def unit_params(self):
        """Flat vectors (Python lists) for the simulation loop."""
        pmax, pmin, cvar, ef, ramp, mu, md, sc, fast, node = ([] for _ in range(10))
        for u in self.fleet:
            c, e = derive_cost_and_emissions(u.fuel, u.efficiency, u.tech,
                                             self.fuel_price_scale)
            pmax.append(u.pmax)
            pmin.append(u.pmin)
            cvar.append(c)
            ef.append(e)
            ramp.append(u.ramp)
            mu.append(u.min_up)
            md.append(u.min_down)
            sc.append(u.start_cost)
            fast.append(u.fast_start)
            node.append(0 if u.node == "LZ" else 1)
        return dict(pmax=pmax, pmin=pmin, cvar=cvar, ef=ef, ramp=ramp,
                    min_up=mu, min_down=md, start_cost=sc, fast=fast, node=node)


# ---------------------------------------------------------------------------
# 5. Structural scenarios
# ---------------------------------------------------------------------------
# S0: official renewable portfolio as of 31/12/2025 [REE-CTSOC-2026, slide 12].
# S1: renewable capacity DERIVED from the 60% PTECan-2030 target with the
#     technology mix and capacity factors calibrated on 2025 (see
#     `scenarios.derive_s1_capacity`).  It is not a postulated figure.

S0_WIND_MW = RENEWABLE_SPLIT_2025["wind"]["LZ"] + RENEWABLE_SPLIT_2025["wind"]["FV"]
S0_PV_MW = RENEWABLE_SPLIT_2025["pv"]["LZ"] + RENEWABLE_SPLIT_2025["pv"]["FV"]


def derive_s1_capacity(target_share: float = PTECAN_2030_TARGET,
                       demand_GWh: float = OFFICIAL_2025["demand_GWh"],
                       cf_wind: float | None = None, cf_pv: float | None = None,
                       expected_curtailment: float = 0.15) -> tuple[float, float]:
    """Derive the renewable capacity of scenario S1 from the official
    penetration target, preserving the wind/solar proportion of the 2025
    portfolio.

    Required integrated renewable energy = target_share x demand.
    Required available energy = integrated / (1 - expected curtailment).
    """
    if cf_wind is None or cf_pv is None:
        cf_wind, cf_pv = derive_capacity_factors()
    required_integrated = target_share * demand_GWh
    required_producible = required_integrated / (1.0 - expected_curtailment)
    share_wind = S0_WIND_MW / (S0_WIND_MW + S0_PV_MW)
    share_pv = 1.0 - share_wind
    # energy per total installed MW, GWh/MW-yr
    energy_per_MW = (share_wind * cf_wind + share_pv * cf_pv) * 8760.0 / 1000.0
    total_MW = required_producible / energy_per_MW
    return round(total_MW * share_wind, 1), round(total_MW * share_pv, 1)


PARAMETER_PROVENANCE = {
    "demanda anual, punta, cuota renovable, vertido": "[REE-CTSOC-2026]",
    "potencia instalada convencional y renovable por isla": "[REE-CTSOC-2026, laminas 11-12]",
    "potencia disponible prevista 2026 e indice de cobertura": "[REE-CTSOC-2026, lamina 30]",
    "separacion del sistema por contingencia del circuito simple": "[REE-CTSOC-2026, lamina 30, nota **]",
    "capacidad y tipologia del enlace de 132 kV": "[REE-2022]",
    "eje de doble circuito de 132 kV en Fuerteventura": "[REE-2024]",
    "necesidad y potencia operativa de los grupos termicos": "[BOC-190-2024]",
    "factores de emision por combustible": "[IPCC-2006, vol.2 tabla 1.4]",
    "objetivo de penetracion renovable 2030": "PTECan-2030",
    "coste variable y factor de emision por grupo": "derivados (derive_cost_and_emissions)",
}
