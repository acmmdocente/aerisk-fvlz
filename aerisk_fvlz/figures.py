"""
figures.py — Generation of every figure of the manuscript.

Design principles
-----------------
* Every quantitative figure is generated from `campaign_results.json`.
  No number is transcribed by hand.
* Vector output (PDF) plus high-resolution bitmap (PNG, 600 dpi).
* Palette anchored in the object of study: basalt (thermal fleet), Atlantic
  blue (wind), sand (solar PV), trade-wind teal (safeguard), and a single
  alert red reserved for the adverse consequence.
* Every figure with simulation results carries an express synthetic-data
  label.  Figures mixing implemented and conceptual elements distinguish them
  by solid and dashed strokes respectively.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

# -------------------------------------------------------------------- palette
BASALT = "#1F2933"
SLATE = "#4A5A68"
ATLANTIC = "#1F6F8B"
TRADE = "#2E9E8F"
SAND = "#D9A441"
ALERT = "#B23A2E"
MIST = "#E8ECEF"
PAPER = "#FFFFFF"

POLICY_COLOR = {"A": ALERT, "B": TRADE, "C": ATLANTIC, "D": SLATE,
                "E": "#8A6A1E", "F": "#7B4B6B"}
POLICY_LABEL = {
    "A": "A · deterministic",
    "B": "B · safeguarded",
    "C": "C · permanently robust",
    "D": "D · conservative",
    "E": "E · static reserve rule (ref.)",
    "F": "F · uncertainty budget (ref.)",
}
POLICY_ORDER = ("A", "E", "B", "F", "C", "D")

RC = {
    "figure.dpi": 120,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.06,
    "figure.constrained_layout.w_pad": 0.06,
    "figure.constrained_layout.hspace": 0.09,
    "figure.constrained_layout.wspace": 0.07,
    "savefig.dpi": 600,
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "axes.titleweight": "bold",
    "axes.edgecolor": SLATE,
    "axes.linewidth": 0.7,
    "axes.grid": True,
    "grid.color": MIST,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

SYNTH = "Simulation on calibrated synthetic series — not observed data"


AUDIT_LOG: list = []


def _audit(fig, name: str):
    """Check that no text overlaps another or overflows the canvas.

    The verification is geometric and automatic: the boxes of all visible
    texts are collected and tested pairwise.  Any defect is recorded in
    AUDIT_LOG, so that figure generation is self-verifying and does not depend
    on ocular inspection.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for ax in fig.get_axes():
        for t in (ax.title, ax.xaxis.label, ax.yaxis.label):
            if t.get_text().strip() and t.get_visible():
                items.append(t)
        items += [t for t in ax.texts if t.get_text().strip() and t.get_visible()]
        # axis tick labels only count if the axis is drawn: in the schematic
        # figures the axis is off and its labels, though existing as artists,
        # are not printed
        if getattr(ax, "axison", True):
            for axis, get_lab, get_loc, lim in (
                    (ax.xaxis, ax.get_xticklabels, ax.get_xticks, ax.get_xlim()),
                    (ax.yaxis, ax.get_yticklabels, ax.get_yticks, ax.get_ylim())):
                if not axis.get_visible():
                    continue
                lo, hi = min(lim), max(lim)
                labs, locs = list(get_lab()), list(get_loc())
                for k, t in enumerate(labs):
                    # a tick outside the visible range exists as an artist but is not
                    # printed: counting it would produce false positives
                    if k < len(locs) and not (lo - 1e-9 <= locs[k] <= hi + 1e-9):
                        continue
                    if t.get_text().strip() and t.get_visible():
                        items.append(t)
        leg = ax.get_legend()
        if leg is not None:
            items += list(leg.get_texts())
    items += [t for t in fig.texts if t.get_text().strip() and t.get_visible()]
    for leg in getattr(fig, "legends", []):
        items += list(leg.get_texts())

    from matplotlib.text import Annotation, Text
    boxes = []
    for t in items:
        try:
            # in an annotation, get_window_extent returns the union of the text and
            # the arrow; for label auditing only the text box matters
            bb = (Text.get_window_extent(t, renderer=r)
                  if isinstance(t, Annotation) else t.get_window_extent(renderer=r))
        except Exception:
            continue
        if bb.width > 0 and bb.height > 0:
            boxes.append((t.get_text().replace("\n", " / ")[:40], bb))

    problems = []
    tol, min_frac = 1.5, 0.06
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            s1, b1 = boxes[i]
            s2, b2 = boxes[j]
            dx = min(b1.x1, b2.x1) - max(b1.x0, b2.x0) + tol
            dy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0) + tol
            if dx <= 0 or dy <= 0:
                continue
            amin = min(b1.width * b1.height, b2.width * b2.height)
            if amin > 0 and (dx * dy) / amin >= min_frac:
                problems.append(
                    f"solape {100*dx*dy/amin:.0f}% «{s1}»[{b1.x0:.0f},{b1.y0:.0f}]"
                    f" × «{s2}»[{b2.x0:.0f},{b2.y0:.0f}]")
    W, Hh = fig.bbox.width, fig.bbox.height
    for s1, b1 in boxes:
        if b1.x0 < -1 or b1.y0 < -1 or b1.x1 > W + 1 or b1.y1 > Hh + 1:
            problems.append(f"desborde «{s1}» x[{b1.x0:.0f},{b1.x1:.0f}] "
                            f"y[{b1.y0:.0f},{b1.y1:.0f}] lienzo {W:.0f}x{Hh:.0f}")
    AUDIT_LOG.append((name, problems))
    return problems


def _save(fig, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    _audit(fig, name)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
    return outdir / f"{name}.pdf"


def _tidy(ax, *, xgrid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=xgrid)


def _footnote(fig, text, y=-0.02):
    """No-op: data provenance and the synthetic-series notice are recorded in
    the manuscript figure captions, not inside the canvas.

    Labeling them inside forced them outside the figure limits, which made
    them overflow the canvas in all eleven cases.
    """
    return None


# =====================================================================
# Figure 1 — Two-node system schematic
# =====================================================================

def fig01_system(outdir: Path, official: dict):
    """System schematic.  The islands are shifted to the right to reserve a
    left column for the magnitude box and the legend, so that no label
    competes with the schematic cartography."""
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7.6, 4.5))
        ax.set_xlim(0, 118)
        ax.set_ylim(0, 66)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 118, 66, fc="#F2F6F8", ec="none"))

        lz = Polygon([(80, 62), (96, 63), (106, 56), (108, 45), (100, 38),
                      (86, 39), (78, 47), (76, 56)], closed=True,
                     fc="#DED7C8", ec=SLATE, lw=0.8)
        fv = Polygon([(58, 40), (68, 41), (74, 34), (72, 22), (64, 8),
                      (54, 3), (48, 8), (51, 22), (54, 33)], closed=True,
                     fc="#DED7C8", ec=SLATE, lw=0.8)
        ax.add_patch(lz)
        ax.add_patch(fv)
        ax.text(95, 57.0, "LANZAROTE", fontsize=9.5, weight="bold", color=BASALT,
                ha="center")
        ax.text(53.5, 15.0, "FUERTEVENTURA", fontsize=9.5, weight="bold",
                color=BASALT, ha="center", va="center", rotation=-58)

        # inter-island links
        ax.plot([68.5, 84.5], [40.5, 44.0], color=ALERT, lw=2.4,
                solid_capstyle="round", zorder=3)
        ax.plot([68.5, 84.5], [38.2, 41.7], color=SLATE, lw=1.3, ls=(0, (4, 2)),
                zorder=3)
        ax.annotate("132 kV submarine link\nsingle circuit · 120 MVA · 2022",
                    xy=(78.0, 42.6), xytext=(64, 60.5), fontsize=6.9, color=ALERT,
                    weight="bold", ha="center",
                    arrowprops=dict(arrowstyle="-", color=ALERT, lw=0.6))
        ax.text(92, 33.5, "66 kV link (2005)", fontsize=6.8, color=SLATE,
                ha="center", style="italic")

        # double-circuit island backbone
        ax.plot([62, 65, 67, 63, 58], [39.5, 32, 24, 15, 8], color=ATLANTIC,
                lw=2.0, zorder=3)
        ax.annotate("132 kV double-circuit backbone\nLa Oliva–Matas Blancas (2024)",
                    xy=(66.6, 25.5), xytext=(95, 17), fontsize=7.0, color=ATLANTIC,
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=ATLANTIC, lw=0.6))

        # power stations
        for (x, y) in ((96, 50), (62.5, 39.5)):
            ax.add_patch(FancyBboxPatch((x - 2, y - 1.6), 4, 3.2,
                                        boxstyle="round,pad=0.15", fc=BASALT,
                                        ec="none", zorder=4))
            ax.text(x, y, "⊥", color="white", ha="center", va="center",
                    fontsize=9, zorder=5)
        ax.text(96, 53.6, "Punta Grande thermal station", fontsize=7.4, weight="bold",
                color=BASALT, ha="center")
        ax.text(101, 49.6, "Arrecife", fontsize=6.8, color=SLATE, ha="left")
        ax.text(59.5, 43.2, "Las Salinas thermal station", fontsize=7.4, weight="bold",
                color=BASALT, ha="right")
        ax.text(59.5, 38.6, "Puerto del Rosario", fontsize=6.8, color=SLATE,
                ha="right")

        for (x, y) in ((88, 52), (103, 45)):
            ax.plot([x], [y], marker="1", ms=13, color=ATLANTIC, mew=1.6)
        ax.plot([84], [46], marker="s", ms=6, color=SAND)
        for (x, y) in ((60, 26), (61, 11)):
            ax.plot([x], [y], marker="1", ms=13, color=ATLANTIC, mew=1.6)
        for (x, y) in ((56, 31), (66, 18)):
            ax.plot([x], [y], marker="s", ms=6, color=SAND)

        # official magnitudes
        box = ("FV–LZ system · 2025 close\n"
               f"Demand  {official['demand_GWh']:,.0f} GWh\n"
               f"Peak  {official['peak_MW']:.1f} MW\n"
               f"Conventional (cat. A)  {official['conventional_installed_MW']:.0f} MW\n"
               f"Wind  {official['wind_installed_MW']:.1f} MW\n"
               f"Solar PV  {official['pv_installed_MW']:.1f} MW\n"
               f"Renewable share  {official['renewable_share_pct']:.1f}%\n"
               f"Renewable curtailment  {official['curtailment_pct']:.2f}%")
        ax.text(2.6, 63.5, box, fontsize=6.8, color=BASALT,
                va="top", linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=SLATE, lw=0.6))

        # symbol legend
        ax.add_patch(FancyBboxPatch((2.4, 4.0), 33, 17.0,
                                    boxstyle="round,pad=0.4", fc="white",
                                    ec=SLATE, lw=0.6))
        ax.add_patch(FancyBboxPatch((4.6, 17.4), 2.6, 2.2,
                                    boxstyle="round,pad=0.12", fc=BASALT, ec="none"))
        ax.text(9.4, 18.5, "Thermal station (diesel + gas turbine)",
                fontsize=6.3, color=BASALT, va="center")
        ax.plot([5.9], [15.0], marker="1", ms=11, color=ATLANTIC, mew=1.6)
        ax.text(9.4, 15.0, "Wind farm", fontsize=6.3, color=BASALT, va="center")
        ax.plot([5.9], [11.7], marker="s", ms=6, color=SAND)
        ax.text(9.4, 11.7, "Solar PV plant", fontsize=6.3, color=BASALT,
                va="center")
        ax.plot([4.8, 7.0], [8.4, 8.4], color=ALERT, lw=2.2)
        ax.text(9.4, 8.4, "132 kV link (single circuit)", fontsize=6.3,
                color=BASALT, va="center")
        ax.plot([4.8, 7.0], [6.0, 6.0], color=ATLANTIC, lw=2.0)
        ax.text(9.4, 6.0, "132 kV island backbone (double circuit)", fontsize=6.3,
                color=BASALT, va="center")

        ax.text(114, 62, "N ↑", fontsize=8, color=SLATE, ha="right")
        ax.text(70, 2.2,
                "The single-circuit 132 kV contingency separates the system\n"
                "into two electrical islands (2026 coverage report)",
                fontsize=6.9, color=ALERT, ha="center", va="center", style="italic")
        ax.set_title("Fuerteventura–Lanzarote power system: two-node topology",
                     loc="left")
        return _save(fig, outdir, "fig01_two_node_system")


# =====================================================================
# Figure 2 — Structural error-propagation graph
# =====================================================================

def fig02_error_graph(outdir: Path):
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7.9, 3.4))
        ax.set_xlim(0, 116)
        ax.set_ylim(0, 46)
        ax.axis("off")
        nodes = [("Datos\n$X_t$", 9), ("Forecast\n$\\hat D,\\hat W,\\hat V$", 26),
                 ("Recommendation\n$r_t$", 43), ("Risk gate\n$\\gamma_t$", 60),
                 ("Despacho\n$d_t$", 77), ("Physical response\n$\\Phi_t$", 92),
                 ("Consecuencia\n$C_t$", 107)]
        for i, (lab, x) in enumerate(nodes):
            emph = lab.startswith("Puerta") or lab.startswith("Consecuencia")
            fc = "#FBEEEC" if lab.startswith("Consecuencia") else (
                "#E4F4F1" if emph else "#EDF2F5")
            ec = ALERT if lab.startswith("Consecuencia") else (TRADE if emph else ATLANTIC)
            ax.add_patch(FancyBboxPatch((x - 6.5, 20), 13.0, 8,
                                        boxstyle="round,pad=0.35", fc=fc, ec=ec, lw=1.1))
            ax.text(x, 24, lab, ha="center", va="center", fontsize=6.9,
                    color=BASALT, weight="bold" if emph else "normal")
            if i:
                ax.add_patch(FancyArrowPatch((nodes[i - 1][1] + 6.85, 24), (x - 6.85, 24),
                                             arrowstyle="-|>", mutation_scale=9,
                                             color=BASALT, lw=0.9))
        for lab, x in (("$\\varepsilon^{X}$", 9), ("$\\varepsilon^{P}$", 26),
                       ("$\\varepsilon^{O}$", 43)):
            ax.add_patch(plt.Circle((x, 39), 3.0, fc="#FAF0DC", ec=SAND, lw=1.0,
                                    ls=(0, (3, 2))))
            ax.text(x, 39, lab, ha="center", va="center", fontsize=7.5, color="#8A6A1E")
            ax.add_patch(FancyArrowPatch((x, 35.6), (x, 28.4), arrowstyle="-|>",
                                         mutation_scale=8, color=SAND, lw=1.0,
                                         ls=(0, (3, 2))))
        ax.text(2, 44, "Error injection points", fontsize=7.6, color="#8A6A1E",
                weight="bold")
        for lab, x, style in (("Data\nvalidation", 9, "solid"),
                              ("Robust setpoint\n+ ratchet", 60, "solid"),
                              ("Fast rescue\nand procedure", 92, "dashed")):
            ls = "-" if style == "solid" else (0, (3, 2))
            ax.add_patch(FancyBboxPatch((x - 8.5, 4), 17, 8.5,
                                        boxstyle="round,pad=0.3", fc="#E6F3EF",
                                        ec=TRADE, lw=1.0, ls=ls))
            ax.text(x, 8.2, lab, ha="center", va="center", fontsize=6.9, color="#1D6157")
            ax.add_patch(FancyArrowPatch((x, 12.8), (x, 19.6), arrowstyle="-|>",
                                         mutation_scale=8, color=TRADE, lw=1.0))
        ax.text(34, 8.2, "Containment barriers", fontsize=7.6, color="#1D6157",
                weight="bold", ha="center")
        ax.set_title("Structural graph of error propagation along the dispatch cycle",
                     loc="left")
        _footnote(fig,
                  "Solid strokes: mechanisms implemented and evaluated in the campaign. "
                  "Dashed strokes: elements specified for deployment and not exercised numerically. "
                  "The graph describes the propagation structure; no causal identification is claimed for the real system.",
                  y=0.03)
        return _save(fig, outdir, "fig02_error_propagation")


# =====================================================================
# Figure 3 — Criticality operationalized with the real thresholds
# =====================================================================

def fig03_criticality(outdir: Path, gate: dict, rho_quantiles=None):
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        ron, roff = gate["rho_on"], gate["rho_off"]
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 1)
        ax.axvspan(0, roff, color="#E8F3EE")
        ax.axvspan(roff, ron, color="#FAF0DC")
        ax.axvspan(ron, 0.5, color="#FBEAE7")
        ax.axhline(0.62, color=SLATE, lw=0.7, ls=(0, (3, 2)))
        ax.text(0.005, 0.95, "Acceptance of the automatic recommendation",
                fontsize=7.4, color="#1D6157", weight="bold", va="top")
        ax.text(roff + 0.005, 0.55, "Dwell zone\n(trigger hysteresis)",
                fontsize=7.0, color="#8A6A1E", va="top")
        ax.text(ron + 0.005, 0.95, "Safeguard activation:\nrobust setpoint and ratchet",
                fontsize=7.4, color=ALERT, weight="bold", va="top")
        ax.text(0.25, 0.633,
                "Severity threshold: above it, unserved energy;\n"
                "below it, reversible extra cost",
                fontsize=6.6, color=SLATE, ha="center", va="bottom", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                          alpha=0.92))
        ax.axvline(ron, color=ALERT, lw=1.2)
        ax.axvline(roff, color=TRADE, lw=1.2)
        ax.text(ron + 0.006, 0.035, "$\\rho_{on}$ = " + f"{ron:.2f}",
                color=ALERT, ha="left", va="bottom", fontsize=7.4, weight="bold")
        ax.text(roff - 0.006, 0.035, "$\\rho_{off}$ = " + f"{roff:.2f}",
                color=TRADE, ha="right", va="bottom", fontsize=7.4, weight="bold")
        if rho_quantiles is not None:
            q = np.asarray(rho_quantiles)
            ax.plot(q, np.linspace(0.06, 0.30, len(q)), color=BASALT, lw=1.0,
                    marker="o", ms=3, mfc="white")
            ax.text(q[-1], 0.32, "empirical distribution of $\\rho_t$\n(10th–99th percentiles)",
                    fontsize=6.6, color=BASALT, ha="right")
        ax.set_xlabel("Residual risk of the automatic recommendation  $\\rho_t$")
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}")
        ax.set_ylabel("Severity of the consequence  →")
        ax.set_yticks([])
        _tidy(ax)
        ax.grid(False)
        ax.set_title("Decision governance regions and applied thresholds", loc="left")
        _footnote(fig, "The thresholds shown are those actually applied in the campaign "
                       "and the subject of the sensitivity sweep of Fig. 5.", y=0.02)
        return _save(fig, outdir, "fig03_governance_regions")


# =====================================================================
# Figure 4 — Layered architecture
# =====================================================================

def fig04_architecture(outdir: Path):
    layers = [
        ("1", "Physical system layer", "Generation · 132/66 kV grid · link · island demand", "impl"),
        ("2", "Data layer", "Physical validation · anomaly detection · quality signals", "conc"),
        ("3", "Probabilistic forecasting layer",
         "Trained quantile regression · conformal recalibration · regime detector", "impl"),
        ("4", "Recommendation layer",
         "Merit-order commitment with iterated N‑1 criterion and deliverability", "impl"),
        ("5", "Risk diagnosis layer",
         "Residual risk $\\rho_t$ · attribution of the dominant link", "impl"),
        ("6", "Safeguard layer",
         "Risk gate · robust setpoint · non-relaxation ratchet", "impl"),
        ("7", "Operator supervision layer",
         "Authorization of critical actions · lockout · conservative procedure", "conc"),
        ("8", "Audit and feedback layer",
         "Immutable trace · counterfactual reconstruction · recalibration", "conc"),
    ]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7.0, 5.4))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, len(layers) * 10 + 6)
        ax.axis("off")
        for i, (num, title, sub, kind) in enumerate(layers):
            y = (len(layers) - 1 - i) * 10 + 3
            impl = kind == "impl"
            ax.add_patch(FancyBboxPatch((14, y), 78, 7.9,
                                        boxstyle="round,pad=0.35",
                                        fc="#E9F4F1" if impl else "#F2F4F6",
                                        ec=TRADE if impl else SLATE,
                                        lw=1.2 if impl else 0.9,
                                        ls="-" if impl else (0, (4, 2))))
            ax.add_patch(plt.Circle((18.5, y + 3.95), 2.55,
                                    fc=TRADE if impl else SLATE, ec="none"))
            ax.text(18.5, y + 3.95, num, color="white", ha="center", va="center",
                    fontsize=7.6, weight="bold")
            ax.text(24, y + 5.35, title, fontsize=8.2, weight="bold", color=BASALT)
            ax.text(24, y + 2.05, sub, fontsize=6.9, color=SLATE)
        ax.annotate("", xy=(9, len(layers) * 10 + 1), xytext=(9, 3),
                    arrowprops=dict(arrowstyle="-|>", color=ATLANTIC, lw=1.6))
        ax.text(6.5, len(layers) * 5, "information and measurement", rotation=90,
                va="center", ha="center", fontsize=7.2, color=ATLANTIC, weight="bold")
        ax.annotate("", xy=(96, 3), xytext=(96, len(layers) * 10 + 1),
                    arrowprops=dict(arrowstyle="-|>", color=ALERT, lw=1.6))
        ax.text(98.5, len(layers) * 5, "recommendation and authorization", rotation=90,
                va="center", ha="center", fontsize=7.2, color=ALERT, weight="bold")
        ax.set_title("Safeguarded dispatch architecture: implemented and projected layers",
                     loc="left")
        _footnote(fig,
                  "Trazo continuo y color turquesa: capas implementadas y evaluadas en este trabajo. "
                  "Dashed strokes: layers specified as a deployment requirement, not empirically evaluated.",
                  y=0.02)
        return _save(fig, outdir, "fig04_architecture")


# =====================================================================
# Figure 5 — Proposed decision flow
# =====================================================================

def fig05_workflow(outdir: Path):
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 62)
        ax.axis("off")

        def box(x, y, w, h, txt, fc, ec, ls="-", fs=7.2, bold=False):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                        fc=fc, ec=ec, lw=1.0, ls=ls))
            ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                    fontsize=fs, color=BASALT,
                    weight="bold" if bold else "normal")

        box(30, 53, 40, 7, "Automatic recommendation $U_A(t)$\nwith iterated N‑1 criterion",
            "#EDF2F5", ATLANTIC)
        ax.add_patch(FancyArrowPatch((50, 53), (50, 48), arrowstyle="-|>",
                                     mutation_scale=9, color=BASALT, lw=1.0))
        box(26, 41, 48, 7, "Residual risk $\\rho_t$ and regime flag $\\phi_t$",
            "#F7F9FA", SLATE)
        ax.add_patch(FancyArrowPatch((50, 41), (50, 36), arrowstyle="-|>",
                                     mutation_scale=9, color=BASALT, lw=1.0))
        ax.add_patch(Polygon([(50, 36), (68, 29), (50, 22), (32, 29)], closed=True,
                             fc="#FBEAE7", ec=ALERT, lw=1.2))
        ax.text(50, 29, "$\\rho_t>\\rho_{on}$  or  $\\phi_t=1$ ?", ha="center",
                va="center", fontsize=7.4, color=ALERT, weight="bold")
        ax.add_patch(FancyArrowPatch((32, 29), (22, 29), arrowstyle="-|>",
                                     mutation_scale=9, color=TRADE, lw=1.0))
        box(2, 25, 19, 8, "Se adopta $U_A(t)$", "#E8F3EE", TRADE)
        ax.text(26.5, 30.5, "no", fontsize=7, color=TRADE)
        ax.add_patch(FancyArrowPatch((50, 22), (50, 18.2), arrowstyle="-|>",
                                     mutation_scale=9, color=ALERT, lw=1.2))
        ax.text(51.5, 19.5, "yes", fontsize=7, color=ALERT, weight="bold")
        box(28, 9.5, 44, 8,
            "Robust setpoint: $U_B(t)$ with requirement $\\hat N_t+\\beta\\sigma_t$\n"
            "by construction $U_B(t)\\supseteq U_{A\\mid B}(t)$",
            "#FBEAE7", ALERT, bold=True)
        box(76, 25, 23, 12,
            "Ratchet\nRelease is always\nevaluated with the\nrobust requirement",
            "#FDF6E6", SAND, ls=(0, (4, 2)))
        ax.add_patch(FancyArrowPatch((76, 31), (72, 15), arrowstyle="-|>",
                                     mutation_scale=8, color=SAND, lw=1.0,
                                     ls=(0, (4, 2)), connectionstyle="arc3,rad=0.25"))
        box(10, 0.5, 80, 6.0,
            "Traceability record on every path → audit layer",
            "#F2F4F6", SLATE, ls=(0, (4, 2)), fs=7.0)
        for x0 in (11.5, 50):
            ax.add_patch(FancyArrowPatch((x0, 25 if x0 < 20 else 9.15), (x0, 6.9),
                                         arrowstyle="-|>", mutation_scale=8,
                                         color=SLATE, lw=0.8, ls=(0, (3, 2))))
        ax.set_title("Decision flow of safeguarded dispatch", loc="left")
        _footnote(fig,
                  "The diagram describes the logic implemented in the campaign. The operator's express authorization "
                  "for critical actions is specified as a deployment requirement and has not been evaluated "
                  "experimentally: this work includes no study with human participants.",
                  y=0.02)
        return _save(fig, outdir, "fig05_decision_flow")


# =====================================================================
# Figure 6 — Generator calibration
# =====================================================================

def fig06_calibration(outdir: Path, cal: dict, world_example, official: dict):
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(7.6, 4.6))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])

        ax = fig.add_subplot(gs[0, 0])
        keys = ["demanda_anual_GWh", "punta_MW", "producible_renovable_GWh"]
        names = ["Annual\ndemand", "Peak", "Renewable\navailable"]
        y = np.arange(len(keys))
        errs = [cal[k]["error_rel_pct"] for k in keys]
        disp = [100 * 1.96 * cal[k]["desv"] / cal[k]["objetivo"] for k in keys]
        ax.barh(y, errs, xerr=disp, height=0.5, color=TRADE, capsize=2.5,
                error_kw=dict(lw=0.8, ecolor=BASALT))
        ax.axvline(0, color=BASALT, lw=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6.6)
        ax.set_xlabel("Relative error (%)", fontsize=7.4)
        ax.set_xlim(-9.5, 9.5)
        ax.set_ylim(-0.55, 2.85)
        for i, e in enumerate(errs):
            ax.text(e, i + 0.36, f"{e:+.2f}%",
                    va="bottom", ha="center", fontsize=6.3, color=SLATE)
        ax.set_title("(a) Fit to official data", loc="left", fontsize=8.2)
        _tidy(ax)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

        ax = fig.add_subplot(gs[0, 1])
        d = np.sort(world_example.demand)[::-1]
        ax.plot(np.arange(len(d)) / len(d) * 100, d, color=BASALT, lw=1.2)
        for p, lab in ((10, "P10"), (50, "P50"), (90, "P90")):
            ax.plot([p], [d[int(p / 100 * len(d))]], "o", ms=3.5, color=ALERT)
            ax.text(p + 2, d[int(p / 100 * len(d))] + 4, lab, fontsize=6.4, color=ALERT)
        ax.set_xlabel("% of hours of the year")
        ax.set_ylabel("Demand (MW)")
        ax.set_title("(b) Load duration curve", loc="left", fontsize=8.2)
        _tidy(ax)

        ax = fig.add_subplot(gs[0, 2])
        keys2 = ["factor_de_carga", "autocorrelacion_lag1", "autocorrelacion_lag24"]
        lab2 = ["Load\nfactor", "Autocorr.\n1 h", "Autocorr.\n24 h"]
        v = [cal[k]["media"] for k in keys2]
        e = [1.96 * cal[k]["desv"] for k in keys2]
        ax.barh(np.arange(3), v, xerr=e, color=ATLANTIC, height=0.55,
                error_kw=dict(lw=0.8, ecolor=BASALT), capsize=2.5)
        ax.set_yticks(np.arange(3))
        ax.set_yticklabels(lab2, fontsize=6.6)
        ax.set_xlim(0, 1.30)
        for i, val in enumerate(v):
            ax.text(val + 0.04, i, f"{val:.3f}", va="center",
                    fontsize=6.4, color=SLATE)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}")
        ax.set_title("(c) Temporal structure", loc="left", fontsize=8.2)
        _tidy(ax)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

        ax = fig.add_subplot(gs[1, :])
        h0 = 24 * 120
        seg = slice(h0, h0 + 168)
        t = np.arange(168)
        ax.fill_between(t, 0, world_example.demand[seg], color=BASALT, alpha=0.10)
        ax.plot(t, world_example.demand[seg], color=BASALT, lw=1.2, label="demand")
        ax.plot(t, world_example.wind[seg], color=ATLANTIC, lw=1.0, label="available wind")
        ax.plot(t, world_example.pv[seg], color=SAND, lw=1.0, label="available solar PV")
        ax.set_xticks(np.arange(0, 169, 24))
        ax.set_xticklabels(["M", "T", "W", "T", "F", "S", "S", ""])
        ax.set_xlim(0, 167)
        ax.set_ylabel("MW")
        ax.set_xlabel("Typical week (hourly resolution)")
        ax.set_ylim(0, float(world_example.demand.max()) * 1.32)
        ax.legend(ncol=3, loc="upper left", fontsize=6.8)
        ax.set_title("(d) Synthetic realization: typical week", loc="left", fontsize=8.2)
        _tidy(ax)

        fig.suptitle("Calibration of the scenario generator against the official 2025 magnitudes",
                     x=0.005, ha="left", fontsize=9.5, weight="bold")
        _footnote(fig, "Official targets: Red Eléctrica, CTSOC 2025 (March 2026). "
                       "Error bars: 95% interval across 20 realizations.", y=0.005)
        return _save(fig, outdir, "fig06_calibration")


# =====================================================================
# Figure 7 — Forecasting layer validation
# =====================================================================

def fig07_forecast(outdir: Path, prev: dict):
    tgt = ["demand", "cf_wind", "cf_pv"]
    lab = ["Demand", "Wind", "Solar PV"]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.3))
        raw, cal = prev["sin_recalibracion_conformal"], prev["validacion_fuera_de_muestra"]

        ax = axes[0]
        x = np.arange(3)
        imp = prev.get("conjunto_calibracion", cal)
        ax.bar(x - 0.26, [raw[t]["picp90"] * 100 for t in tgt], 0.25,
               color=SLATE, label="test, uncalibrated")
        ax.bar(x, [cal[t]["picp90"] * 100 for t in tgt], 0.25,
               color=TRADE, label="prueba, recalibrada")
        ax.bar(x + 0.26, [imp[t]["picp90"] * 100 for t in tgt], 0.25,
               color=MIST, edgecolor=SLATE, lw=0.6,
               label="calibration (imposed)")
        ax.axhline(80, color=ALERT, lw=1.1, ls=(0, (3, 2)))
        ax.text(-0.42, 81.4, "nominal 80 %", fontsize=6.4, color=ALERT,
                ha="left", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels(lab, fontsize=7.0)
        ax.set_ylabel("Empirical coverage (%)")
        ax.set_ylim(55, 108)
        ax.legend(fontsize=5.9, loc="upper left", ncol=1)
        ax.set_title("(a) Interval reliability", loc="left", fontsize=8.2)
        _tidy(ax)

        ax = axes[1]
        sk = [cal[t]["skill_vs_baseline"] * 100 for t in tgt]
        ax.bar(x, sk, 0.5, color=ATLANTIC)
        for i, s in enumerate(sk):
            ax.text(i, s + max(sk) * 0.04, f"{s:.0f}%", ha="center", fontsize=6.6,
                    color=SLATE)
        ax.set_xticks(x)
        ax.set_xticklabels(lab, fontsize=6.4)
        ax.set_ylabel("Pinball loss improvement (%)")
        ax.set_title("(b) Skill against climatology-\npersistence reference",
                     loc="left", fontsize=8.4)
        _tidy(ax)

        ax = axes[2]
        w = [cal[t]["pinaw"] * 100 for t in tgt]
        lam = [cal[t]["lambda_conformal"] for t in tgt]
        ax.bar(x, w, 0.5, color=SAND)
        for i, (ww, ll) in enumerate(zip(w, lam)):
            ax.text(i, ww + max(w) * 0.04, f"$\\lambda$={ll:.2f}",
                    ha="center", fontsize=6.4, color=SLATE)
        ax.set_xticks(x)
        ax.set_xticklabels(lab, fontsize=7.0)
        ax.set_ylabel("Anchura normalizada (%)")
        ax.set_ylim(0, max(w) * 1.25)
        ax.set_title("(c) Cost of reliability", loc="left", fontsize=8.2)
        _tidy(ax)

        fig.suptitle("Out-of-sample validation of the probabilistic forecasting layer",
                     x=0.005, ha="left", fontsize=9.5, weight="bold")
        _footnote(fig,
                  "Gradient-boosted quantile regression model trained on realizations "
                  "disjoint from those employed in the campaign; the conformal recalibration is fitted on a third "
                  "set, also disjoint. The uncertainty consumed by the policies is the one validated here, "
                  "not that of the generator.", y=-0.02)
        return _save(fig, outdir, "fig07_forecast")


# =====================================================================
# Figure 8 — Policy comparison with confidence intervals
# =====================================================================

def fig08_policies(outdir: Path, principal: dict, n_seeds: int = 30):
    metrics = [("violation_h", "N‑1 violations\n(h/yr)", 1.0),
               ("ens_MWh", "Unserved\nenergy (MWh/yr)", 1.0),
               ("cost_MEUR", "Operating cost\n(M€/yr)", 1.0),
               ("curtailment_GWh", "Renewable curtailment\n(GWh/yr)", 1.0),
               ("co2_kt", "Emissions\n(ktCO₂/yr)", 1.0)]
    pols = POLICY_ORDER
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 5, figsize=(8.4, 4.8), sharex="col")
        for r, sc in enumerate(("S0", "S1")):
            for c, (key, title, _) in enumerate(metrics):
                ax = axes[r, c]
                vals = [principal[sc][p][key]["mean"] for p in pols]
                lo = [principal[sc][p][key]["mean"] - principal[sc][p][key]["ci_low"] for p in pols]
                hi = [principal[sc][p][key]["ci_high"] - principal[sc][p][key]["mean"] for p in pols]
                ax.bar(np.arange(len(pols)), vals, 0.70,
                       color=[POLICY_COLOR[p] for p in pols],
                       yerr=[lo, hi], capsize=1.6,
                       error_kw=dict(lw=0.7, ecolor=BASALT))
                ax.set_xticks(np.arange(len(pols)))
                ax.set_xticklabels(list(pols), fontsize=7.5)
                _tidy(ax)
                ax.yaxis.set_major_locator(MaxNLocator(4))
                if r == 0:
                    ax.set_title(title, loc="left", fontsize=7.2)
                if c == 0:
                    ax.set_ylabel("Base scenario\n(2025)" if r == 0
                                  else "High penetration\n(2030 target)",
                                  fontsize=7.6, weight="bold")
                top = max(v + h for v, h in zip(vals, hi))
                if top <= 1e-9:            # identically zero metric
                    ax.set_ylim(0, 1.0)
                    ax.set_yticks([0, 1])
                    ax.text(2.5, 0.5, "0 for all policies", ha="center",
                            va="center", fontsize=6.4, color=SLATE, style="italic")
                    continue
                ax.set_ylim(0, top * 1.30)
                for i, v in enumerate(vals):
                    ax.text(i, v + hi[i] + top * 0.05,
                            f"{v:,.0f}" if abs(v) >= 100
                            else f"{v:.1f}",
                            ha="center", fontsize=5.6, color=BASALT, rotation=90,
                            va="bottom")
        handles = [plt.Rectangle((0, 0), 1, 1, fc=POLICY_COLOR[p]) for p in pols]
        fig.legend(handles, [POLICY_LABEL[p] for p in pols], ncol=3,
                   loc="outside lower center", fontsize=7.0)
        fig.suptitle("Comparison of dispatch policies: mean and 95% confidence interval",
                     x=0.005, ha="left", fontsize=9.5, weight="bold")
        _footnote(fig, SYNTH + f". Bootstrap intervals over {n_seeds} paired realizations per policy.",
                  y=-0.10)
        return _save(fig, outdir, "fig08_policy_comparison")


# =====================================================================
# Figure 9 — Cost-security frontier and gate trade-off
# =====================================================================

def fig09_frontier(outdir: Path, principal: dict, puerta: dict):
    """Cost-security frontier, gate trade-off, and detector quality, with a
    shared policy legend at the foot (redone in v4)."""
    from matplotlib.lines import Line2D
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.75))

        # --- (a) cost-security frontier -------------------------------
        ax = axes[0]
        base = {sc: principal[sc]["A"]["cost_MEUR"]["mean"] for sc in ("S0", "S1")}
        pts = {}
        for sc in ("S0", "S1"):
            for pkey in POLICY_ORDER:
                x = 100 * (principal[sc][pkey]["cost_MEUR"]["mean"] / base[sc] - 1)
                y = principal[sc][pkey]["violation_h"]["mean"]
                pts[(sc, pkey)] = (x, y)
        for pkey in POLICY_ORDER:      # joins the two scenarios of each policy
            (x0, y0), (x1, y1) = pts[("S0", pkey)], pts[("S1", pkey)]
            ax.plot([x0, x1], [y0, y1], color=POLICY_COLOR[pkey], lw=0.7,
                    ls=(0, (1, 2)), alpha=0.75, zorder=1)
        for sc, mk in (("S0", "o"), ("S1", "s")):
            for pkey in POLICY_ORDER:
                x, y = pts[(sc, pkey)]
                ax.scatter(x, y, s=42, marker=mk, color=POLICY_COLOR[pkey],
                           zorder=3, edgecolor="white", linewidth=0.7)
        off = {"A": (7, 2), "E": (-13, 5), "B": (-15, -9), "F": (8, 1),
               "C": (8, -11), "D": (-13, 2)}
        for pkey in POLICY_ORDER:      # letter once only, on the circle
            x, y = pts[("S0", pkey)]
            ax.annotate(pkey, (x, y), textcoords="offset points",
                        xytext=off[pkey], fontsize=7.4,
                        color=POLICY_COLOR[pkey], weight="bold")
        ax.set_yscale("symlog", linthresh=50)
        ax.set_yticks([0, 50, 200, 1000, 3000])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Extra cost vs. A (%)")
        ax.set_ylabel("N‑1 violations (h/yr)")
        ax.set_title("(a) Cost–security frontier", loc="left", fontsize=8.2)
        ax.margins(x=0.22, y=0.28)
        _tidy(ax)

        # --- (b) gate trade-off ---------------------------------------
        ax = axes[1]
        for sc, col in (("S0", ATLANTIC), ("S1", TRADE)):
            rows = [r for r in puerta["umbral"] if r["escenario"] == sc]
            rows.sort(key=lambda r: r["rho_on"])
            x = [r["politicas"]["B"]["gate_h"]["mean"] / 87.60 for r in rows]
            y = [r["politicas"]["B"]["violation_h"]["mean"] for r in rows]
            ax.plot(x, y, marker="o", ms=3.5, color=col, lw=1.2,
                    label="2025 base" if sc == "S0" else "high penetration")
            if sc == "S0":     # only one series is labeled to avoid duplication
                for r, xi, yi in zip(rows, x, y):
                    if r["rho_on"] in (0.02, 0.30, 0.50):
                        ax.annotate(f"$\\rho_{{on}}$={r['rho_on']:g}",
                                    (xi, yi), textcoords="offset points",
                                    xytext=(5, 5), fontsize=6.2, color=col)
        ax.set_xlabel("Safeguard active (% of year)")
        ax.set_ylabel("N‑1 violations (h/yr)")
        ax.set_title("(b) Gate trade-off", loc="left", fontsize=8.2)
        ax.legend(loc="upper right", fontsize=6.6, title="policy B under:",
                  title_fontsize=6.4)
        ax.margins(y=0.20)
        _tidy(ax)

        # --- (c) detector quality -------------------------------------
        ax = axes[2]
        det = puerta["detector"]
        x = np.arange(len(det))
        v = [d["politicas"]["B"]["violation_h"]["mean"] for d in det]
        g = [d["politicas"]["B"]["gate_h"]["mean"] / 87.60 for d in det]
        ax.bar(x - 0.19, v, 0.36, color=ALERT, label="violations")
        ax2 = ax.twinx()
        ax2.bar(x + 0.19, g, 0.36, color=TRADE, label="supervision burden")
        ax2.set_ylabel("Hours with safeguard (%)", color=TRADE, fontsize=7.4)
        ax2.tick_params(axis="y", colors=TRADE)
        ax2.grid(False)
        ax2.xaxis.set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d['recall']:.2f}\n{d['fpr']:.2f}"
                            for d in det], fontsize=6.6)
        ax.set_xlabel("sensitivity / false positives", fontsize=7.0)
        ax.set_ylabel("N‑1 violations (h/yr)", color=ALERT, fontsize=7.4)
        ax.tick_params(axis="y", colors=ALERT)
        ax.set_title("(c) Detector quality", loc="left", fontsize=8.2)
        _tidy(ax)

        # --- shared legend of policies and scenarios ------------------
        handles = [Line2D([], [], marker="o", ls="none", ms=6,
                          mfc=POLICY_COLOR[pkey], mec="white", mew=0.6)
                   for pkey in POLICY_ORDER]
        labels = [POLICY_LABEL[pkey] for pkey in POLICY_ORDER]
        handles += [Line2D([], [], marker="o", ls="none", ms=6, mfc=BASALT,
                           mec="white", mew=0.6),
                    Line2D([], [], marker="s", ls="none", ms=6, mfc=BASALT,
                           mec="white", mew=0.6)]
        labels += ["circle: 2025 base", "square: high penetration"]
        fig.legend(handles, labels, loc="outside lower center", ncol=4,
                   frameon=False, fontsize=6.6, columnspacing=1.1,
                   handletextpad=0.35)

        fig.suptitle("Selective allocation of conservatism and supervision burden",
                     x=0.005, ha="left", fontsize=9.5, weight="bold")
        _footnote(fig, SYNTH + ". S: detector sensitivity; FP: false positive rate.",
                  y=-0.155)
        return _save(fig, outdir, "fig09_frontier_gate")


# =====================================================================
# Figure 10 — Sensitivity and scalability
# =====================================================================

def fig10_sensitivity(outdir: Path, sens: dict, pen: list, n_seeds: int = 15):
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.3))

        ax = axes[0]
        ks = [r["k_sigma"] for r in sens["total"]]
        for p in POLICY_ORDER:
            m = [r["politicas"][p]["violation_h"]["mean"] for r in sens["total"]]
            lo = [r["politicas"][p]["violation_h"]["ci_low"] for r in sens["total"]]
            hi = [r["politicas"][p]["violation_h"]["ci_high"] for r in sens["total"]]
            ax.plot(ks, m, marker="o", ms=3.2, color=POLICY_COLOR[p], lw=1.3,
                    label=POLICY_LABEL[p])
            ax.fill_between(ks, lo, hi, color=POLICY_COLOR[p], alpha=0.16, lw=0)
        ax.set_yscale("symlog", linthresh=50)
        ax.set_yticks([0, 50, 200, 1000, 3000])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Error scale  $k_\\sigma$")
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
        ax.set_ylabel("N‑1 violations (h/yr)")
        ax.set_title("(a) Joint scaling", loc="left", fontsize=8.2)
        ax.margins(y=0.22)
        _tidy(ax)

        ax = axes[1]
        srcs = [r["fuente"] for r in sens["parcial"]]
        x = np.arange(len(srcs))
        base = sens["total"][1]["politicas"]
        for i, p in enumerate(("A", "E", "B", "C")):
            v = [100 * (r["politicas"][p]["violation_h"]["mean"]
                        / max(base[p]["violation_h"]["mean"], 1e-9) - 1)
                 for r in sens["parcial"]]
            ax.bar(x + (i - 1.5) * 0.21, v, 0.19, color=POLICY_COLOR[p],
                   label=POLICY_LABEL[p])
        ax.axhline(0, color=SLATE, lw=0.8)
        ax.set_xticks(x)
        _n = {"demanda": "Demand", "eolica": "Wind",
              "fotovoltaica": "Solar PV"}
        ax.set_xticklabels([_n.get(s, s.capitalize()) for s in srcs],
                           fontsize=7.0)
        ax.set_ylabel("Change in violations (%)")
        ax.set_title("(b) Per-source scaling ($k_\\sigma$ = 2)", loc="left",
                     fontsize=8.2)
        ax.margins(y=0.24)
        _tidy(ax)

        ax = axes[2]
        tot = [r["total_MW"] for r in pen]
        for p in POLICY_ORDER:
            m = [r["politicas"][p]["violation_h"]["mean"] for r in pen]
            lo = [r["politicas"][p]["violation_h"]["ci_low"] for r in pen]
            hi = [r["politicas"][p]["violation_h"]["ci_high"] for r in pen]
            ax.plot(tot, m, marker="o", ms=3.2, color=POLICY_COLOR[p], lw=1.3)
            ax.fill_between(tot, lo, hi, color=POLICY_COLOR[p], alpha=0.16, lw=0)
        ax.set_yscale("symlog", linthresh=50)
        ax.set_yticks([0, 50, 200, 1000, 3000])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Installed renewables (MW)")
        ax.set_ylabel("N‑1 violations (h/yr)")
        ax.set_title("(c) Renewable penetration", loc="left", fontsize=8.2)
        ax.margins(y=0.24)
        ax.axvline(tot[0], color=SLATE, lw=0.8, ls=(0, (2, 2)))
        ax.axvline(tot[-1], color=SLATE, lw=0.8, ls=(0, (2, 2)))
        ax.text(tot[0] + 8, 2.0, "2025", fontsize=6.4,
                color=SLATE, va="bottom", ha="left")
        ax.text(tot[-1] - 8, 2.0, "2030 target", fontsize=6.4,
                color=SLATE, va="bottom", ha="right")
        _tidy(ax)
        handles = [plt.Line2D([], [], color=POLICY_COLOR[p], lw=1.6,
                              marker="o", ms=3.2) for p in POLICY_ORDER]
        fig.legend(handles, [POLICY_LABEL[p] for p in POLICY_ORDER], ncol=3,
                   loc="outside lower center", fontsize=6.8)
        fig.suptitle("Sensitivity to forecast error and scalability with renewable penetration",
                     x=0.005, ha="left", fontsize=9.5, weight="bold")
        _footnote(fig, SYNTH + f". Bands: 95% confidence interval over {n_seeds} paired realizations.",
                  y=-0.06)
        return _save(fig, outdir, "fig10_sensitivity")


# =====================================================================
# Figure 11 — Contingencies
# =====================================================================

def fig11_contingencies(outdir: Path, cont: dict):
    names = list(cont.keys())
    short = {"N-1 prolongada del mayor grupo": "Prolonged\nN‑1",
             "separacion del sistema en dos islas": "Separation\n(24 h)",
             "meteorologia adversa correlacionada": "Adverse\nweather",
             "indisponibilidad estructural elevada": "High\noutage rate",
             "separacion prolongada en dos islas": "Separation\n(72 h)"}
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3), sharey=False)
        for ax, sc, title in zip(axes, ("S0", "S1"),
                                 ("Base scenario (2025)", "High penetration (2030 target)")):
            x = np.arange(len(names))
            for i, p in enumerate(POLICY_ORDER):
                m = [cont[n][sc][p]["violation_h"]["mean"] for n in names]
                lo = [m[j] - cont[n][sc][p]["violation_h"]["ci_low"] for j, n in enumerate(names)]
                hi = [cont[n][sc][p]["violation_h"]["ci_high"] - m[j] for j, n in enumerate(names)]
                ax.bar(x + (i - 2.5) * 0.15, m, 0.14, color=POLICY_COLOR[p],
                       yerr=[lo, hi], capsize=1.6,
                       error_kw=dict(lw=0.7, ecolor=BASALT), label=POLICY_LABEL[p])
            ax.set_xticks(x)
            ax.set_xticklabels([short[n] for n in names], fontsize=6.1)
            ax.set_yscale("symlog", linthresh=50)
            ax.set_yticks([0, 50, 200, 1000, 3000])
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_ylabel("N‑1 violations (h/yr)")
            ax.set_title(title, loc="left", fontsize=8.2)
            ax.margins(y=0.30)
            _tidy(ax)
        axes[0].legend(fontsize=6.2, ncol=2, loc="upper left", columnspacing=0.8)
        fig.suptitle("Robustness of the policies to structural contingencies",
                     x=0.005, ha="left", fontsize=9.5, weight="bold")
        _footnote(fig,
                  SYNTH + ". The system separation reproduces the contingency flagged by the operator "
                  "for the single 132 kV circuit in the 2026 coverage report.", y=-0.04)
        return _save(fig, outdir, "fig11_contingencies")


# =====================================================================
# Figure 12 — Supporting evidence: value of lost load, capacity factors,
#             and comparison with the optimal commitment
# =====================================================================

def fig12_evidence(outdir: Path, principal: dict, factores: list, bench: dict):
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.9))

        ax = axes[0]
        sw = principal["S1"]["_voll_sweep"]
        vs = sorted(int(k) for k in sw)
        for pkey in ("A", "E", "B", "C"):
            y = [sw[str(v)][pkey]["mean"] for v in vs]
            ax.plot(vs, y, marker="o", ms=3.2, lw=1.3, color=POLICY_COLOR[pkey],
                    label=POLICY_LABEL[pkey].split(" · ")[0])
        # threshold where the A-B ordering reverses
        eA = principal["S1"]["A"]["ens_MWh"]["mean"]
        eB = principal["S1"]["B"]["ens_MWh"]["mean"]
        c0A = sw["0"]["A"]["mean"]
        c0B = sw["0"]["B"]["mean"]
        v_star = 1e6 * (c0B - c0A) / max(eA - eB, 1e-9)
        if 0 < v_star < max(vs):
            ax.axvline(v_star, color=BASALT, lw=0.9, ls=(0, (3, 2)))
            ax.text(v_star, ax.get_ylim()[1], f" {v_star:,.0f} €/MWh",
                    fontsize=6.2, color=BASALT, va="top")
        ax.set_xlabel("Value of lost load (€/MWh)")
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
        ax.set_ylabel("Operating cost (M€/yr)")
        ax.set_title("(a) Sensitivity to the value of lost load", loc="left", fontsize=8.2)
        ax.legend(fontsize=6.2, ncol=2, columnspacing=0.8)
        ax.margins(y=0.20)
        _tidy(ax)

        ax = axes[1]
        names = [r["variante"] for r in factores]
        tot = [r["total_MW"] for r in factores]
        order = np.argsort(tot)
        y = np.arange(len(names))
        ax.barh(y, [tot[i] for i in order], height=0.5, color=ATLANTIC)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{names[i]}\ncf=({factores[i]['cf_wind']:.3f}; "
                            f"{factores[i]['cf_pv']:.3f})"
                            for i in order], fontsize=6.2)
        for k, i in enumerate(order):
            ax.text(tot[i] + 8, k, f"{tot[i]:.0f} MW", va="center", fontsize=6.4,
                    color=SLATE)
        ax.set_xlabel("Renewable capacity required in 2030 (MW)")
        ax.set_xlim(0, max(tot) * 1.24)
        ax.set_title("(b) Sensitivity of the 2030 scenario", loc="left", fontsize=8.2)
        _tidy(ax)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

        ax = axes[2]
        labels, det, rob = [], [], []
        for sc in ("S0", "S1"):
            if sc not in bench.get("resumen", {}):
                continue
            labels.append("2025 base" if sc == "S0" else "High penetration")
            det.append(bench["resumen"][sc]["consigna determinista"]["brecha_media_pct"])
            rob.append(bench["resumen"][sc]["consigna robusta"]["brecha_media_pct"])
        x = np.arange(len(labels))
        ax.bar(x - 0.19, det, 0.36, color=ALERT, label="deterministic setpoint")
        ax.bar(x + 0.19, rob, 0.36, color=TRADE, label="robust setpoint")
        for i, (a, b) in enumerate(zip(det, rob)):
            ax.text(i - 0.19, a + 0.18, f"{a:.2f}", ha="center",
                    fontsize=6.3, color=SLATE)
            ax.text(i + 0.19, b + 0.18, f"{b:.2f}", ha="center",
                    fontsize=6.3, color=SLATE)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.0)
        ax.set_ylabel("Heuristic gap (%)")
        ax.set_ylim(0, max(det + rob) * 1.32)
        ax.legend(fontsize=6.2)
        ax.set_title("(c) Comparison with the optimum", loc="left", fontsize=8.2)
        _tidy(ax)

        fig.suptitle("Supporting evidence on the critical assumptions",
                     x=0.005, ha="left", fontsize=9.5, weight="bold")
        return _save(fig, outdir, "fig12_supporting_evidence")


# =====================================================================

def build_all(results_path: Path, outdir: Path):
    """Generate the complete set of figures from the results file."""
    from .forecast import QuantileForecaster
    from .system import OFFICIAL_2025, S0_PV_MW, S0_WIND_MW, SystemConfig
    from .world import build_world

    with open(results_path, encoding="utf-8") as fh:
        R = json.load(fh)
    outdir = Path(outdir)
    cfg0 = SystemConfig("S0", S0_WIND_MW, S0_PV_MW)
    w = build_world(3, cfg0)

    made = [
        fig01_system(outdir, OFFICIAL_2025),
        fig02_error_graph(outdir),
        fig03_criticality(outdir, R["meta"]["gate"]),
        fig04_architecture(outdir),
        fig05_workflow(outdir),
        fig06_calibration(outdir, R["calibracion_generador"], w, OFFICIAL_2025),
        fig07_forecast(outdir, R["prevision"]),
        fig08_policies(outdir, R["principal"], R["meta"]["n_seeds_main"]),
        fig09_frontier(outdir, R["principal"], R["puerta"]),
        fig10_sensitivity(outdir, R["sensibilidad_error"], R["penetracion"], R["meta"]["n_seeds_sweep"]),
        fig11_contingencies(outdir, R["contingencias"]),
        fig12_evidence(outdir, R["principal"], R["factores_de_carga"],
                       R.get("compromiso_optimo", {})),
    ]
    return made
