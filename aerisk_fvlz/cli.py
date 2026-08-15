"""cli.py — Command-line interface of the package.

Full reproduction from a clean installation:

    pip install -e .
    aerisk-fvlz all --outdir results

Each subcommand writes to the path given through `pathlib`; no absolute path
is hard-coded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="aerisk-fvlz",
        description="Reproducible evaluation of safeguarded dispatch "
                    "policies in the Fuerteventura-Lanzarote power system.")
    ap.add_argument("command",
                    choices=["campaign", "figures", "all", "calibration"],
                    help="block to run")
    ap.add_argument("--outdir", type=Path, default=Path("results"),
                    help="output directory (default: ./results)")
    ap.add_argument("--figdir", type=Path, default=None,
                    help="figure directory (default: <outdir>/figures)")
    ap.add_argument("--seeds-main", type=int, default=30)
    ap.add_argument("--seeds-sweep", type=int, default=15)
    ap.add_argument("--rho-on", type=float, default=0.30)
    ap.add_argument("--parts", type=str, default="",
                    help="comma-separated list of campaign blocks")
    a = ap.parse_args(argv)

    outdir = a.outdir
    figdir = a.figdir or (outdir / "figures")
    outdir.mkdir(parents=True, exist_ok=True)

    if a.command == "calibration":
        from .system import OFFICIAL_2025, S0_PV_MW, S0_WIND_MW, SystemConfig
        from .world import build_world, calibration_report
        cfg = SystemConfig("S0", S0_WIND_MW, S0_PV_MW)
        rep = calibration_report([build_world(s, cfg) for s in range(1, 21)],
                                 OFFICIAL_2025)
        print(json.dumps(rep, indent=1, ensure_ascii=False))
        return 0

    if a.command in ("campaign", "all"):
        from .campaign import ALL_PARTS, run_campaign
        parts = tuple(p.strip() for p in a.parts.split(",") if p.strip()) or ALL_PARTS
        run_campaign(outdir, n_seeds_main=a.seeds_main,
                     n_seeds_sweep=a.seeds_sweep, rho_on=a.rho_on, parts=parts)

    if a.command in ("figures", "all"):
        from .figures import build_all
        made = build_all(outdir / "campaign_results.json", figdir)
        print(f"{len(made)} figures written to {figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
