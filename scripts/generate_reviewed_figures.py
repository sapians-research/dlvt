#!/usr/bin/env python3
"""Reproduce the two reviewed DLVT manuscript figures.

This is the public, standalone plotting boundary.  It deliberately exposes
only Figures 2 and 10, whose scenarios live in :mod:`dlvt.scenarios`.  Legacy
development figures are not part of the public release.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from dlvt.analysis import DISPLAY_THRESHOLD_FRACTION, find_interior_equilibria  # noqa: E402
from dlvt.model import dlvt_exogenous, make_params, simulate  # noqa: E402
from dlvt.scenarios import (  # noqa: E402
    FIGURE_SCENARIOS,
    compute_figure_scenario_results,
    simulate_two_phase,
)


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_figure_2(output_dir: Path) -> None:
    """Render the three declared endogenous/exogenous trajectory examples."""
    declared = FIGURE_SCENARIOS["fig2_three_scenarios"]
    scenarios = declared["scenarios"]
    threshold_fraction = float(declared["threshold_fraction"])

    above = scenarios["above_threshold_illustration"]
    p_above = make_params(**above["parameter_overrides"])
    t_above, v_above, c_above, *_ = simulate(
        p_above, T=above["t_end"], **above["initial_conditions"]
    )

    baseline = scenarios["illustrative_baseline"]
    p_baseline = make_params(**baseline["parameter_overrides"])
    t_base, v_base, c_base, *_ = simulate(
        p_baseline, T=baseline["t_end"], **baseline["initial_conditions"]
    )

    ramp = scenarios["exogenous_scope_ramp"]
    p_ramp = make_params(**ramp["parameter_overrides"])
    path = ramp["scope_path"]

    def scope_path(time: float) -> float:
        return float(path["intercept"] + path["slope"] * time)

    ramp_solution = solve_ivp(
        dlvt_exogenous,
        (0.0, ramp["t_end"]),
        [ramp["initial_conditions"]["V0"]],
        args=(p_ramp, scope_path),
        method="RK45",
        max_step=0.05,
        rtol=1e-8,
        atol=1e-10,
    )
    t_ramp = ramp_solution.t
    v_ramp = ramp_solution.y[0]
    c_ramp = np.asarray([scope_path(value) for value in t_ramp])

    trajectories: Iterable[Tuple[str, dict, np.ndarray, np.ndarray, np.ndarray]] = (
        ("(a) Above-threshold illustration", p_above, t_above, v_above, c_above),
        ("(b) Illustrative baseline", p_baseline, t_base, v_base, c_base),
        ("(c) Exogenous enacted-scope ramp", p_ramp, t_ramp, v_ramp, c_ramp),
    )
    colors = ("#238b45", "#d95f0e", "#cb181d")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=300)
    for axis, color, (title, params, time, vitality, scope) in zip(
        axes, colors, trajectories
    ):
        display_threshold = threshold_fraction * params["Vmax"]
        axis.plot(time, vitality, color=color, lw=2.5, label=r"Vitality $V(t)$")
        axis.plot(
            time,
            scope,
            color="black",
            ls="--",
            lw=1.4,
            alpha=0.55,
            label=r"Enacted scope $C(t)$",
        )
        axis.axhline(
            display_threshold, color="#6a51a3", ls=":", lw=1.3, alpha=0.75
        )
        axis.axhspan(0.0, display_threshold, color="#fb6a4a", alpha=0.05)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Time (arbitrary model units)")
        axis.set_ylabel("State value")
        axis.set_ylim(-0.3, 1.1 * max(float(np.max(vitality)), float(np.max(scope))))
        axis.grid(True, alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, output_dir, "fig2_three_scenarios")


def generate_figure_10(output_dir: Path) -> None:
    """Render the declared two-lever intervention comparison."""
    experiment = FIGURE_SCENARIOS["fig10_intervention_comparison"]
    baseline = make_params()
    initial = experiment["initial_conditions"]
    intervention_time = float(experiment["intervention_time"])
    display_threshold = float(experiment["threshold_fraction"]) * baseline["Vmax"]

    style_by_id = {
        "no_intervention": {"color": "black", "ls": "--", "lw": 2.0},
        "scope_coupling_reduction": {"color": "#2171b5", "lw": 2.0},
        "recovery_increase": {"color": "#238b45", "lw": 2.0},
        "combined": {"color": "#d94801", "lw": 2.5},
    }
    runs = []
    for scenario_id, scenario in experiment["scenarios"].items():
        after = make_params(**scenario["parameter_overrides"])
        time, vitality, scope = simulate_two_phase(
            baseline,
            after,
            y0=(initial["V0"], initial["C0"]),
            t_int=intervention_time,
            t_end=experiment["t_end"],
        )
        runs.append(
            (scenario["label"], time, vitality, scope, style_by_id[scenario_id])
        )

    fig, (vitality_axis, scope_axis) = plt.subplots(
        1, 2, figsize=(13, 5.5), dpi=300
    )
    for label, time, vitality, scope, style in runs:
        vitality_axis.plot(time, vitality, label=label, **style)
        scope_axis.plot(time, scope, label=label, **style)

    vitality_axis.axhline(display_threshold, color="gray", ls=":", lw=1.2)
    for axis in (vitality_axis, scope_axis):
        axis.axvline(intervention_time, color="gray", ls=":", lw=1.2)
        axis.set_xlabel("Time (arbitrary model units)")
        axis.grid(True, alpha=0.12)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    vitality_axis.set_ylabel(r"Vitality $V(t)$")
    vitality_axis.set_title("(a) Vitality trajectories")
    scope_axis.set_ylabel(r"Enacted leadership scope $C(t)$")
    scope_axis.set_title("(b) Scope trajectories")
    fig.tight_layout()
    _save(fig, output_dir, "fig10_intervention_comparison")


def write_scenario_manifest(path: Path) -> None:
    """Write the auditable inputs and outputs used by the public figures."""
    payload = {
        "schema_version": "1.0.0",
        "generated_by": "scripts/generate_reviewed_figures.py",
        "scientific_status": "illustrative_not_empirically_calibrated",
        "figure_scenarios": FIGURE_SCENARIOS,
        "figure_scenario_results": compute_figure_scenario_results(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fig",
        default="2,10",
        help="Comma-separated reviewed figure numbers (allowed: 2,10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_ROOT / "figures",
        help="Destination for PDF and PNG figure files.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=CODE_ROOT / "artifacts" / "figure_scenarios.json",
        help="Destination for the machine-readable scenario record.",
    )
    args = parser.parse_args()

    requested = [item.strip() for item in args.fig.split(",") if item.strip()]
    unknown = sorted(set(requested) - {"2", "10"})
    if unknown:
        parser.error(f"unreviewed figure number(s): {', '.join(unknown)}")
    if "2" in requested:
        generate_figure_2(args.output_dir)
    if "10" in requested:
        generate_figure_10(args.output_dir)
    write_scenario_manifest(args.manifest_output)
    print(f"reviewed figures written to {args.output_dir}")
    print(f"scenario manifest written to {args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
