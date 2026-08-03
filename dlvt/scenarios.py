"""Declared scenarios used by publication figures.

Every non-baseline parameter value and every imposed state path used in a
figure must be visible here.  The artifact generator serializes this mapping
to ``tables/artifact_manifest.json`` so captions and code can be audited
without reverse-engineering plotting functions.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .analysis import find_interior_equilibria
from .model import dlvt_exogenous, dlvt_system, make_params


FIGURE_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "fig2_three_scenarios": {
        "threshold_fraction": 0.5,
        "scenarios": {
            "above_threshold_illustration": {
                "dynamics": "endogenous",
                "parameter_overrides": {"delta": 0.008, "beta": 0.15},
                "initial_conditions": {"V0": 8.0, "C0": 0.5},
                "t_end": 200.0,
            },
            "illustrative_baseline": {
                "dynamics": "endogenous",
                "parameter_overrides": {},
                "initial_conditions": {"V0": 8.0, "C0": 0.5},
                "t_end": 200.0,
            },
            "exogenous_scope_ramp": {
                "dynamics": "exogenous_enacted_scope",
                "parameter_overrides": {},
                "initial_conditions": {"V0": 8.0, "C0": 0.5},
                "t_end": 100.0,
                "scope_path": {"kind": "linear", "intercept": 0.5, "slope": 0.5},
            },
        },
    },
    "fig10_intervention_comparison": {
        "threshold_fraction": 0.5,
        "initial_conditions": {"V0": 3.0, "C0": 15.0},
        "intervention_time": 20.0,
        "t_end": 120.0,
        "scenarios": {
            "no_intervention": {
                "label": "No intervention",
                "parameter_overrides": {},
            },
            "scope_coupling_reduction": {
                "label": "Reduce beta by 60%",
                "parameter_overrides": {"beta": 0.10},
            },
            "recovery_increase": {
                "label": "Increase R by 40%",
                "parameter_overrides": {"R": 4.2},
            },
            "combined": {
                "label": "Both simultaneously",
                "parameter_overrides": {"R": 4.2, "beta": 0.10},
            },
        },
    },
}


def simulate_two_phase(
    p_before: Mapping[str, float],
    p_after: Mapping[str, float],
    y0: Tuple[float, float] = (3.0, 15.0),
    t_int: float = 20.0,
    t_end: float = 120.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate a declared two-phase intervention experiment."""
    integration = dict(
        method="RK45", dense_output=True, max_step=0.1, rtol=1e-8, atol=1e-10
    )
    sol1 = solve_ivp(
        dlvt_system, (0.0, t_int), list(y0), args=(dict(p_before),), **integration
    )
    sol2 = solve_ivp(
        dlvt_system,
        (t_int, t_end),
        sol1.y[:, -1],
        args=(dict(p_after),),
        **integration,
    )
    return (
        np.concatenate([sol1.t, sol2.t]),
        np.concatenate([sol1.y[0], sol2.y[0]]),
        np.concatenate([sol1.y[1], sol2.y[1]]),
    )


def compute_figure_scenario_results() -> Dict[str, Dict[str, Any]]:
    """Compute deterministic outputs for every declared publication scenario."""
    results: Dict[str, Dict[str, Any]] = {}

    figure2 = FIGURE_SCENARIOS["fig2_three_scenarios"]
    figure2_results: Dict[str, Any] = {}
    for scenario_id in ("above_threshold_illustration", "illustrative_baseline"):
        scenario = figure2["scenarios"][scenario_id]
        params = make_params(**scenario["parameter_overrides"])
        equilibrium = find_interior_equilibria(params)[0]
        threshold = figure2["threshold_fraction"] * params["Vmax"]
        figure2_results[scenario_id] = {
            "V_star": round(float(equilibrium["V"]), 6),
            "C_star": round(float(equilibrium["C"]), 6),
            "O_star": round(float(equilibrium["O"]), 6),
            "below_display_threshold": bool(equilibrium["V"] < threshold),
        }

    ramp = figure2["scenarios"]["exogenous_scope_ramp"]
    params = make_params(**ramp["parameter_overrides"])
    path = ramp["scope_path"]
    scope_path = lambda t: path["intercept"] + path["slope"] * t
    solution = solve_ivp(
        dlvt_exogenous,
        (0.0, ramp["t_end"]),
        [ramp["initial_conditions"]["V0"]],
        args=(params, scope_path),
        method="RK45",
        max_step=0.05,
    )
    threshold = figure2["threshold_fraction"] * params["Vmax"]
    figure2_results["exogenous_scope_ramp"] = {
        "V_final": round(float(solution.y[0, -1]), 6),
        "C_final": round(float(scope_path(ramp["t_end"])), 6),
        "entered_below_display_threshold": bool(np.any(solution.y[0] < threshold)),
        "remained_strictly_positive": bool(np.all(solution.y[0] > 0.0)),
    }
    results["fig2_three_scenarios"] = figure2_results

    figure10 = FIGURE_SCENARIOS["fig10_intervention_comparison"]
    initial = figure10["initial_conditions"]
    baseline = make_params()
    threshold = figure10["threshold_fraction"] * baseline["Vmax"]
    figure10_results: Dict[str, Any] = {}
    for scenario_id, scenario in figure10["scenarios"].items():
        after = make_params(**scenario["parameter_overrides"])
        _, vitality, scope = simulate_two_phase(
            baseline,
            after,
            y0=(initial["V0"], initial["C0"]),
            t_int=figure10["intervention_time"],
            t_end=figure10["t_end"],
        )
        figure10_results[scenario_id] = {
            "V_final": round(float(vitality[-1]), 6),
            "C_final": round(float(scope[-1]), 6),
            "above_display_threshold": bool(vitality[-1] > threshold),
        }
    results["fig10_intervention_comparison"] = figure10_results
    return results
