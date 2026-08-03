"""
robustness_grid.py
==================
Programmatic construction of the DLVT structural robustness table (R7-7).

For each perturbation of the baseline model, compute three answers
(yes / no / conditional, plus an explicit no-equilibrium outcome for the
absorption column):

  1.  Does an interior equilibrium (V*, C*) with V* > 0 and C* > 0 exist?
  2.  Is V* below the illustrative 0.5 * V_max display threshold?
  3.  Does scope absorption break --- does V*(beta) respond non-trivially
      to beta?  Any drain kernel depending on scope only through O and V
      preserves absorption where the interior equilibrium exists, including
      Hill and exponential-in-O kernels.  Explicit nonabsorbed C dependence
      can break it.  A bounded (saturating) load map O(C) is a different
      failure mode: it violates the unbounded-load hypothesis behind the
      trapping bound, and for sufficiently small beta the load supremum
      falls below the level required by the scope nullcline, so no interior
      equilibrium exists and scope grows without bound.  That case is
      reported explicitly as "no equilibrium", not as a borderline
      "conditional".

The script integrates a small grid of perturbed ODEs to long time and reads
the converged state as the equilibrium. Alternative drain kernels are
supplied as RHS overrides rather than through dlvt.model (which bakes in the
power-law kernel).

Usage
-----
    python3 code/scripts/robustness_grid.py
    python3 code/scripts/robustness_grid.py --csv out.csv --tex out.tex

Outputs CSV and LaTeX booktabs table suitable for \\input into the paper.

Runtime: < 30 s on a laptop.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp

# Make the dlvt package importable when running as a script.
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from dlvt.model import DEFAULT_PARAMS, make_params  # noqa: E402


DISPLAY_THRESHOLD_FRACTION = 0.5
T_FINAL = 1200.0
RTOL = 1e-8
ATOL = 1e-10
# Terminal scope beyond this bound is classified as divergent (unbounded
# scope growth; no interior equilibrium was reached).
DIVERGENCE_BOUND = 1.0e6

# Cell values (plain ASCII; render_latex maps them to TeX where needed).
YES = "yes"
NO = "no"
COND = "conditional"
# Reported when a beta probe destroys the interior equilibrium entirely (for
# a bounded load map at low beta the load supremum falls below the level the
# scope nullcline requires, and scope grows without bound), so the question
# "does V*(beta) move" is not well posed. The wording names the probe that
# actually diverged.
EQ_LOST_LOW_BETA = "equilibrium lost at low beta"
EQ_LOST_HIGH_BETA = "equilibrium lost at high beta"
EQ_LOST_BOTH = "equilibrium lost (unbounded scope)"

# Plain-cell -> LaTeX-cell mapping used only by render_latex.
LATEX_CELLS = {
    EQ_LOST_LOW_BETA: r"equilibrium lost at low $\beta$",
    EQ_LOST_HIGH_BETA: r"equilibrium lost at high $\beta$",
}


# ── Drain kernel factories ────────────────────────────────────────────────────

def power_law_rhs(p: Dict[str, float]) -> Callable[[float, List[float]], List[float]]:
    """Baseline DLVT drain kernel: delta * O^gamma * V/(V + eps)."""
    def rhs(t, y):
        V = max(y[0], 0.0)
        C = max(y[1], 0.0)
        O = p["O0"] + p["beta"] * C ** p["eta"]
        recovery = p["R"] * (1.0 - V / p["Vmax"])
        drain = p["delta"] * O ** p["gamma"] * V / (V + p["eps"])
        I = C * V / (1.0 + p["phi"] * O)
        return [recovery - drain, p["alpha"] * I - p["mu"] * C]
    return rhs


def exponential_in_load_rhs(
    p: Dict[str, float], kappa: float = 0.08
) -> Callable:
    """Exponential drain in experienced load; scope absorption is preserved.

        drain = delta * exp(kappa * O) * V / (V + eps).

    The functional form is non-power-law, but it still depends on scope only
    through O.  The equilibrium pair (V*,O*) is therefore beta-invariant.
    """
    def rhs(t, y):
        V = max(y[0], 0.0)
        C = max(y[1], 0.0)
        O = p["O0"] + p["beta"] * C ** p["eta"]
        recovery = p["R"] * (1.0 - V / p["Vmax"])
        drain = p["delta"] * np.exp(kappa * O) * V / (V + p["eps"])
        I = C * V / (1.0 + p["phi"] * O)
        return [recovery - drain, p["alpha"] * I - p["mu"] * C]
    return rhs


def exponential_in_scope_rhs(
    p: Dict[str, float], kappa: float = 0.08
) -> Callable:
    """Exponential drain kernel with explicit C dependence:
        drain = delta * exp(kappa * C) * V / (V + eps).

    Note that scope absorption holds for any drain of the form
    f(O) * g(V) because the C-isocline forces O to be a function of V alone,
    reducing the V-isocline to a pure V equation. An exponential in
    ``beta*C**eta`` does not break this because it is the excess-load
    composite. The genuine counterexample is a drain that depends on
    C directly, not only through the composite O or beta * C --- hence
    exp(kappa * C), which introduces an un-absorbed C term into the
    V-isocline and lets V*(beta) respond to beta.
    """
    def rhs(t, y):
        V = max(y[0], 0.0)
        C = max(y[1], 0.0)
        O = p["O0"] + p["beta"] * C ** p["eta"]
        recovery = p["R"] * (1.0 - V / p["Vmax"])
        drain = p["delta"] * np.exp(kappa * C) * V / (V + p["eps"])
        I = C * V / (1.0 + p["phi"] * O)
        return [recovery - drain, p["alpha"] * I - p["mu"] * C]
    return rhs


def hill_rhs(p: Dict[str, float], K: float = 4.0) -> Callable:
    """Hill (saturating) drain kernel: delta * O^gamma / (K^gamma + O^gamma) *
    V/(V + eps). The drain saturates at delta as experienced load grows, so a
    high-load leader is not exposed to unbounded drain. Like the power-law
    kernel this is of the form f(O)*g(V), so scope absorption still applies:
    a threshold classification may change, but beta-invariance does not.
    """
    def rhs(t, y):
        V = max(y[0], 0.0)
        C = max(y[1], 0.0)
        O = p["O0"] + p["beta"] * C ** p["eta"]
        recovery = p["R"] * (1.0 - V / p["Vmax"])
        hill = O ** p["gamma"] / (K ** p["gamma"] + O ** p["gamma"])
        drain = p["delta"] * hill * V / (V + p["eps"])
        I = C * V / (1.0 + p["phi"] * O)
        return [recovery - drain, p["alpha"] * I - p["mu"] * C]
    return rhs


def explicit_C_drain_rhs(p: Dict[str, float], theta: float = 0.001) -> Callable:
    """Drain with an explicit C-dependent multiplier not mediated by O.

    drain = delta * O^gamma * (1 + theta * C^2) * V/(V + eps).

    Scope absorption holds for any drain of the form f(O)*g(V)
    because the C-isocline forces O to be a function of V alone, collapsing
    the V-isocline to a pure V equation. The only way to break scope
    absorption while keeping the C-isocline intact is to add an explicit C
    term to the drain that is NOT absorbed into O. This kernel does exactly
    that: the (1 + theta*C^2) factor introduces a beta-dependent term into
    the V-isocline through C = (A(V)/beta)^(1/eta), so V*(beta) now responds
    to beta.
    """
    def rhs(t, y):
        V = max(y[0], 0.0)
        C = max(y[1], 0.0)
        O = p["O0"] + p["beta"] * C ** p["eta"]
        recovery = p["R"] * (1.0 - V / p["Vmax"])
        drain = p["delta"] * O ** p["gamma"] * (1.0 + theta * C * C) * V / (V + p["eps"])
        I = C * V / (1.0 + p["phi"] * O)
        return [recovery - drain, p["alpha"] * I - p["mu"] * C]
    return rhs


def saturating_load_rhs(p: Dict[str, float], K: float = 50.0) -> Callable:
    """Baseline kernel with saturating load ``O = O0 + beta*C/(1 + C/K)``.

    Experienced coordination load plateaus at ``O0 + beta*K``, so drain
    cannot grow without bound --- and neither can the scope contraction
    pressure.  This violates the unbounded-load hypothesis behind the
    trapping bound ``C_trap``: when ``O0 + beta*K`` falls below the load the
    scope nullcline requires, no interior equilibrium exists and scope grows
    without bound (observed here for ``beta = 0.15`` with ``K = 50``).
    """
    def rhs(t, y):
        V = max(y[0], 0.0)
        C = max(y[1], 0.0)
        O = p["O0"] + p["beta"] * C / (1.0 + C / K)
        recovery = p["R"] * (1.0 - V / p["Vmax"])
        drain = p["delta"] * O ** p["gamma"] * V / (V + p["eps"])
        I = C * V / (1.0 + p["phi"] * O)
        return [recovery - drain, p["alpha"] * I - p["mu"] * C]
    return rhs


def saturating_complexity_rhs(p: Dict[str, float], K: float = 50.0) -> Callable:
    """Deprecated compatibility alias for :func:`saturating_load_rhs`."""
    return saturating_load_rhs(p, K=K)


# ── Equilibrium computation ───────────────────────────────────────────────────

def integrate_to_equilibrium(rhs: Callable,
                             V0: float = 8.0,
                             C0: float = 5.0,
                             t_final: float = T_FINAL) -> Tuple[float, float, bool]:
    """Integrate the rhs to t_final. Return (V*, C*, converged?)."""
    sol = solve_ivp(rhs, (0.0, t_final), [V0, C0],
                    method="RK45", rtol=RTOL, atol=ATOL,
                    t_eval=np.linspace(t_final * 0.9, t_final, 50))
    if not sol.success:
        raise RuntimeError(f"equilibrium integration failed: {sol.message}")
    V_tail = sol.y[0]
    C_tail = sol.y[1]
    V_star = float(V_tail[-1])
    C_star = float(C_tail[-1])
    # "Converged" = tail variation small relative to level.
    tail_spread_V = float(V_tail.max() - V_tail.min())
    tail_spread_C = float(C_tail.max() - C_tail.min())
    converged = (tail_spread_V < max(1e-3, 1e-3 * abs(V_star)) and
                 tail_spread_C < max(1e-3, 1e-3 * abs(C_star)))
    return V_star, C_star, converged


def equilibrium_diagnostics(
    rhs: Callable,
    V_star: float,
    C_star: float,
    *,
    residual_tol: float = 1e-5,
    stability_tol: float = 1e-8,
) -> Dict[str, object]:
    """Check stationarity and local stability for any two-state RHS.

    Alternative robustness kernels do not share the baseline analytical
    Jacobian, so this function uses a centered finite-difference Jacobian at
    the candidate terminal state.  A positive terminal coordinate alone is
    not evidence of an equilibrium.
    """
    point = np.asarray([V_star, C_star], dtype=float)
    residual = np.asarray(rhs(0.0, point), dtype=float)
    residual_norm = float(np.linalg.norm(residual, ord=np.inf))

    jacobian = np.empty((2, 2), dtype=float)
    for column in range(2):
        step = 1e-5 * max(1.0, abs(point[column]))
        upper = point.copy()
        lower = point.copy()
        upper[column] += step
        lower[column] -= step
        jacobian[:, column] = (
            np.asarray(rhs(0.0, upper), dtype=float)
            - np.asarray(rhs(0.0, lower), dtype=float)
        ) / (2.0 * step)
    eigenvalues = np.linalg.eigvals(jacobian)
    stable = bool(np.all(np.real(eigenvalues) < -stability_tol))
    return {
        "residual_norm": residual_norm,
        "residual_ok": residual_norm <= residual_tol,
        "eigenvalues": eigenvalues,
        "stable": stable,
    }


def has_interior_equilibrium(V_star: float, C_star: float,
                             V_max: float, tol: float = 1e-3) -> bool:
    """Interior means V* strictly inside (0, V_max) and C* > 0 (strictly).

    A terminal state pinned at the ``V = V_max`` boundary is not an interior
    equilibrium, so the ceiling is checked explicitly.
    """
    return tol < V_star < V_max - tol and C_star > tol


def is_low_vitality(V_star: float, V_max: float) -> bool:
    """Illustrative theta=0.5 classification used only for sensitivity."""
    return V_star < DISPLAY_THRESHOLD_FRACTION * V_max


def scope_absorption_breaks(
    rhs_factory: Callable[[Dict[str, float]], Callable],
    p_base: Dict[str, float],
    beta_low: float = 0.15,
    beta_high: float = 0.40,
) -> str:
    """Check whether ``V*(beta)`` responds materially to beta.

    Returns an "equilibrium lost" outcome if a probe exhibits unbounded
        scope growth (the interior equilibrium ceases to exist, so
        beta-invariance of ``V*`` is not even well posed); the wording
        names the diverging probe,
    YES if |dV*/dbeta| is "large" (relative shift > 1%),
    COND if the shift is 0.1% - 1% (borderline),
    NO  if the shift is < 0.1% (scope-absorbed).
    """
    p_low = dict(p_base); p_low["beta"] = beta_low
    p_high = dict(p_base); p_high["beta"] = beta_high
    V_low, C_low, ok_low = integrate_to_equilibrium(rhs_factory(p_low))
    V_high, C_high, ok_high = integrate_to_equilibrium(rhs_factory(p_high))
    low_diverged = C_low > DIVERGENCE_BOUND
    high_diverged = C_high > DIVERGENCE_BOUND
    if low_diverged and high_diverged:
        return EQ_LOST_BOTH
    if low_diverged:
        return EQ_LOST_LOW_BETA
    if high_diverged:
        return EQ_LOST_HIGH_BETA
    if not (ok_low and ok_high):
        return COND
    if not (has_interior_equilibrium(V_low, C_low, p_low["Vmax"]) and
            has_interior_equilibrium(V_high, C_high, p_high["Vmax"])):
        return COND
    rel_shift = abs(V_high - V_low) / max(1e-6, 0.5 * (V_low + V_high))
    if rel_shift > 1e-2:
        return YES
    if rel_shift > 1e-3:
        return COND
    return NO


# ── Perturbation grid ─────────────────────────────────────────────────────────

def build_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    V_max = DEFAULT_PARAMS["Vmax"]

    def evaluate(label: str, rhs_factory: Callable, p: Dict[str, float]) -> Dict[str, str]:
        rhs = rhs_factory(p)
        V_star, C_star, conv = integrate_to_equilibrium(rhs)
        if C_star > DIVERGENCE_BOUND:
            # Unbounded scope growth at the evaluation parameters: report the
            # absence of an interior equilibrium instead of a numeric state.
            return {
                "perturbation": label,
                "V_star": f"{V_star:.3f}",
                "C_star": "unbounded",
                "exists": NO,
                "low_vitality_theta_0_5": NO,
                "scope_absorption_breaks": scope_absorption_breaks(rhs_factory, p),
            }
        diagnostics = equilibrium_diagnostics(rhs, V_star, C_star)
        if not conv:
            raise RuntimeError(f"{label}: terminal tail did not converge")
        if not diagnostics["residual_ok"]:
            raise RuntimeError(
                f"{label}: terminal residual {diagnostics['residual_norm']:.3e} "
                "exceeds tolerance"
            )
        if not diagnostics["stable"]:
            raise RuntimeError(f"{label}: terminal equilibrium is not locally stable")
        interior = has_interior_equilibrium(V_star, C_star, p["Vmax"])
        if not interior:
            exists = NO
            low_vitality = NO
        else:
            exists = YES
            low_vitality = YES if is_low_vitality(V_star, V_max) else NO
        absorption_breaks = scope_absorption_breaks(rhs_factory, p)
        return {
            "perturbation": label,
            "V_star": f"{V_star:.3f}",
            "C_star": f"{C_star:.3f}",
            "exists": exists,
            "low_vitality_theta_0_5": low_vitality,
            "scope_absorption_breaks": absorption_breaks,
        }

    # Baseline (anchor row).
    rows.append(evaluate(
        r"Baseline (power-law, $\gamma=2$)",
        power_law_rhs,
        make_params(),
    ))

    # Gamma perturbations.
    for g in (1.0, 3.0):
        rows.append(evaluate(
            rf"$\gamma = {int(g)}$ (power-law drain)",
            power_law_rhs,
            make_params(gamma=g),
        ))

    # Drain-kernel perturbations.
    rows.append(evaluate(
        "Exponential drain in load $O$",
        lambda p: exponential_in_load_rhs(p, kappa=0.08),
        make_params(),
    ))
    rows.append(evaluate(
        "Exponential drain with explicit $C$",
        lambda p: exponential_in_scope_rhs(p, kappa=0.35),
        make_params(),
    ))
    rows.append(evaluate(
        "Hill (saturating) drain kernel",
        lambda p: hill_rhs(p, K=4.0),
        make_params(),
    ))
    rows.append(evaluate(
        "Saturating experienced load $O(C)$",
        lambda p: saturating_load_rhs(p, K=50.0),
        make_params(),
    ))
    rows.append(evaluate(
        "Drain with explicit $C$ term",
        lambda p: explicit_C_drain_rhs(p, theta=0.001),
        make_params(),
    ))

    # phi perturbations (impact sensitivity to experienced load).
    for phi in (0.075, 0.30):
        rows.append(evaluate(
            rf"$\phi = {phi}$",
            power_law_rhs,
            make_params(phi=phi),
        ))

    # mu perturbations (enacted-scope contraction rate).
    for mu in (0.10, 0.40):
        rows.append(evaluate(
            rf"$\mu = {mu}$",
            power_law_rhs,
            make_params(mu=mu),
        ))

    # eps perturbations (regularization).
    for eps in (0.01, 1.0):
        rows.append(evaluate(
            rf"$\varepsilon = {eps}$",
            power_law_rhs,
            make_params(eps=eps),
        ))

    return rows


# ── Output writers ────────────────────────────────────────────────────────────

def render_csv(rows: List[Dict[str, str]]) -> str:
    """Render the deterministic CSV representation."""
    fieldnames = [
        "perturbation",
        "V_star",
        "C_star",
        "exists",
        "low_vitality_theta_0_5",
        "scope_absorption_breaks",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return stream.getvalue()


def write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.write_text(render_csv(rows), encoding="utf-8")


def render_latex(rows: List[Dict[str, str]]) -> str:
    """Render the deterministic LaTeX representation."""
    header = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Structural robustness of DLVT predictions. Rows perturb one "
        r"assumption of the illustrative baseline model at a time; "
        r"columns ask whether each core feature of the theory survives. "
        r"``Interior $(V^*, C^*)$ exists'' asks whether an interior stable "
        r"equilibrium is reached from $(V_0, C_0)=(8, 5)$ under long-time integration. "
        r"``Low vitality at $\theta=.5$'' reports an illustrative threshold "
        r"classification, not a structural state. ``Scope absorption breaks'' asks "
        r"whether $V^*(\beta)$ responds non-trivially to $\beta$; drain kernels "
        r"depending on scope only through $O$ and $V$ report ``no'' wherever the "
        r"interior equilibrium exists. The saturating-load row reports "
        r"``equilibrium lost at low $\beta$'': its interior equilibrium exists "
        r"at the evaluation $\beta$ (first column), but a bounded load map "
        r"violates the unbounded-load hypothesis behind the existence and "
        r"trapping constructions, so for sufficiently small $\beta$ no interior "
        r"equilibrium exists and scope grows without bound. "
        r"All rows are generated by \texttt{robustness\_grid.py}.}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{"
        r">{\raggedright\arraybackslash}p{0.37\textwidth}"
        r">{\centering\arraybackslash}p{0.16\textwidth}"
        r">{\centering\arraybackslash}p{0.16\textwidth}"
        r">{\centering\arraybackslash}p{0.16\textwidth}}",
        r"\toprule",
        r"Perturbation & Interior $(V^*, C^*)$ exists & "
        r"Low vitality at $\theta=.5$ & Scope absorption breaks \\",
        r"\midrule",
    ]
    body = []
    for row in rows:
        absorption_cell = LATEX_CELLS.get(
            row["scope_absorption_breaks"], row["scope_absorption_breaks"]
        )
        body.append(
            f"{row['perturbation']} & {row['exists']} & "
            f"{row['low_vitality_theta_0_5']} & "
            f"{absorption_cell} \\\\"
        )
    footer = [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(header + body + footer) + "\n"


def write_latex(rows: List[Dict[str, str]], path: Path) -> None:
    path.write_text(render_latex(rows), encoding="utf-8")


def print_ascii(rows: List[Dict[str, str]]) -> None:
    col_widths = [40, 10, 10, 10, 15]
    headers = ["Perturbation", "V*", "C*", "low V", "absorption breaks"]
    print("  ".join(h.ljust(w) for h, w in zip(headers, col_widths)))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        # Strip LaTeX for readable console output.
        label = (row["perturbation"]
                 .replace("$", "").replace("\\gamma", "gamma")
                 .replace("\\phi", "phi").replace("\\mu", "mu")
                 .replace("\\varepsilon", "eps").replace("O(C)", "O(C)"))
        cells = [label[: col_widths[0]],
                 row["V_star"],
                 row["C_star"],
                 row["low_vitality_theta_0_5"] if row["exists"] == YES else "n/a",
                 row["scope_absorption_breaks"]]
        print("  ".join(str(c).ljust(w) for c, w in zip(cells, col_widths)))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("robustness_grid.csv"))
    ap.add_argument("--tex", type=Path, default=Path("robustness_grid.tex"))
    args = ap.parse_args()

    rows = build_rows()
    print_ascii(rows)
    write_csv(rows, args.csv)
    write_latex(rows, args.tex)
    print(f"\nWrote {args.csv} and {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
