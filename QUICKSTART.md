# DLVT quick start

## 1. Install

```bash
python -m pip install dlvt
```

To work from a checkout instead, run this from the directory that holds
`pyproject.toml` — the repository root in the standalone `dlvt` repository,
`code/` in the research monorepo:

```bash
python -m pip install -e .
```

## 2. Simulate the illustrative baseline

```python
from dlvt import make_params, simulate

p = make_params()
t, V, C, O, I, gamma_ratio = simulate(
    p,
    V0=8.0,
    C0=0.5,
    T=120.0,
)
```

`V` is subjective vitality, `C` enacted leadership scope, `O` experienced
coordination load, and `I` modeled impact flow. `gamma_ratio` is the coefficient
ratio `delta*O**gamma/R`; it is not the sign of `dV/dt`.

## 3. Solve the equilibrium exactly

```python
from dlvt import find_interior_equilibria

equilibria = find_interior_equilibria(p)
if equilibria:
    eq = equilibria[0]
    print(f"V*={eq['V']:.6f}")
    print(f"C*={eq['C']:.6f}")
    print(f"O*={eq['O']:.6f}")
    print(f"stable={eq['stable']}")
else:
    print("No interior equilibrium under the exact existence condition")
```

An empty result does not mean that vitality reaches zero. It means only
that the specified system has no positive interior fixed point; boundary
dynamics must be analyzed separately.

## 4. Apply an explicit display threshold

```python
from dlvt import classify_equilibrium

half = classify_equilibrium(p, threshold_fraction=0.50)
lower = classify_equilibrium(p, threshold_fraction=0.46)

print(half["status"], half["V_star"])
print(lower["status"], lower["V_star"])
```

The two calls return the same equilibrium and can return different labels.
That is intentional: the threshold is calibrational, not a bifurcation or
clinical boundary.

## 5. Inspect load and the two distinct scope levels

```python
from dlvt import (
    coordination_load,
    drain_coefficient_threshold,
    trapping_scope_bound,
)

print(coordination_load(10.0, p))
print(drain_coefficient_threshold(p))
print(trapping_scope_bound(p))
```

`drain_coefficient_threshold()` solves `Gamma == 1`. It is not a carrying
capacity. `trapping_scope_bound()` returns the separate scope-coordinate
bound used in the global-state argument.

## 6. Declare scenario overrides visibly

```python
from dlvt import make_params

# Above-threshold illustration used in Figure 2.
p_scenario = make_params(delta=0.008, beta=0.15)
```

The changed parameters must be disclosed together. Under scope absorption,
changing `beta` alone rescales equilibrium `C*` but does not move `V*` or `O*`.

## 7. Test

From the standalone repository root:

```bash
python -m pytest tests/ -q
```

The defaults are illustrative. The package is research code and must not be
used to diagnose, rank, or make decisions about individual leaders.
