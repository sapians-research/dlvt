# Changelog

## 2.2.0rc2 (2026-08-02)

Security and platform maintenance release; no model, API, or contract
changes.

- Drop Python 3.9 (end-of-life October 2025); supported and CI-tested range
  is now 3.10–3.13.
- Upgrade the locked dependency set so every Dependabot security alert is
  resolved: Pillow 12.3.0 (transitive via matplotlib), pytest 9.1.1,
  black 26.5.1.
- Version metadata synchronized across pyproject, runtime, and citation
  files by `check_version_sync.py`.

This project follows semantic versioning for the public Python API. Scientific
claims and illustrative parameters are versioned alongside the executable
contract; a software version is not an empirical validation milestone.

## 2.2.0rc1 — 2026-07-11

Release candidate for the 2.2.0 compatibility release. The candidate is not a
final archival release: repository publication, hosted CI, tag, and DOI remain
release gates.

### Added

- Canonical construct/API names: `coordination_load`,
  `drain_coefficient_threshold`, `is_low_vitality`,
  `classify_equilibrium`, `trapping_scope_bound`, and
  `quasi_static_nullcline`.
- Exact monotone equilibrium solver and explicit separation of mathematical
  equilibrium from threshold-sensitive classification.
- General-`eta` nondimensionalization and executable dimensionless groups.
- Scope-absorption and absorption-breaker robustness tests.
- Projected stochastic experiments and deterministic least-squares recovery
  diagnostics, explicitly marked experimental.
- Public-export allowlist, executable quick start, citation metadata, and CI
  gates for tests, clean-tree export, reviewed-figure reproduction, wheel and
  sdist inspection, installations, and examples.

### Changed

- `C` now means enacted leadership scope, not career capital or accumulated
  capability.
- `O` now means experienced coordination load, not organizational complexity
  in general.
- Low-vitality categories require an explicit display threshold and do not
  denote a dynamical regime, diagnosis, or performance state.
- Python support is declared for 3.9–3.12 (superseded in 2.2.0rc2: 3.10–3.13) and is exercised by the public CI
  matrix.

### Deprecated

- `complexity` in favor of `coordination_load`.
- `carrying_capacity` in favor of `drain_coefficient_threshold`.
- `is_zombie` in favor of `is_low_vitality`.
- `classify_regime` in favor of `classify_equilibrium`.
- `slow_manifold` in favor of `quasi_static_nullcline`.
- `trapping_capital_bound` in favor of `trapping_scope_bound`.
- `classify_point`, `zombie_boundary_map`, `zombie_boundary_map_beta`, and
  `lhs_zombie_fraction` in favor of continuous equilibrium outputs plus an
  explicit display threshold.
- Sobol `output='regime'` in favor of `output='low_vitality'`.

These compatibility aliases remain for the 2.2 line and are scheduled for
removal in 3.0.0. This deprecation release followed by a major removal is the
chosen migration strategy.

## 2.1.0

Historical development release. Its scientific terminology and several API
names are superseded by the 2.2.0rc1 contract above. Do not infer current
claim semantics from old examples or cached package metadata.
