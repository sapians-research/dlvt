# Contributing to DLVT research code

Contributions that improve correctness, reproducibility, documentation, or
tests are welcome. This package implements a formal research model; it is not
a personnel, diagnostic, clinical, or forecasting tool.

## Development setup

```bash
git clone https://github.com/wbendinelli/dlvt.git
cd dlvt
python -m pip install -e ".[dev]"
python -m pytest tests/ -q
python examples/quickstart.py
python scripts/check_public_export.py
python scripts/check_version_sync.py
```

The supported Python versions are those in `.github/workflows/ci.yml` and the
package metadata. Do not broaden the range without adding the version to CI.

## Scientific contract

The state variables and derived quantities have narrow meanings:

- `V`: subjective vitality available for leadership activity;
- `C`: enacted leadership scope;
- `O`: experienced coordination load;
- `I`: modeled impact flow, not observed performance.

The default parameter vector is illustrative. Never change it silently. A
parameter or equation change requires:

1. a clear scientific rationale in the pull request;
2. tests for the changed result and relevant boundary/counterexample;
3. updated documentation and examples;
4. regenerated numerical artifacts in the source manuscript repository;
5. a changelog entry if public behavior changes.

Do not introduce a categorical threshold without requiring it explicitly in
the API and documenting that it is a calibration. Do not describe
`Gamma = delta*O**gamma/R` as the sign of `dV/dt`, and do not conflate the
drain-coefficient threshold with the trapping bound.

## Code and test expectations

- Follow PEP 8 and use NumPy-style docstrings for public functions.
- Prefer small, explicit functions and deterministic tests.
- Add property, boundary, or counterexample tests for mathematical claims.
- Seed stochastic experiments and state whether projection changes their
  interpretation.
- Mark numerical diagnostics as experimental unless a formal contract and
  validation gate exist.
- Avoid hard-coded test counts; test discovery is authoritative.

Run before opening a pull request:

```bash
python -m pytest tests/ -q
python examples/quickstart.py
python scripts/check_public_export.py
python scripts/check_version_sync.py
python -m build
```

## API compatibility

Deprecated names remain only for the 2.2 compatibility line. New code must use
the canonical names documented in `README.md`. Removing an alias requires the
planned 3.0 major release and a changelog entry.

## Pull requests and issues

Keep pull requests focused. Explain the observed behavior, expected behavior,
scientific or software rationale, and validation commands. Bug reports should
include a minimal example, Python version, operating system, package version
or commit, and the full traceback.

All contributions are licensed under the MIT License.
