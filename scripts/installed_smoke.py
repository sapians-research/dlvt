"""Small smoke test intended to run outside the source tree after install."""
from __future__ import annotations

import math
from importlib.metadata import version

import dlvt


def main() -> int:
    params = dlvt.make_params()
    equilibrium = dlvt.find_interior_equilibria(params)
    if len(equilibrium) != 1:
        raise RuntimeError("installed package did not return one baseline equilibrium")
    point = equilibrium[0]
    if not point["stable"] or not all(
        math.isfinite(float(point[key])) for key in ("V", "C", "O", "I")
    ):
        raise RuntimeError("installed package returned an invalid equilibrium")
    installed_version = version("dlvt")
    if dlvt.__version__ != installed_version:
        raise RuntimeError(
            f"runtime version {dlvt.__version__} != metadata version {installed_version}"
        )
    print(
        f"installed dlvt {dlvt.__version__}: "
        f"V*={point['V']:.6f}, C*={point['C']:.6f}, stable={point['stable']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
