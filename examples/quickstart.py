"""Executable public-API smoke example for the illustrative DLVT baseline."""

from dlvt import (
    classify_equilibrium,
    coordination_load,
    find_interior_equilibria,
    make_params,
    simulate,
)


def main() -> None:
    """Run a trajectory, solve its equilibrium, and report one calibration."""
    params = make_params()
    time, vitality, scope, load, impact, gamma_ratio = simulate(
        params, V0=8.0, C0=0.5, T=120.0
    )
    equilibrium = find_interior_equilibria(params)[0]
    report = classify_equilibrium(params, threshold_fraction=0.5)

    assert len(time) == len(vitality) == len(scope) == len(load)
    assert len(time) == len(impact) == len(gamma_ratio)
    assert abs(equilibrium["V"] - 4.702529) < 1e-5
    assert abs(coordination_load(10.0, params) - 3.5) < 1e-12
    assert report["threshold_fraction"] == 0.5

    print(
        "illustrative equilibrium: "
        f"V*={equilibrium['V']:.6f}, "
        f"C*={equilibrium['C']:.6f}, "
        f"O*={equilibrium['O']:.6f}"
    )
    print(
        "display classification: "
        f"{report['status']} at theta={report['threshold_fraction']:.2f}"
    )


if __name__ == "__main__":
    main()
