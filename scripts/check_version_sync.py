"""Fail when public package and citation versions diverge."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def extract(path: Path, pattern: str, label: str) -> str:
    """Return the single captured version or raise a readable error."""
    matches = re.findall(pattern, path.read_text(encoding="utf-8"), re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(
            f"expected one {label} version in {path.name}, found {len(matches)}"
        )
    return matches[0]


def main() -> int:
    """Compare the public version declarations (four in the standalone
    export; five in the monorepo, where a root CITATION.cff mirror exists).
    """
    versions = {
        "pyproject.toml": extract(
            ROOT / "pyproject.toml", r'^version\s*=\s*"([^"]+)"', "project"
        ),
        "dlvt/__init__.py": extract(
            ROOT / "dlvt" / "__init__.py",
            r'^__version__\s*=\s*"([^"]+)"',
            "runtime",
        ),
        "CITATION.cff": extract(
            ROOT / "CITATION.cff", r"^version:\s*([^\s#]+)", "citation"
        ),
        # The README ships as the package long description on PyPI and carries
        # the version inside its worked citation example. Nothing else pinned
        # that literal, so it could drift silently behind the metadata.
        "README.md": extract(
            ROOT / "README.md", r"\(Version\s+([^\s)]+)\)", "readme citation"
        ),
    }
    # The monorepo keeps a root-level mirror of the citation metadata; the
    # standalone public export has no parent copy, so this check is optional.
    root_citation = ROOT.parent / "CITATION.cff"
    if root_citation.is_file():
        versions["../CITATION.cff"] = extract(
            root_citation, r"^version:\s*([^\s#]+)", "root citation"
        )
    if len(set(versions.values())) != 1:
        for source, version in versions.items():
            print(f"VERSION DRIFT: {source} declares {version}")
        return 1
    version = next(iter(versions.values()))
    print(f"public versions synchronized: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
