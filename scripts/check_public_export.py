"""Materialize and verify the standalone public DLVT source tree.

The allowlist is executable policy, not merely documentation.  The exporter
copies exactly the declared regular files into an empty destination and writes
``EXPORT_SHA256SUMS.txt``.  The tree checker rejects missing files, extra
files, symlinks, path escapes, and hash drift.

Examples::

    python scripts/check_public_export.py
    python scripts/check_public_export.py --export-dir /tmp/dlvt-public
    python /tmp/dlvt-public/scripts/check_public_export.py \
        --check-tree /tmp/dlvt-public
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Set


CODE_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = CODE_ROOT / "PUBLIC_EXPORT_ALLOWLIST.txt"
HASH_MANIFEST = "EXPORT_SHA256SUMS.txt"

BANNED_PARTS = {
    "dlvt_figures.py",
    "dlvt_solver.py",
    "fig8_bifurcation_diagnostic.py",
    "fig9_robustness.py",
    "fig10_intervention_comparison.py",
    "figures.py",
    "run_all_figures.py",
    "generate_contract_artifacts.py",
    "validate_sota.py",
    "verify.py",
    "test_artifacts.py",
    "notebooks",
    ".venv",
    ".pytest_cache",
    "dlvt.egg-info",
}


def load_allowlist(root: Path = CODE_ROOT) -> List[Path]:
    entries: List[Path] = []
    allowlist = root / "PUBLIC_EXPORT_ALLOWLIST.txt"
    for raw in allowlist.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        entries.append(Path(value))
    return entries


def _path_has_symlink(root: Path, relative: Path) -> bool:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def validate(entries: List[Path], root: Path = CODE_ROOT) -> List[str]:
    errors: List[str] = []
    if len(entries) != len(set(entries)):
        errors.append("allowlist contains duplicate paths")

    resolved_root = root.resolve()
    for relative in entries:
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"path escapes code subtree: {relative}")
            continue
        if relative.as_posix() == HASH_MANIFEST:
            errors.append(f"{HASH_MANIFEST} is generated and must not be allowlisted")
        if BANNED_PARTS.intersection(relative.parts):
            errors.append(f"banned legacy/monorepo-only path: {relative}")
        if _path_has_symlink(root, relative):
            errors.append(f"allowlisted path traverses a symlink: {relative}")
            continue
        absolute = root / relative
        try:
            absolute.resolve().relative_to(resolved_root)
        except ValueError:
            errors.append(f"resolved path escapes code subtree: {relative}")
            continue
        if not absolute.is_file():
            errors.append(f"allowlisted file is missing: {relative}")
    return errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_hash_manifest(destination: Path, entries: Iterable[Path]) -> None:
    lines = [
        f"{sha256_file(destination / relative)}  {relative.as_posix()}"
        for relative in sorted(entries, key=lambda item: item.as_posix())
    ]
    (destination / HASH_MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(destination / HASH_MANIFEST, 0o644)


def materialize_export(destination: Path, entries: List[Path]) -> None:
    errors = validate(entries)
    if errors:
        raise ValueError("\n".join(errors))

    destination = destination.expanduser().resolve()
    try:
        destination.relative_to(CODE_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("export destination must be outside the source code subtree")

    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"export destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    for relative in sorted(entries, key=lambda item: item.as_posix()):
        source = CODE_ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        os.chmod(target, 0o644)
    _write_hash_manifest(destination, entries)


def _actual_tree_files(root: Path) -> Set[Path]:
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }


def _read_hash_manifest(root: Path) -> Dict[Path, str]:
    manifest = root / HASH_MANIFEST
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError(f"missing regular hash manifest: {HASH_MANIFEST}")
    hashes: Dict[Path, str] = {}
    for line_number, raw in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = raw.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"malformed hash manifest line {line_number}")
        digest, name = parts
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(f"invalid SHA-256 on manifest line {line_number}") from exc
        relative = Path(name)
        if relative in hashes:
            raise ValueError(f"duplicate hash-manifest path: {name}")
        hashes[relative] = digest
    return hashes


def check_tree(root: Path) -> List[str]:
    errors: List[str] = []
    root = root.expanduser().resolve()
    if not root.is_dir():
        return [f"export tree is not a directory: {root}"]

    try:
        entries = load_allowlist(root)
    except (OSError, UnicodeError) as exc:
        return [f"cannot read exported allowlist: {exc}"]
    errors.extend(validate(entries, root=root))

    for path in root.rglob("*"):
        if path.is_symlink():
            errors.append(f"symlink is forbidden in export tree: {path.relative_to(root)}")

    expected = set(entries) | {Path(HASH_MANIFEST)}
    actual = _actual_tree_files(root)
    for missing in sorted(expected - actual, key=lambda item: item.as_posix()):
        errors.append(f"missing export member: {missing.as_posix()}")
    for extra in sorted(actual - expected, key=lambda item: item.as_posix()):
        errors.append(f"unexpected export member: {extra.as_posix()}")

    try:
        recorded = _read_hash_manifest(root)
    except ValueError as exc:
        errors.append(str(exc))
        return errors
    if set(recorded) != set(entries):
        errors.append("hash manifest paths do not exactly match the allowlist")
    for relative in sorted(set(entries) & set(recorded), key=lambda item: item.as_posix()):
        if (root / relative).is_file() and not (root / relative).is_symlink():
            actual_hash = sha256_file(root / relative)
            if actual_hash != recorded[relative]:
                errors.append(f"SHA-256 mismatch: {relative.as_posix()}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--export-dir", type=Path, help="copy the allowlisted tree to this empty directory"
    )
    actions.add_argument(
        "--check-tree", type=Path, help="verify an already materialized export tree"
    )
    parser.add_argument("--list", action="store_true", help="print validated export paths")
    args = parser.parse_args()

    entries = load_allowlist()
    errors = validate(entries)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    if args.list:
        for entry in entries:
            print(entry.as_posix())

    if args.export_dir is not None:
        try:
            materialize_export(args.export_dir, entries)
        except ValueError as exc:
            for error in str(exc).splitlines():
                print(f"ERROR: {error}")
            return 1
        print(f"materialized public export: {args.export_dir.resolve()}")
        return 0

    if args.check_tree is not None:
        tree_errors = check_tree(args.check_tree)
        if tree_errors:
            for error in tree_errors:
                print(f"ERROR: {error}")
            return 1
        print(f"public export tree valid: {args.check_tree.resolve()} ({len(entries)} files)")
        return 0

    print(f"public export allowlist valid: {len(entries)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
