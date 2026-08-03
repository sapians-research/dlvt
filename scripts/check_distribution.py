"""Inspect DLVT wheel/sdist members and metadata before release."""
from __future__ import annotations

import argparse
import email.parser
import hashlib
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import List, Set
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
GENERATED_SDIST_MEMBERS = {
    "PKG-INFO",
    "setup.cfg",
    "dlvt.egg-info/PKG-INFO",
    "dlvt.egg-info/SOURCES.txt",
    "dlvt.egg-info/dependency_links.txt",
    "dlvt.egg-info/requires.txt",
    "dlvt.egg-info/top_level.txt",
}


def source_version() -> str:
    match = VERSION_RE.search((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if match is None:
        raise ValueError("cannot read version from pyproject.toml")
    return match.group(1)


def allowed_source_members() -> Set[str]:
    entries: Set[str] = set()
    for raw in (ROOT / "PUBLIC_EXPORT_ALLOWLIST.txt").read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if value and not value.startswith("#"):
            entries.add(PurePosixPath(value).as_posix())
    return entries | {"EXPORT_SHA256SUMS.txt"}


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def _metadata_errors(payload: bytes, expected_version: str) -> List[str]:
    message = email.parser.BytesParser().parsebytes(payload)
    errors: List[str] = []
    if message.get("Name") != "dlvt":
        errors.append(f"unexpected distribution name: {message.get('Name')!r}")
    if message.get("Version") != expected_version:
        errors.append(f"unexpected distribution version: {message.get('Version')!r}")
    if message.get("Requires-Python") != ">=3.10":
        errors.append(f"unexpected Requires-Python: {message.get('Requires-Python')!r}")
    requirements = message.get_all("Requires-Dist", [])
    for dependency in ("numpy", "scipy", "matplotlib"):
        if not any(item.lower().startswith(dependency) for item in requirements):
            errors.append(f"missing runtime dependency metadata: {dependency}")
    return errors


def inspect_wheel(path: Path, version: str) -> List[str]:
    errors: List[str] = []
    expected_modules = {
        item for item in allowed_source_members() if item.startswith("dlvt/") and item.endswith(".py")
    }
    dist_info = f"dlvt-{version}.dist-info"
    allowed_metadata = {
        f"{dist_info}/METADATA",
        f"{dist_info}/WHEEL",
        f"{dist_info}/top_level.txt",
        f"{dist_info}/RECORD",
        f"{dist_info}/LICENSE",
        f"{dist_info}/licenses/LICENSE",
    }
    with zipfile.ZipFile(path) as archive:
        members = {name for name in archive.namelist() if not name.endswith("/")}
        for name in members:
            if not _safe_member(name):
                errors.append(f"unsafe wheel member: {name}")
        unexpected = members - expected_modules - allowed_metadata
        missing = expected_modules - members
        errors.extend(f"unexpected wheel member: {name}" for name in sorted(unexpected))
        errors.extend(f"missing wheel module: {name}" for name in sorted(missing))
        metadata_name = f"{dist_info}/METADATA"
        if metadata_name not in members:
            errors.append("wheel METADATA is missing")
        else:
            errors.extend(_metadata_errors(archive.read(metadata_name), version))
    return errors


def inspect_sdist(path: Path, version: str) -> List[str]:
    errors: List[str] = []
    prefix = f"dlvt-{version}"
    allowed = allowed_source_members() | GENERATED_SDIST_MEMBERS
    required = {
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "dlvt/__init__.py",
        "dlvt/model.py",
        "dlvt/analysis.py",
    }
    relative_members: Set[str] = set()
    with tarfile.open(path, mode="r:gz") as archive:
        files = [member for member in archive.getmembers() if member.isfile() or member.issym()]
        for member in files:
            if member.issym() or member.islnk():
                errors.append(f"link is forbidden in sdist: {member.name}")
                continue
            if not _safe_member(member.name):
                errors.append(f"unsafe sdist member: {member.name}")
                continue
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != prefix:
                errors.append(f"sdist member has unexpected root: {member.name}")
                continue
            relative = PurePosixPath(*parts[1:]).as_posix()
            relative_members.add(relative)
            if relative not in allowed:
                errors.append(f"unexpected sdist member: {relative}")
        errors.extend(
            f"missing required sdist member: {name}" for name in sorted(required - relative_members)
        )
        pkg_info = f"{prefix}/PKG-INFO"
        try:
            payload = archive.extractfile(pkg_info)
        except KeyError:
            payload = None
        if payload is None:
            errors.append("sdist PKG-INFO is missing")
        else:
            errors.extend(_metadata_errors(payload.read(), version))
    return errors


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    args = parser.parse_args()

    version = source_version()
    errors = inspect_wheel(args.wheel, version) + inspect_sdist(args.sdist, version)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"wheel sha256  {sha256(args.wheel)}  {args.wheel.name}")
    print(f"sdist sha256  {sha256(args.sdist)}  {args.sdist.name}")
    print(f"distribution members and metadata valid for dlvt {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
