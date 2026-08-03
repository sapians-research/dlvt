# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 2.2.x (current line) | ✅ security fixes via dependency-lock updates |
| ≤ 2.1.x | ❌ superseded; upgrade to the current line |

The package targets Python 3.10–3.13. Python 3.9 support ended with 2.2.0rc2
(Python 3.9 reached end-of-life in October 2025).

## Reporting a vulnerability

This is research software with no network surface of its own; the most
likely issues are vulnerable transitive dependencies or unsafe example
usage.

- Preferred: open a **private security advisory** on GitHub
  (Security → Advisories → "Report a vulnerability") on
  `wbendinelli/dlvt`.
- Alternatively, email the author listed in `CITATION.cff`.

Please include the affected version/commit, a minimal reproduction, and the
impact you foresee. You should receive an acknowledgment within 7 days.

## Dependency policy

`uv.lock` records the tested resolution; Dependabot alerts on the public
repository are triaged on every release, and security patches to locked
dependencies are shipped as maintenance releases (see `CHANGELOG.md`).

## Scope-of-use reminder

Independently of security, the model must not be used to diagnose, rank, or
make personnel decisions about individuals; see "Scientific boundaries" in
`README.md`.
