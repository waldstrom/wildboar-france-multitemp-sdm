#!/usr/bin/env python3
"""Validate that a release tree contains code and documentation only.

The check uses only the Python standard library, so it can run before the
geospatial environment is installed. It is intentionally conservative: data,
model artefacts and generated outputs must live outside version control.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path, PurePosixPath

MAX_TRACKED_BYTES = 5 * 1024 * 1024

FORBIDDEN_EXTENSIONS = {
    ".asc",
    ".aux",
    ".csv",
    ".dbf",
    ".docx",
    ".feather",
    ".gdb",
    ".gpkg",
    ".grb",
    ".grib",
    ".h5",
    ".hdf5",
    ".joblib",
    ".nc",
    ".npy",
    ".npz",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".prj",
    ".shp",
    ".shx",
    ".tif",
    ".tiff",
    ".tsv",
    ".xls",
    ".xlsx",
    ".zip",
}

FORBIDDEN_FILENAMES = {
    ".env",
    "credentials.json",
    "secrets.json",
}

PLACEHOLDER_ONLY_PREFIXES = (
    ("data",),
    ("outputs",),
    ("review", "input"),
    ("review", "output"),
)

REQUIRED_PLACEHOLDERS = {
    "data/README.md",
    "data/bias-maps/README.md",
    "data/bioclim/README.md",
    "data/dem/README.md",
    "data/era5/README.md",
    "data/hunting-bag/README.md",
    "data/land-cover/README.md",
    "data/lc-distances/README.md",
    "data/sat-indices/README.md",
    "data/special/README.md",
    "data/stacks/README.md",
    "data/theia-special/README.md",
    "outputs/README.md",
    "review/input/README.md",
    "review/output/README.md",
}

CONFLICT_PREFIXES = ("<<<<<<< ", ">>>>>>> ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        help="Tree to validate. Defaults to the enclosing Git repository.",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Inspect every file below --root instead of only Git-tracked files.",
    )
    return parser.parse_args()


def repository_root(explicit_root: Path | None) -> Path:
    """Return an explicit root or the enclosing Git work tree."""
    if explicit_root is not None:
        root = explicit_root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"validation root does not exist: {root}")
        return root

    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("not inside a Git repository; pass --root and --all-files")
    return Path(result.stdout.strip()).resolve()


def all_files(root: Path) -> list[Path]:
    """List regular files while excluding an embedded Git administration tree."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    )


def tracked_files(root: Path) -> list[Path]:
    """List files tracked by Git below *root*."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError("could not list Git-tracked files; use --all-files for an export")
    names = [name for name in result.stdout.decode("utf-8").split("\0") if name]
    return [root / PurePosixPath(name) for name in names]


def is_placeholder_only(relative: PurePosixPath) -> bool:
    parts = relative.parts
    return any(parts[: len(prefix)] == prefix for prefix in PLACEHOLDER_ONLY_PREFIXES)


def scan_conflict_markers(path: Path) -> list[str]:
    """Return unresolved merge-marker descriptions from a small text file."""
    if path.stat().st_size > 2 * 1024 * 1024:
        return []
    raw = path.read_bytes()
    if b"\0" in raw:
        return []
    text = raw.decode("utf-8", errors="replace")
    findings: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped == "=======" or stripped.startswith(CONFLICT_PREFIXES):
            findings.append(f"{line_number}: {stripped[:80]}")
    return findings


def main() -> int:
    args = parse_args()
    try:
        root = repository_root(args.root)
        files = all_files(root) if args.all_files else tracked_files(root)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Repository hygiene check could not start: {exc}", file=sys.stderr)
        return 2

    relative_files = {
        PurePosixPath(path.relative_to(root).as_posix()): path for path in files
    }
    errors: list[str] = []

    for relative, path in sorted(relative_files.items(), key=lambda item: str(item[0])):
        if not path.exists():
            errors.append(f"tracked path is missing from the work tree: {relative}")
            continue

        if path.is_symlink():
            errors.append(f"symbolic links are not allowed in a release tree: {relative}")

        if is_placeholder_only(relative) and relative.name.lower() != "readme.md":
            errors.append(f"data/output location may contain README.md only: {relative}")

        if relative.suffix.lower() in FORBIDDEN_EXTENSIONS:
            errors.append(f"forbidden data or binary extension: {relative}")

        if relative.name.lower() in FORBIDDEN_FILENAMES:
            errors.append(f"forbidden credential filename: {relative}")

        size = path.stat().st_size
        if size > MAX_TRACKED_BYTES:
            errors.append(
                f"file exceeds {MAX_TRACKED_BYTES // (1024 * 1024)} MiB: "
                f"{relative} ({size / (1024 * 1024):.1f} MiB)"
            )

        for marker in scan_conflict_markers(path):
            errors.append(f"unresolved merge marker in {relative}:{marker}")

    present = {str(path) for path in relative_files}
    for missing in sorted(REQUIRED_PLACEHOLDERS - present):
        errors.append(f"required placeholder is missing: {missing}")

    if errors:
        print("Repository hygiene check FAILED:\n", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    scope = "files" if args.all_files else "tracked files"
    print(
        f"Repository hygiene check passed: {len(files)} {scope}; "
        "protected locations contain documentation only."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
