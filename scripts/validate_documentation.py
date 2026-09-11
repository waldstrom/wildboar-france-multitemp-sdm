#!/usr/bin/env python3
"""Check configuration inventory, local documentation links and core metadata.

Requires PyYAML only. Does not import the modelling stack, read research data,
resolve remote links or launch models. Works in a checkout or code-only export.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys instead of silently keeping the last one."""


def unique_mapping(loader: UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping", node.start_mark,
                f"duplicate key {key!r}", key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping
)


def repository_files(root: Path, all_files: bool = False) -> list[Path]:
    if all_files:
        return sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.relative_to(root).parts)
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        capture_output=True, check=False,
    )
    if result.returncode == 0:
        return sorted({root / p for p in result.stdout.decode().split("\0") if p and (root / p).is_file()})
    return sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts)


def without_fences(text: str) -> str:
    return re.sub(r"^```[^\n]*\n.*?^```\s*$", "", text, flags=re.M | re.S)


def heading_ids(text: str) -> set[str]:
    seen: dict[str, int] = {}
    anchors = set()
    for match in re.finditer(r"^#{1,6}\s+(.+?)\s*#*\s*$", without_fences(text), re.M):
        title = re.sub(r"[^\w\-\s]", "", match.group(1).lower())
        slug = title.replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(f"{slug}-{count}" if count else slug)
    return anchors


def config_refs(value, root: Path, context: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"feature_config", "aggregate_config"} and isinstance(item, str):
                if not (root / item).is_file():
                    errors.append(f"{context}: missing {key}: {item}")
            config_refs(item, root, context, errors)
    elif isinstance(value, list):
        for item in value:
            config_refs(item, root, context, errors)


def validate(root: Path, all_files: bool = False) -> list[str]:
    errors: list[str] = []
    files = repository_files(root, all_files=all_files)
    parsed = {}
    for path in files:
        if path.suffix in {".yaml", ".yml", ".cff"}:
            rel = path.relative_to(root).as_posix()
            try:
                parsed[rel] = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
                config_refs(parsed[rel], root, rel, errors)
            except (yaml.YAMLError, ValueError, TypeError) as exc:
                errors.append(f"{rel}: {exc}")

    catalog = parsed.get("configs/catalog.yaml") or {}
    entries = catalog.get("entries", [])
    actual = {p.relative_to(root).as_posix() for p in files if p.suffix == ".yaml"
              and (p.is_relative_to(root / "configs") or p.is_relative_to(root / "review/config"))}
    actual.discard("configs/catalog.yaml")
    recorded: set[str] = set()
    for entry in entries:
        name = entry.get("path", "")
        if name in recorded:
            errors.append(f"duplicate config catalog entry: {name}")
        recorded.add(name)
        for key in ("path", "kind", "status", "runner", "study_references"):
            if not entry.get(key):
                errors.append(f"{name}: catalog lacks {key}")
        for key in ("path", "runner", "derived_from"):
            if entry.get(key) and not (root / entry[key]).is_file():
                errors.append(f"{name}: missing catalog {key}: {entry[key]}")
    for name in sorted(actual - recorded):
        errors.append(f"configuration missing from catalog: {name}")
    for name in sorted(recorded - actual):
        errors.append(f"catalog entry has no versioned configuration: {name}")

    expected = {(s, m) for s in ("Winter", "Summer") for m in ("monotemporal", "multitemporal")}
    for name in ("loyo_all_sources.yaml", "loyo_gbif.yaml"):
        rel = f"configs/reviewed/{name}"
        scenarios = (parsed.get(rel) or {}).get("scenarios", [])
        found = set()
        for item in scenarios:
            cfg = item.get("config", {})
            mt = cfg.get("multitemporal", {})
            seasons = mt.get("seasons", [])
            season = seasons[0] if len(seasons) == 1 else None
            pair = (season, item.get("script"))
            if pair in found:
                errors.append(f"{rel}: duplicate season/temporal pair {pair}")
            found.add(pair)
            if mt.get("include_hunting") is not (season == "Winter"):
                errors.append(f"{rel}: hunting setting inconsistent with reviewed {pair}")
            years = list(range(2017 if season == "Winter" else 2018, 2024))
            if mt.get("years") != years or cfg.get("run_test_year") != "all":
                errors.append(f"{rel}: unexpected LOYO coverage for {pair}")
            if cfg.get("variable_importance", {}).get("permutation", {}).get("n_repeats") != 5:
                errors.append(f"{rel}: reviewed permutation repeat count must be five")
        if found != expected:
            errors.append(f"{rel}: expected complete season/temporal matrix")

    for path in files:
        rel = path.relative_to(root).as_posix()
        if path.suffix == ".md":
            text = without_fences(path.read_text(encoding="utf-8"))
            # Inline Markdown links/images, including badge destinations.
            for match in re.finditer(r"\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)", text):
                target = match.group(1).strip("<>")
                parsed_url = urlsplit(target)
                if parsed_url.scheme or parsed_url.netloc:
                    continue
                dest = (path.parent / unquote(parsed_url.path)).resolve() if parsed_url.path else path
                if not dest.exists():
                    errors.append(f"{rel}: broken local link {target}")
                elif parsed_url.fragment and dest.is_file() and dest.suffix == ".md":
                    if unquote(parsed_url.fragment) not in heading_ids(dest.read_text(encoding="utf-8")):
                        errors.append(f"{rel}: missing heading in {target}")
        if path.suffix == ".py" and not path.is_relative_to(root / "tests") and not path.is_relative_to(root / "review/tests"):
            for name in re.findall(r'''["'](configs/[A-Za-z0-9_./-]+\.yaml)["']''', path.read_text(encoding="utf-8")):
                if not (root / name).is_file():
                    errors.append(f"{rel}: missing literal configuration path {name}")

    cff = parsed.get("CITATION.cff") or {}
    try:
        metadata = json.loads((root / "codemeta.json").read_text(encoding="utf-8"))
        if cff.get("license") != "BSD-3-Clause" or "BSD 3-Clause License" not in (root / "LICENSE").read_text():
            errors.append("CITATION.cff and LICENSE must identify BSD-3-Clause")
        if metadata.get("name") != cff.get("title") or metadata.get("codeRepository") != cff.get("repository-code"):
            errors.append("CITATION.cff and codemeta.json disagree on title or repository")
        if metadata.get("license") != "https://spdx.org/licenses/BSD-3-Clause.html":
            errors.append("codemeta.json licence differs from repository licence")
        citation = cff.get("preferred-citation", {})
        if metadata.get("citation", {}).get("identifier") != "https://doi.org/" + citation.get("doi", ""):
            errors.append("preferred study DOI differs between metadata files")
        if not citation.get("authors") or not cff.get("authors") or cff.get("cff-version") != "1.2.0":
            errors.append("CITATION.cff lacks authors or expected format version")
    except (OSError, ValueError) as exc:
        errors.append(f"software metadata: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--all-files", action="store_true", help="Check an exported tree without consulting Git")
    args = parser.parse_args()
    errors = validate(args.root.resolve(), all_files=args.all_files)
    if errors:
        print("Documentation/configuration check FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Documentation/configuration check passed: YAML, catalog, reviewed presets, local links and core metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
