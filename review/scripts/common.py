from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import yaml

LOG = logging.getLogger("review_corrections")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and fail clearly for missing or malformed files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Top level of {path} must be a mapping")
    return data


def dump_yaml(data: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically on the same filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def write_json(data: Any, path: Path) -> None:
    atomic_write_text(
        Path(path),
        json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
    )


def write_csv(rows: Iterable[dict[str, Any]], path: Path, fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def deep_get(mapping: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    cur: Any = mapping
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def deep_set(mapping: dict[str, Any], dotted_path: str, value: Any) -> None:
    cur = mapping
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``patch`` into ``base`` in place and return ``base``."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def flatten_mapping(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_mapping(child, path))
    elif isinstance(value, list):
        out[prefix] = value
    else:
        out[prefix] = value
    return out


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_inventory(path: Path, hash_limit_bytes: int = 50 * 1024 * 1024) -> dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    if path.is_file() and stat.st_size <= hash_limit_bytes:
        record["sha256"] = sha256_file(path)
        record["hash_status"] = "complete"
    else:
        record["sha256"] = None
        record["hash_status"] = "skipped_large_or_directory"
    return record


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    value = re.sub(r"_+", "_", value).strip("_.")
    return value or "unnamed"


def local_run_id(timezone_name: str, suffix: str = "hunting-correction") -> str:
    now = datetime.now(ZoneInfo(timezone_name))
    zone = now.tzname() or "LOCAL"
    return f"{now:%Y%m%d_%H%M%S}_{sanitize_name(zone)}_{sanitize_name(suffix)}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(Path(path).resolve())


def git_metadata(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        return result.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "remote": run("config", "--get", "remote.origin.url"),
        "dirty": bool(status),
        "status_porcelain": status,
    }


def environment_metadata(repo_root: Path) -> dict[str, Any]:
    return {
        "created_utc": now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cwd": str(Path.cwd()),
        "repo_root": str(Path(repo_root).resolve()),
        "git": git_metadata(repo_root),
    }


def stream_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    prefix: str,
    env: dict[str, str] | None = None,
) -> int:
    """Run a child process while teeing merged stdout/stderr to console and file."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})
    LOG.info("Command: %s", " ".join(command))
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n$ {' '.join(command)}\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            clean = line.rstrip("\n")
            rendered = f"[{prefix}] {clean}"
            print(rendered, flush=True)
            log_handle.write(rendered + "\n")
            log_handle.flush()
        return process.wait()


def hardlink_or_copy(source: Path, destination: Path) -> str:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def newest(paths: Iterable[Path]) -> Path | None:
    candidates = [Path(p) for p in paths if Path(p).exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def numbered_directory(parent: Path, number: int, label: str) -> Path:
    path = Path(parent) / f"{number:02d}_{sanitize_name(label)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class RunLock:
    """Simple run-directory lock that leaves useful stale-lock diagnostics."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.acquired = False

    def acquire(self, force: bool = False) -> None:
        if self.path.exists() and not force:
            data = self.path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"Run lock already exists: {self.path}\n{data}\n"
                "Use --force only after confirming that no other correction run is active."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.path,
            json.dumps(
                {
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "created_utc": now_iso(),
                },
                indent=2,
            )
            + "\n",
        )
        self.acquired = True

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
