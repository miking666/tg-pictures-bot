from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / ".data"
_FALLBACK_LOCK = Lock()


def resolve_runtime_path(env_name: str, default_path: Path | str) -> Path:
    raw_value = os.getenv(env_name, "").strip()
    path = Path(raw_value).expanduser() if raw_value else Path(default_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    ensure_parent_dir(path)
    lock_path = path.with_name(f"{path.name}.lock")
    if fcntl is None:  # pragma: no cover
        with _FALLBACK_LOCK:
            yield
        return

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def atomic_write_json(path: Path, payload: Any) -> None:
    ensure_parent_dir(path)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def backup_corrupt_file(path: Path) -> Path | None:
    if not path.exists():
        return None

    backup_path = path.with_name(
        f"{path.name}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}.bak"
    )
    try:
        os.replace(path, backup_path)
    except OSError:
        logging.exception("Failed to move corrupt file %s", path)
        return None
    return backup_path
