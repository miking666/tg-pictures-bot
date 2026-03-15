from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from aiogram.exceptions import DataNotDictLikeError
from aiogram.fsm.storage.base import BaseStorage, StateType, StorageKey

from storage_utils import (
    DEFAULT_DATA_DIR,
    atomic_write_json,
    backup_corrupt_file,
    exclusive_file_lock,
)


class JsonFileStorage(BaseStorage):
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (DEFAULT_DATA_DIR / "fsm_storage.json")

    async def close(self) -> None:
        return None

    async def set_state(self, key: StorageKey, state: StateType = None) -> None:
        with exclusive_file_lock(self.path):
            payload = self._read_payload()
            records = payload["records"]
            record = records.setdefault(self._key_to_str(key), {"state": None, "data": {}})
            record["state"] = state.state if hasattr(state, "state") else state
            self._cleanup_record(records, key)
            atomic_write_json(self.path, payload)

    async def get_state(self, key: StorageKey) -> str | None:
        record = self._load_records().get(self._key_to_str(key), {})
        state = record.get("state")
        return state if isinstance(state, str) else None

    async def set_data(self, key: StorageKey, data: Mapping[str, Any]) -> None:
        if not isinstance(data, Mapping):
            msg = f"Data must be a dict or dict-like object, got {type(data).__name__}"
            raise DataNotDictLikeError(msg)

        with exclusive_file_lock(self.path):
            payload = self._read_payload()
            records = payload["records"]
            record = records.setdefault(self._key_to_str(key), {"state": None, "data": {}})
            record["data"] = deepcopy(dict(data))
            self._cleanup_record(records, key)
            atomic_write_json(self.path, payload)

    async def get_data(self, key: StorageKey) -> dict[str, Any]:
        record = self._load_records().get(self._key_to_str(key), {})
        data = record.get("data", {})
        if isinstance(data, dict):
            return deepcopy(data)
        return {}

    @staticmethod
    def _key_to_str(key: StorageKey) -> str:
        return "|".join(
            (
                str(key.bot_id),
                str(key.chat_id),
                str(key.user_id),
                str(key.thread_id or ""),
                key.business_connection_id or "",
                key.destiny,
            )
        )

    def _cleanup_record(self, records: dict[str, dict[str, Any]], key: StorageKey) -> None:
        storage_key = self._key_to_str(key)
        record = records.get(storage_key)
        if not record:
            return
        if not record.get("state") and not record.get("data"):
            records.pop(storage_key, None)

    def _load_records(self) -> dict[str, dict[str, Any]]:
        with exclusive_file_lock(self.path):
            payload = self._read_payload()
            return deepcopy(payload["records"])

    def _read_payload(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "records": {}}

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            backup_path = backup_corrupt_file(self.path)
            logging.exception(
                "FSM storage at %s is corrupt. Backed up to %s and reset state.",
                self.path,
                backup_path,
            )
            payload = {"schema_version": 1, "records": {}}
            atomic_write_json(self.path, payload)
            return payload

        records = payload.get("records", {})
        if not isinstance(records, dict):
            records = {}
        normalized = {"schema_version": 1, "records": records}
        if normalized != payload:
            atomic_write_json(self.path, normalized)
        return normalized
