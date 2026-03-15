from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aiogram.fsm.storage.base import StorageKey

from file_storage import JsonFileStorage


class JsonFileStorageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = Path(self.temp_dir.name) / "fsm_storage.json"
        self.key = StorageKey(bot_id=1, chat_id=2, user_id=3)

    async def asyncTearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_state_persists_between_instances(self) -> None:
        storage = JsonFileStorage(self.storage_path)
        await storage.set_state(self.key, "GreetingFlow:waiting_for_style")
        await storage.set_data(self.key, {"greeting_text": "Привет"})

        reloaded_storage = JsonFileStorage(self.storage_path)
        self.assertEqual(
            await reloaded_storage.get_state(self.key),
            "GreetingFlow:waiting_for_style",
        )
        self.assertEqual(
            await reloaded_storage.get_data(self.key),
            {"greeting_text": "Привет"},
        )

    async def test_empty_record_is_removed(self) -> None:
        storage = JsonFileStorage(self.storage_path)
        await storage.set_state(self.key, "GreetingFlow:waiting_for_style")
        await storage.set_data(self.key, {"greeting_text": "Привет"})

        await storage.set_state(self.key, None)
        await storage.set_data(self.key, {})

        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["records"], {})


if __name__ == "__main__":
    unittest.main()
