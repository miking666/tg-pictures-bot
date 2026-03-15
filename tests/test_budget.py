from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import budget


class BudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.budget_path = Path(self.temp_dir.name) / "budget.json"
        budget.configure_budget_file(self.budget_path)
        self.budget_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "stats_message_id": None,
                    "lifetime": {
                        "spent": 0.0,
                        "image_generations": 0,
                        "fallback_cards": 0,
                    },
                    "events": [],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_thread_safe_record_image_generation(self) -> None:
        def worker() -> None:
            for _ in range(50):
                budget.record_image_generation("user", 42)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        snapshot = budget.get_budget_snapshot()
        self.assertEqual(snapshot.window_image_generations, 400)
        self.assertEqual(snapshot.lifetime_image_generations, 400)
        self.assertAlmostEqual(snapshot.window_spent, 40.0, places=4)

    def test_old_events_do_not_block_new_generation(self) -> None:
        old_ts = time.time() - budget.MONTH_SECONDS - 10
        self.budget_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "stats_message_id": None,
                    "lifetime": {
                        "spent": 1.0,
                        "image_generations": 1,
                        "fallback_cards": 0,
                    },
                    "events": [
                        {
                            "kind": "image_generation",
                            "cost": 0.1,
                            "ts": old_ts,
                            "uid": 1,
                            "name": "legacy",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        snapshot = budget.get_budget_snapshot()
        self.assertEqual(snapshot.window_image_generations, 0)
        self.assertTrue(budget.can_generate(0.1))

    def test_corrupt_file_is_backed_up(self) -> None:
        self.budget_path.write_text("{broken json", encoding="utf-8")

        with self.assertLogs(level="ERROR"):
            snapshot = budget.get_budget_snapshot()

        self.assertEqual(snapshot.window_image_generations, 0)
        backups = list(self.budget_path.parent.glob("budget.json.corrupt-*.bak"))
        self.assertTrue(backups)

    def test_status_escapes_user_names(self) -> None:
        now = time.time()
        self.budget_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "stats_message_id": None,
                    "lifetime": {
                        "spent": 0.1,
                        "image_generations": 1,
                        "fallback_cards": 0,
                    },
                    "events": [
                        {
                            "kind": "image_generation",
                            "cost": 0.1,
                            "ts": now,
                            "uid": 1,
                            "name": "<b>bad</b>",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        status = budget.get_budget_status(10.0)
        self.assertIn("&lt;b&gt;bad&lt;/b&gt;", status)


if __name__ == "__main__":
    unittest.main()
