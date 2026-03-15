from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from storage_utils import DEFAULT_DATA_DIR, atomic_write_json, backup_corrupt_file, exclusive_file_lock


DEFAULT_IMAGE_GENERATION_COST = 0.10
MONTH_SECONDS = 30 * 24 * 3600
LEGACY_BUDGET_FILE = Path(__file__).resolve().parent / "budget.json"

_BUDGET_FILE = DEFAULT_DATA_DIR / "budget.json"
_IMAGE_GENERATION_COST = DEFAULT_IMAGE_GENERATION_COST


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    window_spent: float
    window_image_generations: int
    window_fallback_cards: int
    lifetime_spent: float
    lifetime_image_generations: int
    lifetime_fallback_cards: int
    top_users: list[tuple[int, str, int]]


def configure_budget_file(path: Path, image_generation_cost: float | None = None) -> None:
    global _BUDGET_FILE, _IMAGE_GENERATION_COST
    _BUDGET_FILE = path
    if image_generation_cost is not None:
        _IMAGE_GENERATION_COST = image_generation_cost


def record_image_generation(user_name: str = "", user_id: int = 0) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["events"].append(
            {
                "kind": "image_generation",
                "cost": _IMAGE_GENERATION_COST,
                "ts": time.time(),
                "uid": user_id or None,
                "name": user_name,
            }
        )
        lifetime = payload["lifetime"]
        lifetime["spent"] = round(lifetime["spent"] + _IMAGE_GENERATION_COST, 4)
        lifetime["image_generations"] += 1

    _update_payload(mutate)


def record_fallback(reason: str = "") -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["events"].append(
            {
                "kind": "fallback_card",
                "cost": 0.0,
                "ts": time.time(),
                "reason": reason,
            }
        )
        payload["lifetime"]["fallback_cards"] += 1

    _update_payload(mutate)


def can_generate(max_budget: float) -> bool:
    snapshot = get_budget_snapshot()
    return snapshot.window_spent + _IMAGE_GENERATION_COST <= max_budget


def check_budget_warnings(max_budget: float) -> str | None:
    if max_budget <= 0:
        return None

    snapshot = get_budget_snapshot()
    ratio = snapshot.window_spent / max_budget
    if ratio < 0.8:
        return None

    remaining = max(0.0, max_budget - snapshot.window_spent)
    remaining_images = int(remaining / _IMAGE_GENERATION_COST)
    if ratio >= 1.0:
        return (
            f"🚨 <b>Бюджет исчерпан за 30 дней!</b>\n"
            f"Потрачено: ${snapshot.window_spent:.2f} из ${max_budget:.2f}\n"
            f"Осталось генераций: {remaining_images}"
        )
    return (
        f"⚠️ <b>Бюджет почти исчерпан ({ratio:.0%})</b>\n"
        f"Потрачено: ${snapshot.window_spent:.2f} из ${max_budget:.2f}\n"
        f"Осталось генераций: ~{remaining_images}"
    )


def get_stats_message_id() -> int | None:
    payload = _read_payload()
    message_id = payload.get("stats_message_id")
    return message_id if isinstance(message_id, int) else None


def set_stats_message_id(message_id: int) -> None:
    _update_payload(lambda payload: payload.__setitem__("stats_message_id", message_id))


def get_budget_snapshot() -> BudgetSnapshot:
    payload = _read_payload()
    return _build_snapshot(payload)


def get_budget_status(max_budget: float) -> str:
    snapshot = get_budget_snapshot()
    remaining = max(0.0, max_budget - snapshot.window_spent)
    lines = [
        "📊 <b>Статистика бота</b>",
        "",
        "За последние 30 дней:",
        f"💰 Потрачено: <b>${snapshot.window_spent:.2f}</b> из ${max_budget:.2f}",
        f"🖼 Gemini-генераций: <b>{snapshot.window_image_generations}</b>",
        f"🧰 Локальных fallback-открыток: <b>{snapshot.window_fallback_cards}</b>",
        f"📦 Осталось генераций: ~{int(remaining / _IMAGE_GENERATION_COST)}",
        "",
        "За все время:",
        f"💳 Потрачено: <b>${snapshot.lifetime_spent:.2f}</b>",
        f"🖼 Gemini-генераций: <b>{snapshot.lifetime_image_generations}</b>",
        f"🧰 Fallback-открыток: <b>{snapshot.lifetime_fallback_cards}</b>",
    ]
    if snapshot.top_users:
        lines.append("")
        lines.append("👥 <b>Топ за 30 дней:</b>")
        for user_id, raw_name, count in snapshot.top_users[:5]:
            display_name = escape(raw_name or f"id:{user_id}")
            link = f"<a href=\"tg://user?id={user_id}\">{display_name}</a>"
            lines.append(f"  • {link} — {count} шт.")
        lines.append(f"\nВсего активных пользователей: {len(snapshot.top_users)}")
    return "\n".join(lines)


def _update_payload(mutator) -> None:
    path = _BUDGET_FILE
    with exclusive_file_lock(path):
        payload, changed = _load_payload_locked(path)
        mutator(payload)
        atomic_write_json(path, payload)


def _read_payload() -> dict[str, Any]:
    path = _BUDGET_FILE
    with exclusive_file_lock(path):
        payload, changed = _load_payload_locked(path)
        if changed:
            atomic_write_json(path, payload)
        return payload


def _load_payload_locked(path: Path) -> tuple[dict[str, Any], bool]:
    source_path = path
    if not source_path.exists() and path != LEGACY_BUDGET_FILE and LEGACY_BUDGET_FILE.exists():
        source_path = LEGACY_BUDGET_FILE

    if not source_path.exists():
        return _default_payload(), False

    try:
        raw_payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        backup_path = backup_corrupt_file(source_path)
        logging.exception(
            "Budget storage at %s is corrupt. Backed up to %s and reset state.",
            source_path,
            backup_path,
        )
        return _default_payload(), True

    payload, changed = _normalize_payload(raw_payload)
    if source_path != path:
        changed = True
        logging.info("Migrating budget storage from %s to %s", source_path, path)
    return payload, changed


def _normalize_payload(raw_payload: Any) -> tuple[dict[str, Any], bool]:
    if isinstance(raw_payload, dict) and raw_payload.get("schema_version") == 2:
        payload = {
            "schema_version": 2,
            "stats_message_id": _coerce_optional_int(raw_payload.get("stats_message_id")),
            "lifetime": {
                "spent": round(_coerce_float(raw_payload.get("lifetime", {}).get("spent", 0.0)), 4),
                "image_generations": max(
                    0,
                    _coerce_int(raw_payload.get("lifetime", {}).get("image_generations", 0)),
                ),
                "fallback_cards": max(
                    0,
                    _coerce_int(raw_payload.get("lifetime", {}).get("fallback_cards", 0)),
                ),
            },
            "events": _normalize_events(raw_payload.get("events", [])),
        }
        changed = payload != raw_payload
    else:
        payload = _migrate_legacy_payload(raw_payload if isinstance(raw_payload, dict) else {})
        changed = True

    changed = _prune_old_events(payload) or changed
    return payload, changed


def _migrate_legacy_payload(raw_payload: dict[str, Any]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for generation in raw_payload.get("generations", []):
        if not isinstance(generation, dict):
            continue
        ts = _coerce_float(generation.get("ts", 0.0))
        if ts <= 0:
            continue
        event: dict[str, Any] = {
            "kind": "image_generation",
            "cost": _IMAGE_GENERATION_COST,
            "ts": ts,
        }
        user_id = _coerce_int(generation.get("uid", 0))
        if user_id > 0:
            event["uid"] = user_id
        name = str(generation.get("name", "") or "")
        if name:
            event["name"] = name
        events.append(event)

    return {
        "schema_version": 2,
        "stats_message_id": _coerce_optional_int(raw_payload.get("stats_message_id")),
        "lifetime": {
            "spent": round(_coerce_float(raw_payload.get("spent", 0.0)), 4),
            "image_generations": max(0, _coerce_int(raw_payload.get("images", 0))),
            "fallback_cards": 0,
        },
        "events": events,
    }


def _normalize_events(raw_events: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_events, list):
        return []

    normalized: list[dict[str, Any]] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            continue
        kind = raw_event.get("kind")
        ts = _coerce_float(raw_event.get("ts", 0.0))
        if kind not in {"image_generation", "fallback_card"} or ts <= 0:
            continue
        event: dict[str, Any] = {
            "kind": kind,
            "ts": ts,
            "cost": round(_coerce_float(raw_event.get("cost", 0.0)), 4),
        }
        if kind == "image_generation":
            user_id = _coerce_int(raw_event.get("uid", 0))
            if user_id > 0:
                event["uid"] = user_id
            name = str(raw_event.get("name", "") or "")
            if name:
                event["name"] = name
            event["cost"] = _IMAGE_GENERATION_COST if event["cost"] <= 0 else event["cost"]
        else:
            reason = str(raw_event.get("reason", "") or "")
            if reason:
                event["reason"] = reason
            event["cost"] = 0.0
        normalized.append(event)
    return normalized


def _prune_old_events(payload: dict[str, Any]) -> bool:
    cutoff = time.time() - MONTH_SECONDS
    events = payload.get("events", [])
    fresh_events = [event for event in events if event.get("ts", 0.0) > cutoff]
    if fresh_events == events:
        return False
    payload["events"] = fresh_events
    return True


def _build_snapshot(payload: dict[str, Any]) -> BudgetSnapshot:
    window_spent = 0.0
    window_image_generations = 0
    window_fallback_cards = 0
    user_stats: dict[int, dict[str, Any]] = {}

    for event in payload.get("events", []):
        kind = event.get("kind")
        if kind == "image_generation":
            window_image_generations += 1
            window_spent = round(window_spent + _coerce_float(event.get("cost", _IMAGE_GENERATION_COST)), 4)
            user_id = _coerce_int(event.get("uid", 0))
            if user_id > 0:
                stats = user_stats.setdefault(user_id, {"name": "", "count": 0})
                stats["count"] += 1
                if event.get("name"):
                    stats["name"] = str(event["name"])
        elif kind == "fallback_card":
            window_fallback_cards += 1

    top_users = [
        (user_id, data["name"], data["count"])
        for user_id, data in sorted(
            user_stats.items(),
            key=lambda item: item[1]["count"],
            reverse=True,
        )
    ]
    lifetime = payload.get("lifetime", {})
    return BudgetSnapshot(
        window_spent=window_spent,
        window_image_generations=window_image_generations,
        window_fallback_cards=window_fallback_cards,
        lifetime_spent=round(_coerce_float(lifetime.get("spent", 0.0)), 4),
        lifetime_image_generations=max(0, _coerce_int(lifetime.get("image_generations", 0))),
        lifetime_fallback_cards=max(0, _coerce_int(lifetime.get("fallback_cards", 0))),
        top_users=top_users,
    )


def _default_payload() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "stats_message_id": None,
        "lifetime": {
            "spent": 0.0,
            "image_generations": 0,
            "fallback_cards": 0,
        },
        "events": [],
    }


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_optional_int(value: Any) -> int | None:
    result = _coerce_int(value)
    return result or None


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
