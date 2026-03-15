from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from storage_utils import BASE_DIR, DEFAULT_DATA_DIR, resolve_runtime_path


@dataclass(slots=True)
class Settings:
    bot_token: str
    gemini_api_key: str
    text_model: str
    audio_model: str
    image_model: str
    font_path: Path
    image_size: int
    max_text_length: int
    max_budget: float
    image_generation_cost: float
    admin_chat_id: int | None
    telegram_proxy: str | None
    data_dir: Path
    budget_file_path: Path
    fsm_storage_path: Path

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(BASE_DIR / ".env")

        bot_token = os.getenv("BOT_TOKEN", "").strip()
        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()

        missing = [
            name
            for name, value in (
                ("BOT_TOKEN", bot_token),
                ("GEMINI_API_KEY", gemini_api_key),
            )
            if not value
        ]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Missing required environment variables: {missing_str}")

        data_dir = resolve_runtime_path("DATA_DIR", DEFAULT_DATA_DIR)
        budget_file_path = resolve_runtime_path(
            "BUDGET_FILE_PATH", data_dir / "budget.json",
        )
        fsm_storage_path = resolve_runtime_path(
            "FSM_STORAGE_PATH", data_dir / "fsm_storage.json",
        )
        font_path = resolve_runtime_path(
            "CARD_FONT_PATH", BASE_DIR / "fonts/LiberationSans-Bold.ttf",
        )

        image_size = _parse_int("CARD_IMAGE_SIZE", os.getenv("CARD_IMAGE_SIZE", "1024"))
        max_text_length = _parse_int(
            "MAX_GREETING_LENGTH",
            os.getenv("MAX_GREETING_LENGTH", "350"),
        )
        max_budget = _parse_float(
            "MAX_BUDGET_USD",
            os.getenv("MAX_BUDGET_USD", "3.0"),
        )
        image_generation_cost = _parse_float(
            "IMAGE_GENERATION_COST",
            os.getenv("IMAGE_GENERATION_COST", "0.10"),
        )
        admin_chat_id = _parse_optional_int(
            "ADMIN_CHAT_ID",
            os.getenv("ADMIN_CHAT_ID", "").strip(),
        )
        telegram_proxy = os.getenv("TELEGRAM_PROXY", "").strip() or None

        data_dir.mkdir(parents=True, exist_ok=True)
        if not font_path.exists():
            raise ValueError(f"CARD_FONT_PATH does not exist: {font_path}")

        return cls(
            bot_token=bot_token,
            gemini_api_key=gemini_api_key,
            text_model=os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash").strip(),
            audio_model=os.getenv("GEMINI_AUDIO_MODEL", "gemini-2.5-flash").strip(),
            image_model=os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image").strip(),
            font_path=font_path,
            image_size=max(512, image_size),
            max_text_length=max(40, max_text_length),
            max_budget=max_budget,
            image_generation_cost=image_generation_cost,
            admin_chat_id=admin_chat_id,
            telegram_proxy=telegram_proxy,
            data_dir=data_dir,
            budget_file_path=budget_file_path,
            fsm_storage_path=fsm_storage_path,
        )


def _parse_int(name: str, raw_value: str) -> int:
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


def _parse_optional_int(name: str, raw_value: str) -> int | None:
    if not raw_value:
        return None
    return _parse_int(name, raw_value)


def _parse_float(name: str, raw_value: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0, got {raw_value!r}")
    return value
