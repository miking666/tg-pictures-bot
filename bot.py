from __future__ import annotations

import asyncio
import logging
import traceback
from html import escape

from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import SimpleEventIsolation
from aiogram.types import BufferedInputFile, CallbackQuery, Message, TelegramObject
from aiogram.utils.keyboard import InlineKeyboardBuilder

from budget import (
    check_budget_warnings,
    configure_budget_file,
    get_budget_status,
    get_stats_message_id,
    set_stats_message_id,
)
from config import Settings
from file_storage import JsonFileStorage
from image_gen import BudgetExceededError, GreetingCardService
from styles import CARD_STYLES


router = Router()
settings: Settings | None = None
card_service: GreetingCardService | None = None


class StatsAccountFilter(BaseMiddleware):
    """Block all bot functionality for the stats/admin account."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        stats_id = get_settings().admin_chat_id
        if stats_id:
            user = data.get("event_from_user")
            if user and user.id == stats_id:
                return None
        return await handler(event, data)


class GreetingFlow(StatesGroup):
    waiting_for_style = State()


def get_settings() -> Settings:
    if settings is None:
        raise RuntimeError("Application settings are not initialized.")
    return settings


def get_card_service() -> GreetingCardService:
    if card_service is None:
        raise RuntimeError("Greeting card service is not initialized.")
    return card_service


def _user_info(user) -> str:
    if not user:
        return "unknown"
    name = user.full_name or ""
    return f"{name} (id={user.id}, @{user.username or '—'})"


def _fit_greeting_text(text: str, max_length: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized

    truncated = normalized[:max_length].rsplit(" ", maxsplit=1)[0].strip()
    return truncated or normalized[:max_length].strip()


async def update_stats(bot: Bot) -> None:
    admin_id = get_settings().admin_chat_id
    if not admin_id:
        return
    text = get_budget_status(get_settings().max_budget)
    msg_id = get_stats_message_id()
    if msg_id:
        try:
            await bot.edit_message_text(text, chat_id=admin_id, message_id=msg_id)
            return
        except (TelegramBadRequest, TelegramNetworkError):
            logging.debug("Could not edit stats message %s, sending new one", msg_id)
    try:
        msg = await bot.send_message(admin_id, text)
        set_stats_message_id(msg.message_id)
    except Exception:
        logging.exception("Failed to send stats message")


async def check_and_warn_budget(bot: Bot) -> None:
    admin_id = get_settings().admin_chat_id
    if not admin_id:
        return
    warning = check_budget_warnings(get_settings().max_budget)
    if warning:
        try:
            await bot.send_message(admin_id, warning)
        except Exception:
            logging.exception("Failed to send budget warning")


async def notify_admin(bot: Bot, error: Exception, user_info: str = "") -> None:
    tb = traceback.format_exception(error)
    short_tb = "".join(tb[-3:])[:1500]
    text = f"⚠️ <b>Ошибка:</b> {escape(str(error))}\n"
    if user_info:
        text += f"<b>От:</b> {escape(user_info)}\n"
    text += f"\n<pre>{escape(short_tb)}</pre>"
    admin_id = get_settings().admin_chat_id
    if not admin_id:
        return
    try:
        await bot.send_message(admin_id, text)
    except Exception:
        logging.exception("Failed to notify admin %s", admin_id)


async def _restore_style_buttons(
    callback: CallbackQuery,
    greeting_text: str,
    progress: Message,
    error_msg: str,
) -> None:
    """Restore style selection buttons after a generation error."""
    try:
        await callback.message.edit_text(  # type: ignore[union-attr]
            "Текст для открытки:\n"
            f"{escape(greeting_text)}\n\n"
            f"{error_msg}",
            reply_markup=build_styles_keyboard(),
        )
    except TelegramBadRequest:
        logging.debug("Could not edit style selection message after error")
    try:
        await progress.delete()
    except TelegramBadRequest:
        logging.debug("Could not delete progress message")


def build_styles_keyboard():
    builder = InlineKeyboardBuilder()
    for key, style in CARD_STYLES.items():
        builder.button(text=style.label, callback_data=f"style:{key}")
    builder.adjust(2)
    return builder.as_markup()


async def store_greeting_and_ask_style(
    message: Message,
    state: FSMContext,
    raw_text: str,
) -> None:
    text = " ".join(raw_text.split())
    if not text:
        await message.answer("Нужно прислать текст поздравления или голосовое сообщение.")
        return

    app_settings = get_settings()
    if len(text) > app_settings.max_text_length:
        await message.answer(
            "Текст слишком длинный для открытки. "
            f"Сократите его до {app_settings.max_text_length} символов."
        )
        return

    progress = await message.answer("Обрабатываю текст...")
    try:
        greeting_text, context_hint = await get_card_service().refine_greeting(text)
    except (OSError, RuntimeError, ValueError) as exc:
        logging.warning("Greeting refinement failed: %s", exc)
        greeting_text = text
        context_hint = ""

    greeting_text = _fit_greeting_text(greeting_text, app_settings.max_text_length)
    if not greeting_text:
        greeting_text = _fit_greeting_text(text, app_settings.max_text_length)

    try:
        await progress.delete()
    except TelegramBadRequest:
        logging.debug("Could not delete progress message")
    await state.set_state(GreetingFlow.waiting_for_style)
    await state.update_data(greeting_text=greeting_text, context_hint=context_hint)
    await message.answer(
        "Текст для открытки:\n"
        f"{escape(greeting_text)}\n\n"
        "Выберите стиль открытки:",
        reply_markup=build_styles_keyboard(),
    )


@router.message(CommandStart())
async def handle_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "Пришлите текст поздравления или голосовое сообщение. "
        "После этого я покажу стили и соберу открытку."
    )


@router.message(Command("help"))
async def handle_help(message: Message) -> None:
    await message.answer(
        "Как пользоваться:\n"
        "1. Отправьте текст поздравления или голосовое.\n"
        "2. Выберите стиль кнопкой.\n"
        "3. Получите готовую открытку."
    )


@router.message(Command("cancel"))
async def handle_cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Текущий сценарий сброшен. Пришлите новый текст или голосовое.")


@router.message(F.voice)
async def handle_voice(message: Message, state: FSMContext, bot: Bot) -> None:
    progress = await message.answer("Распознаю голосовое сообщение...")
    try:
        voice_buffer = await bot.download(message.voice)
        if voice_buffer is None:
            raise RuntimeError("Telegram returned no voice file.")

        greeting_text = await get_card_service().transcribe_audio(
            voice_buffer.getvalue(),
            mime_type=message.voice.mime_type or "audio/ogg",
        )
    except Exception as exc:
        logging.exception("Voice transcription failed")
        await notify_admin(bot, exc, _user_info(message.from_user))
        await progress.edit_text(
            "Не удалось распознать голосовое. Попробуйте ещё раз или пришлите текст."
        )
        return

    await progress.edit_text(f"Распознал:\n{escape(greeting_text)}")
    await store_greeting_and_ask_style(message, state, greeting_text)


@router.message(F.text)
async def handle_text(message: Message, state: FSMContext) -> None:
    await store_greeting_and_ask_style(message, state, message.text or "")


async def _send_card_photo(callback: CallbackQuery, rendered_card) -> None:
    """Send the rendered card photo with retries and fallback notification."""
    photo = BufferedInputFile(rendered_card.image_bytes, filename="greeting-card.jpg")

    for attempt in range(3):
        try:
            await callback.message.answer_photo(photo=photo)  # type: ignore[union-attr]
            break
        except TelegramNetworkError:
            if attempt == 2:
                raise
            logging.warning("Photo send failed (attempt %d), retrying...", attempt + 1)
            await asyncio.sleep(2)

    if rendered_card.used_fallback_background:
        if rendered_card.fallback_reason and callback.bot:
            fallback_exc = RuntimeError(f"Image fallback: {rendered_card.fallback_reason}")
            await notify_admin(callback.bot, fallback_exc, _user_info(callback.from_user))
        await callback.message.answer(  # type: ignore[union-attr]
            "Открытка собрана в резервном режиме. "
            "Фон локальный, но текст и результат корректные."
        )


async def _post_generation_admin_update(callback: CallbackQuery, state: FSMContext) -> None:
    """Clear state and update admin stats after successful generation."""
    await state.clear()
    if callback.bot:
        try:
            await update_stats(callback.bot)
            await check_and_warn_budget(callback.bot)
        except (TelegramBadRequest, TelegramNetworkError):
            logging.exception("Post-send admin update failed")


@router.callback_query(GreetingFlow.waiting_for_style, F.data.startswith("style:"))
async def handle_style_choice(callback: CallbackQuery, state: FSMContext) -> None:
    if callback.message is None or callback.data is None:
        await callback.answer()
        return

    style_key = callback.data.split(":", maxsplit=1)[1]
    if style_key not in CARD_STYLES:
        await callback.answer("Неизвестный стиль.", show_alert=True)
        return

    data = await state.get_data()
    greeting_text = data.get("greeting_text")
    if not greeting_text:
        await state.clear()
        await callback.answer(
            "Текст поздравления потерялся. Пришлите его заново.",
            show_alert=True,
        )
        return

    context_hint = data.get("context_hint", "")
    style_label = CARD_STYLES[style_key].label
    await callback.answer()

    try:
        await callback.message.edit_text(
            "Текст для открытки:\n"
            f"{escape(greeting_text)}\n\n"
            f"Стиль: {escape(style_label)}",
            reply_markup=None,
        )
    except TelegramBadRequest:
        logging.debug("Could not update style selection message")

    progress = await callback.message.answer(
        "Генерирую открытку. Это может занять до минуты..."
    )
    user = callback.from_user
    delete_progress = True
    try:
        rendered_card = await get_card_service().create_card(
            greeting_text, style_key, context_hint,
            user_name=user.full_name if user else "",
            user_id=user.id if user else 0,
        )
        await _send_card_photo(callback, rendered_card)
        await _post_generation_admin_update(callback, state)
    except BudgetExceededError as exc:
        logging.warning("Budget exceeded: %s", exc)
        await progress.edit_text(str(exc))
        delete_progress = False
        await state.clear()
        return
    except Exception as exc:
        logging.exception("Greeting card generation failed")
        if callback.bot:
            await notify_admin(callback.bot, exc, _user_info(callback.from_user))
        await _restore_style_buttons(callback, greeting_text, progress,
            "Не удалось собрать открытку. Выберите стиль ещё раз:")
        return
    finally:
        if delete_progress:
            try:
                await progress.delete()
            except TelegramBadRequest:
                logging.debug("Could not delete progress message in finally")


@router.callback_query(F.data.startswith("style:"))
async def handle_stale_style(callback: CallbackQuery) -> None:
    await callback.answer(
        "Сначала пришлите текст поздравления или голосовое сообщение.",
        show_alert=True,
    )


@router.message()
async def handle_unsupported_message(message: Message) -> None:
    await message.answer("Поддерживаются только текстовые и голосовые сообщения.")


async def main() -> None:
    global settings, card_service

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = Settings.from_env()
    configure_budget_file(settings.budget_file_path, settings.image_generation_cost)
    card_service = GreetingCardService(settings)

    router.message.middleware(StatsAccountFilter())
    router.callback_query.middleware(StatsAccountFilter())

    dp = Dispatcher(
        storage=JsonFileStorage(settings.fsm_storage_path),
        events_isolation=SimpleEventIsolation(),
    )
    dp.include_router(router)

    session = AiohttpSession(proxy=settings.telegram_proxy, timeout=300)
    if settings.telegram_proxy:
        logging.info("Using Telegram proxy: %s", settings.telegram_proxy.split("@")[-1])
    async with Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        session=session,
    ) as bot:
        try:
            await dp.start_polling(bot)
        finally:
            await card_service.close()


if __name__ == "__main__":
    asyncio.run(main())
