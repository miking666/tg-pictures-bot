from __future__ import annotations

import asyncio
import io
import logging
import random
from dataclasses import dataclass
from html import unescape
from typing import Iterable, Sequence

from google import genai
from google.genai import errors, types
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from budget import can_generate, record_fallback, record_image_generation
from config import Settings
from styles import CARD_STYLES


QUALITY_PRESETS = (
    (
        "Fine art photography with a dark, moody, cinematic feel. "
        "Shot on medium format camera with vintage lens. "
        "Deep rich shadows, warm highlights, muted color palette, "
        "dramatic chiaroscuro lighting, shallow depth of field. "
        "Painterly textures, atmospheric haze. "
        "Think Caravaggio meets Annie Leibovitz."
    ),
    (
        "Warm golden hour photography, rich amber tones, "
        "soft diffused sunlight, gentle lens flare, "
        "shallow depth of field with creamy bokeh. "
        "Intimate and tender mood, film grain texture. "
        "Think nostalgic 35mm film photography."
    ),
    (
        "Ethereal dreamy photography with soft pastel light, "
        "misty atmosphere, delicate and airy composition, "
        "overexposed highlights, gentle color grading. "
        "Shot through a prism or vintage glass. "
        "Think Tim Walker or Paolo Roversi editorial."
    ),
    (
        "Rich baroque still life aesthetic, deep jewel tones, "
        "dramatic side lighting on dark background, "
        "luxurious velvet and silk textures, gold accents. "
        "Opulent and theatrical. "
        "Think Dutch Golden Age painting brought to life as photography."
    ),
    (
        "Vibrant cinematic photography with bold saturated colors, "
        "neon accents against deep shadows, dynamic contrast, "
        "urban or modern setting with dramatic lighting. "
        "Think Wong Kar-wai or Nicolas Winding Refn color palette."
    ),
    (
        "Soft romantic photography, candlelit warmth, "
        "rose and peach tones, delicate textures like lace and petals, "
        "shallow focus with glowing highlights, "
        "intimate and personal atmosphere. "
        "Think vintage love letter aesthetic."
    ),
)

PIL_TEXT_STYLES = (
    # (fill, stroke_fill, panel_fill) — all RGBA
    {"fill": (255, 255, 255, 255), "stroke": (0, 0, 0, 180), "panel": (0, 0, 0, 80)},
    {"fill": (255, 255, 255, 255), "stroke": (60, 30, 30, 200), "panel": (40, 20, 20, 90)},
    {"fill": (255, 245, 220, 255), "stroke": (80, 40, 10, 180), "panel": (50, 25, 5, 80)},
    {"fill": (255, 255, 255, 255), "stroke": (30, 30, 80, 200), "panel": (20, 20, 60, 80)},
    {"fill": (255, 240, 200, 255), "stroke": (0, 0, 0, 200), "panel": (0, 0, 0, 100)},
    {"fill": (240, 248, 255, 255), "stroke": (40, 50, 70, 190), "panel": (20, 30, 50, 80)},
    {"fill": (255, 230, 230, 255), "stroke": (100, 20, 40, 180), "panel": (60, 10, 20, 70)},
    {"fill": (220, 255, 220, 255), "stroke": (20, 60, 20, 180), "panel": (10, 40, 10, 70)},
)

EDGE_ELEMENTS = (
    "soft natural vignette darkening at edges",
    "out-of-focus flower petals in the foreground framing the scene",
    "warm light leaks and lens flare bleeding in from corners",
    "blurred foliage and leaves at the edges creating natural framing",
    "soft golden dust particles floating in the air near edges",
    "gentle fog or mist creeping in from the borders",
    "dreamy bokeh orbs scattered around the periphery",
    "overhanging branches or vines at the top creating a canopy",
    "scattered confetti or petals drifting through the scene",
    "volumetric light rays entering from the side",
    "soft candlelight glow warming the edges",
    "morning dew drops on a glass surface in the foreground",
    "silk or velvet fabric draped at the edges of the frame",
    "snow gently falling through the scene",
    "fireflies or glowing particles floating in the atmosphere",
    "shallow depth of field with blurred objects in the foreground",
    "rain drops on glass with the scene visible behind",
    "prismatic rainbow light refractions at the edges",
    "steam or smoke wisps curling through the frame",
    "fairy lights or string lights blurred in the foreground",
)

COMPOSITION_SEEDS = (
    "close-up macro shot with shallow depth of field",
    "wide cinematic composition with negative space",
    "overhead flat lay arrangement",
    "dramatic side-lit scene with long shadows",
    "backlit with golden rim light and lens flare",
    "symmetrical centered composition",
    "diagonal dynamic composition with leading lines",
    "layered foreground and background with bokeh",
    "bird's eye view looking down",
    "low angle looking up through elements",
    "scattered organic arrangement",
    "minimalist composition with single focal point",
)

TEXT_POSITIONS = (
    "top",
    "bottom",
    "center",
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
)

@dataclass(frozen=True, slots=True)
class TextLayout:
    font: ImageFont.FreeTypeFont
    lines: list[str]
    total_height: int
    max_width: int
    line_height: int
    stroke_width: int


@dataclass(frozen=True, slots=True)
class RenderedCard:
    image_bytes: bytes
    used_fallback_background: bool
    fallback_reason: str = ""


class BudgetExceededError(Exception):
    pass


class GreetingCardService:
    GEMINI_TIMEOUT: float = 120.0
    FONT_SIZE_MAX: int = 86
    FONT_SIZE_MIN: int = 44
    FONT_SIZE_STEP: int = 2
    TEXT_WIDTH_RATIO: float = 0.65
    TEXT_HEIGHT_RATIO: float = 0.40
    MAX_TEXT_LINES: int = 6
    JPEG_QUALITY: int = 85
    FALLBACK_JPEG_QUALITY: int = 90

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = genai.Client(api_key=settings.gemini_api_key).aio

    async def close(self) -> None:
        await self._client.aclose()

    async def transcribe_audio(self, audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = await asyncio.wait_for(
                    self._client.models.generate_content(
                        model=self.settings.audio_model,
                        contents=[
                            types.Part.from_text(
                                text=(
                                    "Transcribe the speech from this audio. Return only the recognized "
                                    "text in the original language, with no commentary."
                                )
                            ),
                            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                        ],
                        config=types.GenerateContentConfig(
                            temperature=0,
                            max_output_tokens=2048,
                        ),
                    ),
                    timeout=self.GEMINI_TIMEOUT,
                )
                transcript = self._extract_text(response).strip()
                if not transcript:
                    raise RuntimeError("Gemini returned an empty transcription.")
                return transcript
            except (errors.APIError, OSError, asyncio.TimeoutError) as exc:
                last_exc = exc
                logging.warning("Transcription failed (attempt %d/3): %s", attempt + 1, exc)
        raise RuntimeError(f"Transcription failed after 3 attempts: {last_exc}") from last_exc

    async def refine_greeting(self, raw_text: str) -> tuple[str, str]:
        """Extract the greeting text and context from raw user input.

        Returns (greeting_text, context_hint) where greeting_text goes on the
        card and context_hint influences the image generation.
        """
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = await asyncio.wait_for(
                    self._client.models.generate_content(
                        model=self.settings.text_model,
                        contents=raw_text,
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                            max_output_tokens=2048,
                            system_instruction=(
                                "Ты помогаешь создавать поздравительные открытки. "
                                "Пользователь присылает сообщение — это может быть готовый текст "
                                "поздравления, просьба сделать открытку, голосовое с пожеланиями, "
                                "или просто описание того, что он хочет.\n\n"
                                "Твоя задача:\n"
                                "1. Извлечь надпись для открытки — текст до 8-12 слов. "
                                "ВАЖНО: сохрани юмор, иронию, личные шутки если они есть! "
                                "Если сообщение шуточное/ироничное — надпись тоже должна быть "
                                "шуточной, не превращай её в сухое 'С Днём Рождения'. "
                                "Примеры: 'Даня, ты наконец моешься! Горжусь!', "
                                "'С Днём Рождения, бабуля!', 'С 8 Марта!', "
                                "'Поздравляю с повышением, босс!'.\n"
                                "2. Определи ПОДРОБНЫЙ контекст для генерации изображения. "
                                "Это САМАЯ ВАЖНАЯ часть — от неё зависит, что будет нарисовано. "
                                "Если пользователь упоминает конкретную игру, фильм, аниме, "
                                "книгу, бренд, хобби — ОБЯЗАТЕЛЬНО укажи это ЯВНО с названием. "
                                "Опиши визуальную эстетику этой вселенной/темы: ключевые визуальные "
                                "элементы, цветовую палитру, стиль арта, узнаваемые образы. "
                                "Также укажи повод, кому адресовано, настроение.\n\n"
                                "Ответь СТРОГО в формате двух строк:\n"
                                "GREETING: <надпись для открытки, 8-12 слов максимум, с юмором если уместно>\n"
                                "CONTEXT: <подробное описание на английском — "
                                "название темы/вселенной + её визуальные элементы + повод>"
                            ),
                        ),
                    ),
                    timeout=self.GEMINI_TIMEOUT,
                )
                text = self._extract_text(response).strip()
                greeting = raw_text.strip()
                context = ""
                for line in text.splitlines():
                    if line.startswith("GREETING:"):
                        greeting = line[len("GREETING:"):].strip()
                    elif line.startswith("CONTEXT:"):
                        context = line[len("CONTEXT:"):].strip()
                return greeting or raw_text.strip(), context
            except (errors.APIError, OSError, asyncio.TimeoutError) as exc:
                last_exc = exc
                logging.warning("Greeting refinement failed (attempt %d/3): %s", attempt + 1, exc)
        raise RuntimeError(f"Greeting refinement failed after 3 attempts: {last_exc}") from last_exc

    async def create_card(
        self, greeting_text: str, style_key: str, context_hint: str = "",
        user_name: str = "", user_id: int = 0,
    ) -> RenderedCard:
        if style_key not in CARD_STYLES:
            raise KeyError(f"Unknown style: {style_key}")

        if not can_generate(self.settings.max_budget):
            raise BudgetExceededError(
                f"Бюджет исчерпан (${self.settings.max_budget:.2f}). "
                "Обратитесь к администратору."
            )

        text_position = random.choice(TEXT_POSITIONS)
        background_bytes, used_fallback, fallback_reason = await self._build_background(
            greeting_text,
            style_key,
            context_hint,
            text_position,
            user_name=user_name,
            user_id=user_id,
        )
        image_bytes = self._render_text_on_image(background_bytes, greeting_text, text_position)
        if used_fallback:
            record_fallback(fallback_reason)

        return RenderedCard(
            image_bytes=image_bytes,
            used_fallback_background=used_fallback,
            fallback_reason=fallback_reason,
        )

    async def _build_background(
        self,
        greeting_text: str,
        style_key: str,
        context_hint: str,
        text_position: str,
        user_name: str = "",
        user_id: int = 0,
    ) -> tuple[bytes, bool, str]:
        last_exc: Exception | None = None
        for gen_attempt in range(4):
            try:
                image_bytes = await self._generate_image_gemini(
                    greeting_text, style_key, context_hint, text_position
                )
                record_image_generation(user_name, user_id)
                return image_bytes, False, ""
            except (errors.APIError, OSError, RuntimeError, ValueError, asyncio.TimeoutError) as exc:
                last_exc = exc
                logging.warning(
                    "Image generation failed (attempt %d/4): %s",
                    gen_attempt + 1, self._describe_generation_error(exc),
                )
        reason = self._describe_generation_error(last_exc) if last_exc else "unknown"
        logging.warning("Falling back to local background for style '%s': %s", style_key, reason)
        return self._render_fallback_background(style_key, greeting_text), True, reason

    async def _generate_image_gemini(
        self,
        greeting_text: str,
        style_key: str,
        context_hint: str,
        text_position: str,
    ) -> bytes:
        style = CARD_STYLES[style_key]
        composition = random.choice(COMPOSITION_SEEDS)
        edge = random.choice(EDGE_ELEMENTS)
        quality_preset = random.choice(QUALITY_PRESETS)

        quality = (
            f"QUALITY: {quality_preset} "
            "NOT stock photography, NOT clipart, NOT flat illustration."
        )
        card_vibe = (
            f"Atmospheric detail: {edge}. "
            "This must feel like a natural part of the scene, organic and cinematic."
        )

        content_req = (
            f"\n{style.content_requirement}\n" if style.content_requirement else ""
        )

        reserved_area_instruction = (
            f"Reserve clean negative space in the {text_position} area for a later greeting overlay. "
            "That reserved area must remain visually calm, readable, and free of clutter. "
            "Do NOT render any text, letters, numbers, signs, banners, captions, logos, watermarks, "
            "or frames anywhere in the image."
        )

        if context_hint:
            prompt = (
                f"{quality}\n"
                f"{content_req}\n"
                f"Greeting occasion: {greeting_text}\n"
                f"THEME (most important): {context_hint}\n"
                f"The image MUST visually represent the theme above — "
                f"use recognizable elements, aesthetics, and color palette "
                f"from that theme.\n"
                f"Mood/lighting: {style.prompt_fragment}\n"
                f"Composition: {composition}\n"
                f"{card_vibe}\n"
                f"{reserved_area_instruction}"
            )
        else:
            prompt = (
                f"{quality}\n"
                f"{content_req}\n"
                f"Greeting occasion: {greeting_text}\n"
                f"Visual style: {style.prompt_fragment}\n"
                f"Composition: {composition}\n"
                f"{card_vibe}\n"
                f"{reserved_area_instruction}"
            )

        logging.info("Generating image with Gemini: %s", prompt[:200])

        response = await asyncio.wait_for(
            self._client.models.generate_content(
                model=self.settings.image_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    temperature=1.0,
                ),
            ),
            timeout=self.GEMINI_TIMEOUT,
        )

        # Extract image from response
        for part in self._iter_response_parts(response):
            inline_data = getattr(part, "inline_data", None)
            if inline_data and hasattr(inline_data, "data") and inline_data.data:
                mime = getattr(inline_data, "mime_type", "") or ""
                if mime.startswith("image/"):
                    logging.info("Got image from Gemini (%s, %d bytes)", mime, len(inline_data.data))
                    return inline_data.data

        raise RuntimeError("Gemini did not return an image.")

    def _render_text_on_image(
        self, image_bytes: bytes, greeting_text: str, text_position: str = "center"
    ) -> bytes:
        if not self.settings.font_path.exists():
            raise FileNotFoundError(
                f"Font file not found: {self.settings.font_path}. "
                "Put a Cyrillic-compatible .ttf into the fonts directory."
            )

        with Image.open(io.BytesIO(image_bytes)) as image:
            base = ImageOps.fit(
                ImageOps.exif_transpose(image).convert("RGBA"),
                (self.settings.image_size, self.settings.image_size),
                method=Image.Resampling.LANCZOS,
            )

        text_style = random.choice(PIL_TEXT_STYLES)

        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        layout = self._find_layout(draw, greeting_text)

        margin = max(40, self.settings.image_size // 16)
        text_origin_x, text_origin_y = self._compute_text_origin(
            base.width, base.height,
            layout.max_width, layout.total_height,
            text_position, margin,
        )

        base = self._draw_text_panel(
            base, text_origin_x, text_origin_y,
            layout.max_width, layout.total_height, text_style,
        )
        self._draw_text_lines(
            draw, layout, text_origin_x, text_origin_y,
            layout.max_width, text_style,
        )

        result = Image.alpha_composite(base, overlay).convert("RGB")
        buffer = io.BytesIO()
        result.save(buffer, format="JPEG", quality=self.JPEG_QUALITY)
        return buffer.getvalue()

    def _draw_text_panel(
        self,
        base: Image.Image,
        text_x: int,
        text_y: int,
        text_w: int,
        text_h: int,
        text_style: dict,
    ) -> Image.Image:
        panel_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        panel_draw = ImageDraw.Draw(panel_layer)
        pad_x = max(16, self.settings.image_size // 24)
        pad_y = max(10, self.settings.image_size // 32)
        panel_rect = (
            text_x - pad_x,
            text_y - pad_y,
            text_x + text_w + pad_x,
            text_y + text_h + pad_y,
        )
        panel_radius = max(12, self.settings.image_size // 48)
        panel_draw.rounded_rectangle(
            panel_rect, radius=panel_radius, fill=text_style["panel"],
        )
        panel_layer = panel_layer.filter(
            ImageFilter.GaussianBlur(radius=max(3, self.settings.image_size // 200))
        )
        return Image.alpha_composite(base, panel_layer)

    @staticmethod
    def _draw_text_lines(
        draw: ImageDraw.ImageDraw,
        layout: TextLayout,
        text_origin_x: int,
        text_origin_y: int,
        text_block_width: int,
        text_style: dict,
    ) -> None:
        y = text_origin_y
        for line in layout.lines:
            bbox = draw.textbbox(
                (0, 0), line, font=layout.font, stroke_width=layout.stroke_width,
            )
            line_width = bbox[2] - bbox[0]
            x = text_origin_x + (text_block_width - line_width) // 2
            draw.text(
                (x, y), line, font=layout.font,
                fill=text_style["fill"],
                stroke_width=layout.stroke_width,
                stroke_fill=text_style["stroke"],
            )
            y += layout.line_height

    @staticmethod
    def _compute_text_origin(
        img_w: int,
        img_h: int,
        text_w: int,
        text_h: int,
        position: str,
        margin: int,
    ) -> tuple[int, int]:
        cx = (img_w - text_w) // 2
        cy = (img_h - text_h) // 2

        positions = {
            "center": (cx, cy),
            "top": (cx, margin),
            "bottom": (cx, img_h - text_h - margin),
            "top-left": (margin, margin),
            "top-right": (img_w - text_w - margin, margin),
            "bottom-left": (margin, img_h - text_h - margin),
            "bottom-right": (img_w - text_w - margin, img_h - text_h - margin),
        }
        return positions.get(position, (cx, cy))

    def _render_fallback_background(self, style_key: str, greeting_text: str) -> bytes:
        style = CARD_STYLES[style_key]
        size = self.settings.image_size
        rng = random.Random(f"{style_key}:{greeting_text}")
        base = self._render_gradient_background(size, *style.gradient)

        soft_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        detail_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        soft_draw = ImageDraw.Draw(soft_layer, "RGBA")
        detail_draw = ImageDraw.Draw(detail_layer, "RGBA")

        glow_margin = size // 5
        soft_draw.ellipse(
            (
                glow_margin,
                glow_margin,
                size - glow_margin,
                size - glow_margin,
            ),
            fill=style.accents[0] + (64,),
        )

        match style.pattern:
            case "confetti":
                self._draw_confetti(rng, size, style.accents, soft_draw, detail_draw)
            case "petals":
                self._draw_petals(rng, size, style.accents, soft_draw, detail_draw)
            case "snow":
                self._draw_snow(rng, size, style.accents, soft_draw, detail_draw)
            case "sunset":
                self._draw_sunset(rng, size, style.accents, soft_draw, detail_draw)
            case _:
                self._draw_watercolor(rng, size, style.accents, soft_draw, detail_draw)

        soft_layer = soft_layer.filter(
            ImageFilter.GaussianBlur(radius=max(8, size // 36))
        )
        result = Image.alpha_composite(base, soft_layer)
        result = Image.alpha_composite(result, detail_layer).convert("RGB")

        buffer = io.BytesIO()
        result.save(buffer, format="JPEG", quality=self.FALLBACK_JPEG_QUALITY)
        return buffer.getvalue()

    def _find_layout(self, draw: ImageDraw.ImageDraw, greeting_text: str) -> TextLayout:
        clean_text = " ".join(unescape(greeting_text).split())
        max_width = int(self.settings.image_size * self.TEXT_WIDTH_RATIO)
        max_height = int(self.settings.image_size * self.TEXT_HEIGHT_RATIO)

        for font_size in range(self.FONT_SIZE_MAX, self.FONT_SIZE_MIN - 1, -self.FONT_SIZE_STEP):
            font = ImageFont.truetype(str(self.settings.font_path), font_size)
            stroke_width = max(2, font_size // 18)
            lines = self._wrap_text(draw, clean_text, font, stroke_width, max_width)
            line_height = self._line_height(draw, font, stroke_width)
            total_height = line_height * len(lines)
            widest = max(
                self._text_width(draw, line, font, stroke_width)
                for line in lines
            )
            if total_height <= max_height and widest <= max_width and len(lines) <= self.MAX_TEXT_LINES:
                return TextLayout(
                    font=font,
                    lines=lines,
                    total_height=total_height,
                    max_width=widest,
                    line_height=line_height,
                    stroke_width=stroke_width,
                )

        fallback_font = ImageFont.truetype(str(self.settings.font_path), self.FONT_SIZE_MIN)
        fallback_stroke = 2
        fallback_lines = self._wrap_text(
            draw,
            clean_text,
            fallback_font,
            fallback_stroke,
            max_width,
        )
        fallback_height = self._line_height(draw, fallback_font, fallback_stroke)
        fallback_width = max(
            self._text_width(draw, line, fallback_font, fallback_stroke)
            for line in fallback_lines
        )
        return TextLayout(
            font=fallback_font,
            lines=fallback_lines[:self.MAX_TEXT_LINES],
            total_height=fallback_height * min(len(fallback_lines), self.MAX_TEXT_LINES),
            max_width=fallback_width,
            line_height=fallback_height,
            stroke_width=fallback_stroke,
        )

    def _wrap_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        stroke_width: int,
        max_width: int,
    ) -> list[str]:
        words = text.split()
        if not words:
            return [text]

        lines: list[str] = []
        current_line = ""
        for word in words:
            candidate = word if not current_line else f"{current_line} {word}"
            if self._text_width(draw, candidate, font, stroke_width) <= max_width:
                current_line = candidate
                continue

            if current_line:
                lines.append(current_line)
                current_line = ""

            if self._text_width(draw, word, font, stroke_width) <= max_width:
                current_line = word
            else:
                split_word_lines = self._split_long_word(
                    draw,
                    word,
                    font,
                    stroke_width,
                    max_width,
                )
                lines.extend(split_word_lines[:-1])
                current_line = split_word_lines[-1]

        if current_line:
            lines.append(current_line)
        return lines

    def _split_long_word(
        self,
        draw: ImageDraw.ImageDraw,
        word: str,
        font: ImageFont.FreeTypeFont,
        stroke_width: int,
        max_width: int,
    ) -> list[str]:
        chunks: list[str] = []
        current = ""
        for char in word:
            candidate = f"{current}{char}"
            if current and self._text_width(draw, candidate, font, stroke_width) > max_width:
                chunks.append(current)
                current = char
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks or [word]

    @staticmethod
    def _render_gradient_background(
        size: int,
        start_color: tuple[int, int, int],
        end_color: tuple[int, int, int],
    ) -> Image.Image:
        top = Image.new("RGBA", (size, size), start_color + (255,))
        bottom = Image.new("RGBA", (size, size), end_color + (255,))
        mask = Image.linear_gradient("L").resize((size, size))
        return Image.composite(bottom, top, mask)

    @staticmethod
    def _sample_decor_point(
        rng: random.Random,
        size: int,
        bleed: int,
    ) -> tuple[int, int]:
        center_left = int(size * 0.24)
        center_top = int(size * 0.24)
        center_right = int(size * 0.76)
        center_bottom = int(size * 0.76)

        for _ in range(64):
            x = rng.randint(-bleed, size + bleed)
            y = rng.randint(-bleed, size + bleed)
            if not (center_left <= x <= center_right and center_top <= y <= center_bottom):
                return x, y
        return -bleed, -bleed

    @staticmethod
    def _draw_confetti(
        rng: random.Random,
        size: int,
        accents: Sequence[tuple[int, int, int]],
        soft_draw: ImageDraw.ImageDraw,
        detail_draw: ImageDraw.ImageDraw,
    ) -> None:
        for _ in range(28):
            x, y = GreetingCardService._sample_decor_point(rng, size, bleed=size // 10)
            radius = rng.randint(size // 22, size // 11)
            color = rng.choice(accents)
            soft_draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color + (56,),
            )

        for _ in range(150):
            x, y = GreetingCardService._sample_decor_point(rng, size, bleed=size // 14)
            width = rng.randint(size // 65, size // 28)
            height = rng.randint(size // 130, size // 70)
            color = rng.choice(accents)
            alpha = rng.randint(156, 232)
            if rng.random() < 0.55:
                detail_draw.ellipse(
                    (x, y, x + width, y + width),
                    fill=color + (alpha,),
                )
            else:
                detail_draw.rounded_rectangle(
                    (x, y, x + width * 2, y + height * 2),
                    radius=max(2, height // 2),
                    fill=color + (alpha,),
                )

    @staticmethod
    def _draw_petals(
        rng: random.Random,
        size: int,
        accents: Sequence[tuple[int, int, int]],
        soft_draw: ImageDraw.ImageDraw,
        detail_draw: ImageDraw.ImageDraw,
    ) -> None:
        for _ in range(18):
            x, y = GreetingCardService._sample_decor_point(rng, size, bleed=size // 8)
            width = rng.randint(size // 8, size // 5)
            height = rng.randint(size // 18, size // 10)
            color = rng.choice(accents)
            soft_draw.ellipse(
                (x - width, y - height, x + width, y + height),
                fill=color + (72,),
            )

        for _ in range(16):
            x, y = GreetingCardService._sample_decor_point(rng, size, bleed=size // 8)
            width = rng.randint(size // 18, size // 9)
            height = rng.randint(size // 28, size // 16)
            color = rng.choice(accents)
            detail_draw.ellipse(
                (x - width, y - height, x + width, y + height),
                fill=color + (148,),
                outline=(255, 255, 255, 90),
                width=max(1, size // 320),
            )

    @staticmethod
    def _draw_snow(
        rng: random.Random,
        size: int,
        accents: Sequence[tuple[int, int, int]],
        soft_draw: ImageDraw.ImageDraw,
        detail_draw: ImageDraw.ImageDraw,
    ) -> None:
        for _ in range(36):
            x, y = GreetingCardService._sample_decor_point(rng, size, bleed=size // 10)
            radius = rng.randint(size // 28, size // 14)
            color = rng.choice(accents)
            soft_draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color + (50,),
            )

        for _ in range(180):
            x = rng.randint(0, size)
            y = rng.randint(0, size)
            radius = rng.randint(max(1, size // 320), max(2, size // 170))
            detail_draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(255, 255, 255, rng.randint(140, 220)),
            )

    @staticmethod
    def _draw_watercolor(
        rng: random.Random,
        size: int,
        accents: Sequence[tuple[int, int, int]],
        soft_draw: ImageDraw.ImageDraw,
        detail_draw: ImageDraw.ImageDraw,
    ) -> None:
        del detail_draw
        for _ in range(20):
            x, y = GreetingCardService._sample_decor_point(rng, size, bleed=size // 9)
            radius_x = rng.randint(size // 10, size // 4)
            radius_y = rng.randint(size // 12, size // 5)
            color = rng.choice(accents)
            soft_draw.ellipse(
                (x - radius_x, y - radius_y, x + radius_x, y + radius_y),
                fill=color + (60,),
            )

    @staticmethod
    def _draw_sunset(
        rng: random.Random,
        size: int,
        accents: Sequence[tuple[int, int, int]],
        soft_draw: ImageDraw.ImageDraw,
        detail_draw: ImageDraw.ImageDraw,
    ) -> None:
        sun_radius = size // 7
        sun_x = size // 2 + rng.randint(-(size // 16), size // 16)
        sun_y = size // 3
        soft_draw.ellipse(
            (
                sun_x - sun_radius,
                sun_y - sun_radius,
                sun_x + sun_radius,
                sun_y + sun_radius,
            ),
            fill=accents[0] + (110,),
        )

        for _ in range(9):
            x = rng.randint(-size // 10, size + size // 10)
            y = rng.randint(size // 12, size // 2)
            width = rng.randint(size // 7, size // 3)
            height = rng.randint(size // 22, size // 10)
            color = rng.choice(accents[1:])
            soft_draw.ellipse(
                (x - width, y - height, x + width, y + height),
                fill=color + (48,),
            )

        horizon = int(size * 0.82)
        detail_draw.polygon(
            [
                (0, size),
                (0, horizon),
                (size * 0.18, horizon - size * 0.04),
                (size * 0.35, horizon - size * 0.015),
                (size * 0.56, horizon - size * 0.06),
                (size * 0.74, horizon - size * 0.02),
                (size, horizon - size * 0.05),
                (size, size),
            ],
            fill=(38, 24, 59, 210),
        )

    @staticmethod
    def _describe_generation_error(exc: Exception) -> str:
        if isinstance(exc, errors.APIError):
            message = exc.message or str(exc)
            return f"{exc.code} {exc.status}: {message}"
        return f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _text_width(
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        stroke_width: int,
    ) -> int:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        return bbox[2] - bbox[0]

    @staticmethod
    def _line_height(
        draw: ImageDraw.ImageDraw,
        font: ImageFont.FreeTypeFont,
        stroke_width: int,
    ) -> int:
        bbox = draw.textbbox((0, 0), "Ау", font=font, stroke_width=stroke_width)
        return int((bbox[3] - bbox[1]) * 1.2)

    @staticmethod
    def _iter_response_parts(response: object) -> Iterable[object]:
        direct_parts = getattr(response, "parts", None)
        if direct_parts:
            return direct_parts

        collected_parts: list[object] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            if content and getattr(content, "parts", None):
                collected_parts.extend(content.parts)
        return collected_parts

    @staticmethod
    def _extract_text(response: object) -> str:
        direct_text = getattr(response, "text", None)
        if direct_text:
            return direct_text

        fragments: list[str] = []
        for part in GreetingCardService._iter_response_parts(response):
            text = getattr(part, "text", None)
            if text:
                fragments.append(text)
        return "\n".join(fragments)
