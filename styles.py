from __future__ import annotations

from dataclasses import dataclass


RGBColor = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class CardStyle:
    label: str
    prompt_fragment: str
    gradient: tuple[RGBColor, RGBColor]
    accents: tuple[RGBColor, ...]
    pattern: str
    content_requirement: str = ""


CARD_STYLES: dict[str, CardStyle] = {
    "flowers": CardStyle(
        label="🌸 Цветы",
        prompt_fragment=(
            "Elegant floral still life, lush blooms with dewy petals, "
            "draped silk or velvet fabric, soft diffused natural light, "
            "muted earthy tones, warm shadows, medium format photography, "
            "cinematic depth of field"
        ),
        gradient=((93, 48, 85), (232, 172, 186)),
        accents=((255, 223, 231), (214, 116, 149), (255, 241, 214)),
        pattern="petals",
    ),
    "portrait": CardStyle(
        label="👤 Портрет",
        prompt_fragment=(
            "Artistic portrait, shot on 85mm lens, cinematic lighting "
            "with soft rim light, shallow depth of field, rich atmospheric scene, "
            "high-end fashion or editorial photography, "
            "real skin textures, natural expression"
        ),
        gradient=((58, 36, 61), (186, 139, 121)),
        accents=((232, 196, 174), (186, 139, 151), (255, 224, 198)),
        pattern="watercolor",
        content_requirement=(
            "The image MUST prominently feature a PERSON (human figure) "
            "as the main subject, surrounded by thematic elements. "
            "The person should match the context (age, gender if implied). "
            "This is a PORTRAIT — without a person the image is wrong."
        ),
    ),
    "animals": CardStyle(
        label="🐾 Животные",
        prompt_fragment=(
            "Adorable animal in a festive celebratory setting, "
            "warm golden lighting, heartwarming cozy atmosphere, "
            "professional pet photography style with bokeh background"
        ),
        gradient=((255, 213, 168), (180, 140, 120)),
        accents=((255, 236, 210), (220, 170, 140), (255, 255, 255), (255, 190, 150)),
        pattern="confetti",
        content_requirement=(
            "The image MUST prominently feature a CUTE ANIMAL (cat, dog, bunny, "
            "or other pet) as the main subject. The animal should look adorable "
            "and festive. Without an animal the image is wrong."
        ),
    ),
    "fantasy": CardStyle(
        label="✨ Фэнтези",
        prompt_fragment=(
            "Magical fantasy scene with glowing particles, enchanted forest or "
            "mystical landscape, ethereal light rays, floating luminous elements, "
            "rich jewel tones, fairy tale atmosphere, digital art style "
            "with cinematic volumetric lighting"
        ),
        gradient=((22, 15, 60), (85, 40, 130)),
        accents=((180, 130, 255), (100, 220, 255), (255, 200, 100), (255, 255, 255)),
        pattern="watercolor",
    ),
    "festive": CardStyle(
        label="🎉 Праздничный",
        prompt_fragment=(
            "Festive celebration scene, wrapped gifts with satin ribbons, "
            "birthday cake with candles, golden confetti, champagne glasses, "
            "warm candlelight, cozy intimate atmosphere, soft bokeh fairy lights"
        ),
        gradient=((85, 34, 112), (255, 126, 95)),
        accents=((255, 214, 10), (255, 78, 109), (66, 214, 200), (255, 255, 255)),
        pattern="confetti",
    ),
    "landscape": CardStyle(
        label="🏞️ Пейзаж",
        prompt_fragment=(
            "Breathtaking scenic landscape, golden hour lighting, "
            "mountains or sea or rolling hills, dramatic sky with clouds, "
            "rich natural colors, sense of vastness and beauty, "
            "cinematic wide angle photography, National Geographic style"
        ),
        gradient=((40, 70, 100), (200, 160, 100)),
        accents=((255, 200, 100), (100, 180, 220), (255, 255, 255), (180, 210, 140)),
        pattern="watercolor",
    ),
    "illustration": CardStyle(
        label="🎨 Иллюстрация",
        prompt_fragment=(
            "Stylized colorful illustration, vibrant cartoon or anime-inspired art, "
            "bold outlines, expressive characters or objects, playful composition, "
            "bright saturated palette, digital illustration style, "
            "Pixar or Studio Ghibli inspired warmth"
        ),
        gradient=((255, 116, 82), (48, 213, 200)),
        accents=((255, 232, 31), (255, 78, 109), (58, 134, 255), (255, 255, 255)),
        pattern="confetti",
    ),
    "cozy": CardStyle(
        label="☕ Уютный",
        prompt_fragment=(
            "Cozy hygge scene, warm knit blankets, steaming cup of cocoa or tea, "
            "soft candlelight, old books, autumn leaves or cinnamon sticks, "
            "warm amber tones, intimate comfortable atmosphere, "
            "lifestyle photography with shallow depth of field"
        ),
        gradient=((80, 50, 30), (200, 150, 100)),
        accents=((255, 210, 160), (200, 130, 80), (255, 240, 220), (180, 120, 70)),
        pattern="watercolor",
    ),
}
