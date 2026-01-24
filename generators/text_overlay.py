from PIL import Image, ImageDraw, ImageFont
import os

FONT_DIR = "fonts"

HEADLINE_FONT_PATH = os.path.join(
    FONT_DIR, "Inter-Bold-700.otf"
)

BODY_FONT_PATH = os.path.join(
    FONT_DIR, "Inter-Regular-400.otf"
)


def overlay_text(
    image: Image.Image,
    headline: str,
    copy: str
):
    """
    Adds deterministic, brand-safe text overlay.
    No AI involved here.
    """

    draw = ImageDraw.Draw(image)
    width, height = image.size

    headline_font = ImageFont.truetype(
        HEADLINE_FONT_PATH, size=54
    )

    body_font = ImageFont.truetype(
        BODY_FONT_PATH, size=28
    )

    # ---- Headline shadow ----
    draw.text(
        (width * 0.08 + 2, height * 0.62 + 2),
        headline,
        fill=(0, 0, 0),
        font=headline_font
    )

    # ---- Headline ----
    draw.text(
        (width * 0.08, height * 0.62),
        headline,
        fill=(255, 255, 255),
        font=headline_font
    )

    # ---- Copy ----
    draw.text(
        (width * 0.08, height * 0.72),
        copy,
        fill=(230, 230, 230),
        font=body_font
    )
    return image
