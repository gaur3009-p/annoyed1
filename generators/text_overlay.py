from PIL import Image, ImageDraw, ImageFont
import os

# --------------------------------------------------
# 🔒 RESOLVE PROJECT ROOT SAFELY
# --------------------------------------------------

def get_project_root():
    """
    Finds the project root dynamically.
    Assumes this file lives in <root>/generators/
    """
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

PROJECT_ROOT = get_project_root()
FONT_DIR = os.path.join(PROJECT_ROOT, "fonts")

HEADLINE_FONT_PATH = os.path.join(
    FONT_DIR, "Inter-Bold-700.otf"
)

BODY_FONT_PATH = os.path.join(
    FONT_DIR, "Inter-Regular-400.otf"
)


def _load_font_safe(path, size):
    """
    Loads font safely with fallback to default PIL font.
    Never crashes the app.
    """
    try:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
        else:
            print(f"[WARN] Font not found: {path}")
    except Exception as e:
        print(f"[WARN] Font load failed: {e}")

    # Fallback (never fails)
    return ImageFont.load_default()


def overlay_text(
    image: Image.Image,
    headline: str,
    copy: str
):
    """
    Deterministic, brand-safe text overlay.
    Will NEVER crash due to font issues.
    """

    draw = ImageDraw.Draw(image)
    width, height = image.size

    headline_font = _load_font_safe(
        HEADLINE_FONT_PATH, size=54
    )

    body_font = _load_font_safe(
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
