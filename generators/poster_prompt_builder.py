def build_poster_prompt(campaign):
    """
    Builds a STRICT background-only prompt.
    No text, no typography, no marketing language.
    """

    return f"""
Purely visual abstract brand background.

Industry: {campaign['industry']}
Brand tone: {campaign['tone']}
Primary emotion: {campaign['emotion'][0]}

Visual style:
organic flowing shapes,
soft gradients,
premium cinematic lighting,
modern beauty-tech aesthetic,
depth and subtle motion

STRICT RULES:
no text,
no letters,
no words,
no typography,
no logos,
no symbols,
background only
"""
