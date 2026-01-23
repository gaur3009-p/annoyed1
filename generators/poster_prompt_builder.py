def build_poster_prompt(campaign, headline, copy):
    return f"""
Abstract premium marketing poster background.

Industry: {campaign['industry']}
Brand tone: {campaign['tone']}
Emotion: {campaign['emotion'][0]}

Visual inspiration:
{headline}
{copy}

Style:
clean, minimal, premium,
abstract tech, soft gradients,
cinematic lighting, high contrast,
background only, no text
"""
