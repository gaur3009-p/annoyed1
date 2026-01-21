def build_poster_prompt(campaign, headline, copy):
    return f"""
Create a high-conversion advertising poster.

Brand: {campaign['brand']}
Industry: {campaign['industry']}
Tone: {campaign['tone']}

Headline:
{headline}

Supporting Copy:
{copy}

Visual Style:
- Clean
- Modern
- High contrast
- Platform-agnostic

No text overflow. Center-aligned.
"""
