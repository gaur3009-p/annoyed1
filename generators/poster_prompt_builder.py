def build_poster_prompt(campaign, headline, copy):
    return f"""
A high-quality marketing poster background for a SaaS brand.

Context:
- Industry: {campaign['industry']}
- Brand tone: {campaign['tone']}
- Primary emotions: {', '.join(campaign['emotion'])}

Creative intent (semantic only):
- Headline meaning: {headline}
- Supporting message: {copy}

Style rules (STRICT):
- Clean
- Minimal
- Professional
- Abstract tech visuals
- Soft gradients
- Premium lighting
- High contrast
- No text
- No letters
- No words
- No typography
- No logos
- Background only
"""
