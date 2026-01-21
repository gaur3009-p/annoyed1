def build_poster_prompt(campaign):
    return f"""
A modern SaaS marketing poster background.

Industry: {campaign['industry']}
Brand tone: {campaign['tone']}

Style:
- Clean
- Minimal
- Professional
- Abstract tech visuals
- Soft gradients
- High contrast
- No text
- No letters
- No words
- No typography
- Background only
"""
