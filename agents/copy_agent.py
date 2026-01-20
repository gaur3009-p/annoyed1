def build_copy_prompt(strategy, campaign):
    return f"""
You are a world-class advertising copywriter.

Campaign Strategy:
{strategy}

Brand Tone: {campaign['tone']}
Emotions: {', '.join(campaign['emotion'])}

Write:
- 5 ad copies
- 5 headlines
- 3 CTAs

### COPY OUTPUT:
"""
