def build_variant_prompt(campaign, variant_type):
    return f"""
You are a senior ad strategist.

Variant Type: {variant_type}

Brand: {campaign['brand']}
Industry: {campaign['industry']}
Audience: {campaign['audience']}
Goal: {campaign['goal']}
Tone: {campaign['tone']}
Emotions: {', '.join(campaign['emotion'])}

Generate:
- 5 ad copies
- 5 punchy headlines
- 3 CTAs

Ensure this variant strongly reflects its strategy.

### VARIANT OUTPUT:
"""
