def build_prompt(campaign):
    return f"""
You are a senior creative strategist working at a top global ad agency.

Brand: {campaign['brand']}
Industry: {campaign['industry']}
Target Audience: {campaign['audience']}
Business Goal: {campaign['goal']}
Brand Tone: {campaign['tone']}
Primary Emotions to Trigger: {', '.join(campaign['emotion'])}

TASK:
1. Create 5 high-conversion ad copies (2–3 lines each)
2. Create 5 short punchy headlines
3. Create 3 strong CTA options

RULES:
- Be brand-safe
- Avoid exaggerated claims
- Write like a human creative director
"""
