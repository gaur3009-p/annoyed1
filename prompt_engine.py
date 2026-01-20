def build_prompt(campaign):
    return f"""
You are a senior creative strategist at a top global advertising agency.

Brand Name: {campaign['brand']}
Brand Description: {campaign['brand_description']}
Industry: {campaign['industry']}
Target Audience: {campaign['audience']}
Business Goal: {campaign['goal']}
Brand Tone: {campaign['tone']}
Primary Emotions to Trigger: {', '.join(campaign['emotion'])}

TASK:
1. Write 5 high-quality ad copies (2–3 lines each)
2. Write 5 short punchy headlines
3. Write 3 CTA options

RULES:
- Be realistic and brand-safe
- Avoid exaggerated claims
- Sound human, not AI
"""
