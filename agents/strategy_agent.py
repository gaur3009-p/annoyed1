def build_strategy_prompt(campaign):
    return f"""
You are a senior marketing strategist.

Brand: {campaign['brand']}
Industry: {campaign['industry']}
Audience: {campaign['audience']}
Goal: {campaign['goal']}

Return:
- Core campaign angle
- Key pain points
- Emotional hook
- Messaging direction

### STRATEGY OUTPUT:
"""
