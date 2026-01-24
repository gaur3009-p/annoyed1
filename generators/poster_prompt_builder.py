def build_poster_prompt(campaign):
    return f"""
Purely visual abstract brand background.

Industry: {campaign['industry']}
Brand tone: {campaign['tone']}
Emotion: {campaign['emotion'][0]}

Visual style:
organic shapes,
soft gradients,
premium lighting,
cinematic depth,
modern beauty aesthetic

STRICT:
no text,
no letters,
no words,
no typography,
no logos,
background only
"""
