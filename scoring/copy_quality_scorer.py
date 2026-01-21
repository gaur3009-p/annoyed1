def score_copy_quality(text):
    score = 0.0

    if len(text) < 900:
        score += 0.4
    if any(cta in text.lower() for cta in ["get", "start", "join", "try"]):
        score += 0.3
    if text.count("\n") >= 5:
        score += 0.3

    return round(score, 3)
