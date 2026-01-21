def decision_score(emotion, clarity, visual, rules):
    return (
        emotion * rules["emotion_weight"]
        + clarity * rules["clarity_weight"]
        + visual * rules["visual_weight"]
    )
