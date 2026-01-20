def infer_emotion(objective: str):
    mapping = {
        "Leads": ["Trust", "Urgency"],
        "Sales": ["Desire", "Urgency"],
        "Awareness": ["Curiosity", "Aspiration"]
    }
    return mapping.get(objective, ["Clarity"])


def structure_campaign(
    brand_name,
    industry,
    target_audience,
    objective,
    tone,
    platforms
):
    return {
        "brand": brand_name,
        "industry": industry,
        "audience": target_audience,
        "goal": objective,
        "emotion": infer_emotion(objective),
        "tone": tone,
        "platforms": platforms
    }
