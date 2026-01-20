def infer_emotion(objective):
    mapping = {
        "Awareness": ["Curiosity", "Aspiration"],
        "Leads": ["Trust", "Urgency"],
        "Sales": ["Desire", "Urgency"]
    }
    return mapping.get(objective, ["Clarity"])


def structure_campaign(
    brand_name,
    brand_description,
    industry,
    target_audience,
    objective,
    tone,
    platforms
):
    return {
        "brand": brand_name,
        "brand_description": brand_description,
        "industry": industry,
        "audience": target_audience,
        "goal": objective,
        "tone": tone,
        "emotion": infer_emotion(objective),
        "platforms": platforms
    }
