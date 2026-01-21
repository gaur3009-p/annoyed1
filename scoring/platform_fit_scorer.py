def score_platform_fit(platform, variant_type):
    if platform == "Meta" and variant_type in ["Emotional", "Bold"]:
        return 0.9
    if platform == "Google" and variant_type == "Trust":
        return 0.9
    if platform == "LinkedIn" and variant_type == "Trust":
        return 0.95
    return 0.6
