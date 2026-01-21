from decision_engine.platform_rules import PLATFORM_RULES
from decision_engine.decision_tree import decision_score

def select_best_variant(variant_scores, platform):
    rules = PLATFORM_RULES[platform]

    scored = []
    for v in variant_scores:
        score = decision_score(
            emotion=v["emotion_score"],
            clarity=v["clarity_score"],
            visual=v["visual_score"],
            rules=rules
        )
        scored.append((v["variant"], score))

    return max(scored, key=lambda x: x[1])
