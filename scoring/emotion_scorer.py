from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

EMOTION_LABELS = {
    "Trust": "This message builds trust and credibility",
    "Urgency": "This message creates urgency to act now",
    "Aspiration": "This message inspires growth and ambition",
    "Curiosity": "This message sparks curiosity and interest"
}

def score_emotions(text):
    text_emb = model.encode(text)

    scores = {}
    for emotion, desc in EMOTION_LABELS.items():
        e_emb = model.encode(desc)
        score = np.dot(text_emb, e_emb) / (
            np.linalg.norm(text_emb) * np.linalg.norm(e_emb)
        )
        scores[emotion] = round(float(score), 3)

    return scores
