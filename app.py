import gradio as gr

# =========================
# Core Imports
# =========================
from campaign_structurer import structure_campaign
from prompt_engine import build_prompt
from llm_client import generate_text
from memory_store import save_campaign
from scoring.emotion_scorer import score_emotions

# =========================
# Phase 2 Imports
# =========================
from agents.variant_agent import build_variant_prompt
from scoring.copy_quality_scorer import score_copy_quality
from scoring.platform_fit_scorer import score_platform_fit
from generators.poster_prompt_builder import build_poster_prompt
from generators.image_generator import generate_poster
from decision_engine.variant_selector import select_best_variant


# ======================================================
# 🧠 Emotion Ranking (Phase 1.5)
# ======================================================
def rank_output(output: str):
    scores = score_emotions(output)
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "emotion_scores": scores,
        "dominant_emotion": ranking[0][0]
    }


# ======================================================
# 🧩 Headline & Copy Extraction
# ======================================================
def extract_headlines_and_copies(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    headlines = []
    copies = []

    for line in lines:
        if len(line) <= 120:
            headlines.append(line)
        else:
            copies.append(line)

    return headlines, copies


def select_best_headline(headlines):
    scored = []
    for h in headlines:
        emotion = max(score_emotions(h).values())
        clarity = score_copy_quality(h)
        scored.append((h, emotion + clarity))
    return max(scored, key=lambda x: x[1])[0]


def select_best_copy(copies):
    scored = []
    for c in copies:
        emotion = max(score_emotions(c).values())
        clarity = score_copy_quality(c)
        scored.append((c, emotion + clarity))
    return max(scored, key=lambda x: x[1])[0]


# ======================================================
# 🔁 Revision Logic
# ======================================================
def revise_campaign(original_output, feedback):
    prompt = f"""
You are revising a marketing campaign.

Original Campaign:
{original_output}

Client Feedback:
{feedback}

Return only the improved campaign.

### REVISED OUTPUT:
"""
    raw = generate_text(prompt)
    return raw.split("### REVISED OUTPUT")[-1].strip()


# ======================================================
# 🚀 Phase 1.5 — Single Campaign
# ======================================================
def generate_campaign(
    brand_name,
    brand_description,
    industry,
    target_audience,
    objective,
    tone,
    platforms
):
    campaign = structure_campaign(
        brand_name,
        brand_description,
        industry,
        target_audience,
        objective,
        tone,
        platforms=[p.strip() for p in platforms.split(",")]
    )

    raw_output = generate_text(build_prompt(campaign))
    output = raw_output.split("### OUTPUT")[-1].strip()

    save_campaign(campaign, output)

    emotion_result = rank_output(output)
    emotion_summary = "\n".join(
        [f"{e}: {s}" for e, s in emotion_result["emotion_scores"].items()]
    )

    return output, f"Dominant Emotion: {emotion_result['dominant_emotion']}\n\n{emotion_summary}"


# ======================================================
# 🚀 Phase 2 — Multi-Variant + Posters
# ======================================================
def run_phase2(
    brand_name,
    brand_description,
    industry,
    target_audience,
    objective,
    tone,
    platforms,
    decision_platform
):
    campaign = structure_campaign(
        brand_name,
        brand_description,
        industry,
        target_audience,
        objective,
        tone,
        platforms.split(",")
    )

    # -------- Variants --------
    variants = {}
    for vtype in ["Emotional", "Trust", "Bold"]:
        raw = generate_text(build_variant_prompt(campaign, vtype))
        variants[vtype] = raw.split("### VARIANT OUTPUT")[-1].strip()

    # -------- Scoring --------
    scored_variants = []
    for vtype, text in variants.items():
        scored_variants.append({
            "variant": vtype,
            "text": text,
            "emotion_score": max(score_emotions(text).values()),
            "clarity_score": score_copy_quality(text),
            "visual_score": score_platform_fit(decision_platform, vtype)
        })

    best_variant, final_score = select_best_variant(
        scored_variants, decision_platform
    )

    best_text = next(
        v["text"] for v in scored_variants if v["variant"] == best_variant
    )

    # -------- BEST HEADLINE & COPY --------
    headlines, copies = extract_headlines_and_copies(best_text)
    best_headline = select_best_headline(headlines)
    best_copy = select_best_copy(copies)

    # -------- Posters --------
    posters = []
    for _ in range(3):
        prompt = build_poster_prompt(
            campaign,
            best_headline,
            best_copy
        )
        posters.append(generate_poster(prompt))

    # -------- UI --------
    copy_block = ""
    for v in scored_variants:
        copy_block += f"\n\n=== {v['variant']} VARIANT ===\n{v['text']}"
        copy_block += (
            f"\nEmotion: {v['emotion_score']}"
            f"\nClarity: {v['clarity_score']}"
            f"\nPlatform Fit: {v['visual_score']}\n"
        )

    decision_summary = (
        f"🏆 Best Variant for {decision_platform}: {best_variant}\n"
        f"Final Score: {round(final_score, 3)}\n\n"
        f"🎯 Selected Headline:\n{best_headline}\n\n"
        f"📝 Selected Copy:\n{best_copy}"
    )

    return copy_block, decision_summary, posters


# ======================================================
# 🎨 GRADIO UI
# ======================================================
with gr.Blocks(title="Rookus – Creative Campaign Studio") as demo:
    gr.Markdown("## 🚀 Rookus – AI-Powered Creative Campaign Studio")

    brand_name = gr.Textbox(label="Brand Name")
    brand_description = gr.Textbox(label="Brand Description", lines=3)
    industry = gr.Textbox(label="Industry")
    target_audience = gr.Textbox(label="Target Audience")

    objective = gr.Dropdown(
        ["Awareness", "Leads", "Sales"],
        label="Campaign Objective"
    )

    tone = gr.Dropdown(
        ["Premium", "Friendly", "Bold", "Trustworthy", "Aggressive"],
        label="Brand Tone"
    )

    platforms = gr.Textbox(
        label="Platforms",
        placeholder="Meta, Google, LinkedIn"
    )

    generate_btn = gr.Button("Generate Campaign")
    output = gr.Textbox(lines=20)
    emotion_output = gr.Textbox(lines=6)

    generate_btn.click(
        generate_campaign,
        [
            brand_name,
            brand_description,
            industry,
            target_audience,
            objective,
            tone,
            platforms
        ],
        [output, emotion_output]
    )

    decision_platform = gr.Dropdown(
        ["Meta", "Google", "LinkedIn"],
        label="Optimize For Platform"
    )

    phase2_btn = gr.Button("Run Phase 2")
    phase2_output = gr.Textbox(lines=22)
    decision_output = gr.Textbox(lines=8)
    poster_gallery = gr.Gallery(columns=3)

    phase2_btn.click(
        run_phase2,
        [
            brand_name,
            brand_description,
            industry,
            target_audience,
            objective,
            tone,
            platforms,
            decision_platform
        ],
        [phase2_output, decision_output, poster_gallery]
    )
demo.launch(share=True)
