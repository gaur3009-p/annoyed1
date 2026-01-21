import gradio as gr

# =========================
# Core Imports (Existing)
# =========================
from campaign_structurer import structure_campaign
from prompt_engine import build_prompt
from llm_client import generate_text
from memory_store import save_campaign
from scoring.emotion_scorer import score_emotions

# =========================
# Phase 2 Imports (NEW)
# =========================
from agents.variant_agent import build_variant_prompt
from scoring.copy_quality_scorer import score_copy_quality
from scoring.platform_fit_scorer import score_platform_fit
from generators.poster_prompt_builder import build_poster_prompt
from generators.image_generator import generate_poster
from decision_engine.variant_selector import select_best_variant


# ======================================================
# 🧠 Emotion Ranking Logic (UNCHANGED – Phase 1.5)
# ======================================================
def rank_output(output: str):
    scores = score_emotions(output)
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "emotion_scores": scores,
        "dominant_emotion": ranking[0][0]
    }


# ======================================================
# 🔁 Revision Logic (UNCHANGED)
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
# 🚀 PHASE 1.5 — Single Campaign Generation (UNCHANGED)
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
        brand_name=brand_name,
        brand_description=brand_description,
        industry=industry,
        target_audience=target_audience,
        objective=objective,
        tone=tone,
        platforms=[p.strip() for p in platforms.split(",")]
    )

    prompt = build_prompt(campaign)
    raw_output = generate_text(prompt)

    output = (
        raw_output.split("### OUTPUT")[-1].strip()
        if "### OUTPUT" in raw_output
        else raw_output.strip()
    )

    save_campaign(campaign, output)

    emotion_result = rank_output(output)
    emotion_scores_text = "\n".join(
        [f"{e}: {s}" for e, s in emotion_result["emotion_scores"].items()]
    )

    emotion_summary = (
        f"Dominant Emotion: {emotion_result['dominant_emotion']}\n\n"
        f"Emotion Scores:\n{emotion_scores_text}"
    )

    return output, emotion_summary


# ======================================================
# 🚀 PHASE 2 — CONSTRUCTIVE MULTI-VARIANT SYSTEM (NEW)
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
        [p.strip() for p in platforms.split(",")]
    )

    # ---------- LEVEL 2: Variants ----------
    variants = {}
    for vtype in ["Emotional", "Trust", "Bold"]:
        prompt = build_variant_prompt(campaign, vtype)
        raw = generate_text(prompt)
        variants[vtype] = raw.split("### VARIANT OUTPUT")[-1].strip()

    # ---------- Scoring ----------
    scored_variants = []
    for vtype, text in variants.items():
        emotion_score = max(score_emotions(text).values())
        clarity_score = score_copy_quality(text)
        platform_score = score_platform_fit(decision_platform, vtype)

        scored_variants.append({
            "variant": vtype,
            "text": text,
            "emotion_score": emotion_score,
            "clarity_score": clarity_score,
            "visual_score": platform_score
        })

    # ---------- Decision ----------
    best_variant, final_score = select_best_variant(
        scored_variants, decision_platform
    )

    best_text = next(
        v["text"] for v in scored_variants if v["variant"] == best_variant
    )

    # ---------- LEVEL 3: Posters ----------
    lines = [l for l in best_text.split("\n") if l.strip()]
    headline = lines[0][:120]
    copy = "\n".join(lines[1:4])

    posters = []
    for _ in range(3):
        poster_prompt = build_poster_prompt(campaign, headline, copy)
        posters.append(generate_poster(poster_prompt))

    # ---------- UI Formatting ----------
    copy_block = ""
    for v in scored_variants:
        copy_block += f"\n\n=== {v['variant']} VARIANT ===\n"
        copy_block += v["text"]
        copy_block += (
            f"\n\nEmotion: {v['emotion_score']}"
            f"\nClarity: {v['clarity_score']}"
            f"\nPlatform Fit: {v['visual_score']}\n"
        )

    decision_summary = (
        f"🏆 Best Variant for {decision_platform}: {best_variant}\n"
        f"Final Score: {round(final_score, 3)}"
    )

    return copy_block, decision_summary, posters


# ======================================================
# 🎨 GRADIO UI (PHASE 1.5 + PHASE 2)
# ======================================================
with gr.Blocks(title="Rookus – Creative Campaign Studio") as demo:
    gr.Markdown(
        """
        ## 🚀 Rookus – AI-Powered Creative Campaign Studio  
        """
    )

    # ---------------- LEVEL 1 INPUTS ----------------
    with gr.Row():
        brand_name = gr.Textbox(label="Brand Name")
        industry = gr.Textbox(label="Industry")

    brand_description = gr.Textbox(label="Brand Description", lines=3)
    target_audience = gr.Textbox(label="Target Audience")

    with gr.Row():
        objective = gr.Dropdown(
            ["Awareness", "Leads", "Sales"],
            label="Campaign Objective",
            value="Awareness"
        )
        tone = gr.Dropdown(
            ["Premium", "Friendly", "Bold", "Trustworthy", "Aggressive"],
            label="Brand Tone",
            value="Premium"
        )

    platforms = gr.Textbox(
        label="Platforms (comma-separated)",
        placeholder="Meta, Google, LinkedIn"
    )

    # ---------------- PHASE 1.5 ----------------
    gr.Markdown("### 🧪 Phase 1.5 — Single Campaign")

    generate_btn = gr.Button("🚀 Generate Campaign")

    output = gr.Textbox(label="Campaign Output", lines=20)
    emotion_output = gr.Textbox(label="Emotion Analysis", lines=6)

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

    feedback = gr.Textbox(label="Client Feedback")
    revise_btn = gr.Button("🔁 Revise Campaign")
    revised_output = gr.Textbox(label="Revised Output", lines=18)

    revise_btn.click(revise_campaign, [output, feedback], revised_output)

    # ---------------- PHASE 2 ----------------
    gr.Markdown("### 🧠 Phase 2 — Multi-Variant + Posters")

    decision_platform = gr.Dropdown(
        ["Meta", "Google", "LinkedIn"],
        label="Optimize For Platform",
        value="Meta"
    )

    phase2_btn = gr.Button("🚀 Run Phase 2")

    phase2_output = gr.Textbox(label="Variants & Scores", lines=22)
    decision_output = gr.Textbox(label="Decision Engine Result", lines=3)
    poster_gallery = gr.Gallery(label="Generated Posters", columns=3)

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

# 🚀 Launch
demo.launch(share=True)
