import gradio as gr

from campaign_structurer import structure_campaign
from prompt_engine import build_prompt
from llm_client import generate_text
from memory_store import save_campaign
from scoring.emotion_scorer import score_emotions


# =========================
# 🧠 Emotion Ranking Logic
# =========================
def rank_output(output: str):
    scores = score_emotions(output)
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "emotion_scores": scores,
        "dominant_emotion": ranking[0][0]
    }


# =========================
# 🔁 Revision Logic (UNCHANGED)
# =========================
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


# =========================
# 🚀 Main Generation Logic
# =========================
def generate_campaign(
    brand_name,
    brand_description,
    industry,
    target_audience,
    objective,
    tone,
    platforms
):
    # 1️⃣ Structure campaign
    campaign = structure_campaign(
        brand_name=brand_name,
        brand_description=brand_description,
        industry=industry,
        target_audience=target_audience,
        objective=objective,
        tone=tone,
        platforms=[p.strip() for p in platforms.split(",")]
    )

    # 2️⃣ Build prompt
    prompt = build_prompt(campaign)

    # 3️⃣ Generate raw output
    raw_output = generate_text(prompt)

    # 4️⃣ Clean prompt leakage
    if "### OUTPUT" in raw_output:
        output = raw_output.split("### OUTPUT")[-1].strip()
    else:
        output = raw_output.strip()

    # 5️⃣ Save campaign (learning memory)
    save_campaign(campaign, output)

    # 6️⃣ Emotion scoring
    emotion_result = rank_output(output)

    emotion_scores_text = "\n".join(
        [f"{emotion}: {score}" for emotion, score in emotion_result["emotion_scores"].items()]
    )

    emotion_summary = (
        f"Dominant Emotion: {emotion_result['dominant_emotion']}\n\n"
        f"Emotion Scores:\n{emotion_scores_text}"
    )

    return output, emotion_summary


# =========================
# 🎨 Gradio UI (MINIMALLY EXTENDED)
# =========================
with gr.Blocks(title="Rookus – Creative Campaign as a Service (Phase 1.5)") as demo:
    gr.Markdown(
        """
        ## 🚀 Rookus – AI-Powered Creative Campaign Studio  
        **Phase 1.5 | Human-in-the-Loop | Emotion-Aware AI**
        """
    )

    with gr.Row():
        brand_name = gr.Textbox(
            label="Brand Name",
            placeholder="e.g. Rookus"
        )
        industry = gr.Textbox(
            label="Industry",
            placeholder="e.g. SaaS, Fintech, E-commerce"
        )

    brand_description = gr.Textbox(
        label="Brand Description",
        lines=3,
        placeholder="Describe what your brand does and what makes it unique"
    )

    target_audience = gr.Textbox(
        label="Target Audience",
        placeholder="e.g. Startup founders, 25–40, India"
    )

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

    generate_btn = gr.Button("🚀 Generate Campaign")

    output = gr.Textbox(
        label="AI-Generated Campaign",
        lines=22
    )

    emotion_output = gr.Textbox(
        label="🧠 Emotion Analysis",
        lines=6
    )

    generate_btn.click(
        fn=generate_campaign,
        inputs=[
            brand_name,
            brand_description,
            industry,
            target_audience,
            objective,
            tone,
            platforms
        ],
        outputs=[output, emotion_output]
    )

    # 🔁 Revision UI (UNCHANGED FLOW)
    feedback = gr.Textbox(
        label="Client Feedback / Revision Request",
        placeholder="Make it more premium, less salesy, etc."
    )

    revise_btn = gr.Button("🔁 Revise Campaign")

    revised_output = gr.Textbox(
        label="Revised Campaign Output",
        lines=20
    )

    revise_btn.click(
        fn=revise_campaign,
        inputs=[output, feedback],
        outputs=revised_output
    )


# 🚀 Launch app
demo.launch()
