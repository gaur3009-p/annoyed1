import gradio as gr

from campaign_structurer import structure_campaign
from prompt_engine import build_prompt
from llm_client import generate_text
from memory_store import save_campaign


def generate_campaign(
    brand_name,
    brand_description,
    industry,
    target_audience,
    objective,
    tone,
    platforms
):
    # 1️⃣ Structure campaign (explicit reasoning layer)
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

    # 3️⃣ Generate campaign using LLM
    output = generate_text(prompt)

    # 4️⃣ Store campaign + output (Phase 1 memory)
    save_campaign(campaign, output)

    return output


# =========================
# 🎨 Gradio UI
# =========================

with gr.Blocks(title="Rookus – Creative Campaign as a Service (Phase 1)") as demo:
    gr.Markdown(
        """
        ## 🚀 Rookus – AI-Powered Creative Campaign Studio  
        **Phase 1 | Human-in-the-Loop | Brand-Safe AI**
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
        outputs=output
    )

# 🚀 Launch app
demo.launch()
