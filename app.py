import gradio as gr

from campaign_structurer import structure_campaign
from prompt_engine import build_prompt
from llm_client import generate_text
from memory_store import save_campaign


def generate_campaign(
    brand_name,
    industry,
    target_audience,
    objective,
    tone,
    platforms
):
    campaign = structure_campaign(
        brand_name,
        industry,
        target_audience,
        objective,
        tone,
        platforms.split(",")
    )

    prompt = build_prompt(campaign)
    output = generate_text(prompt)

    save_campaign(campaign, output)

    return output


with gr.Blocks(title="Rookus – Creative Campaign as a Service") as demo:
    gr.Markdown("## 🚀 Rookus – AI-Powered Creative Campaign Studio (Phase 1)")

    with gr.Row():
        brand_name = gr.Textbox(label="Brand Name")
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
        label="Platforms (comma-separated)",
        placeholder="Meta, Google, LinkedIn"
    )

    generate_btn = gr.Button("Generate Campaign 🚀")

    output = gr.Textbox(
        label="AI Generated Campaign",
        lines=20
    )

    generate_btn.click(
        generate_campaign,
        inputs=[
            brand_name,
            industry,
            target_audience,
            objective,
            tone,
            platforms
        ],
        outputs=output
    )

demo.launch()
