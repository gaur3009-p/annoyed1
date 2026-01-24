"""
Rookus - Enhanced Demo Version
FREE AI Models, Better UI, More Features!
"""

import gradio as gr
from campaign_structurer import structure_campaign
from prompt_engine import build_prompt
from llm_client import generate_text, generate_strategy, generate_creative
from memory_store import save_campaign
from scoring.emotion_scorer import score_emotions
from generators.text_overlay import overlay_text
from agents.variant_agent import build_variant_prompt
from scoring.copy_quality_scorer import score_copy_quality
from scoring.platform_fit_scorer import score_platform_fit
from generators.poster_prompt_builder import build_poster_prompt
from generators.image_generator import generate_poster
from decision_engine.variant_selector import select_best_variant
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def rank_output(output: str):
    """Analyze emotional resonance of output"""
    scores = score_emotions(output)
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return {
        "emotion_scores": scores,
        "dominant_emotion": ranking[0][0]
    }

def extract_headlines_and_copies(text):
    """Extract headlines and body copy from generated text"""
    lines = [l.strip("-•1234567890. ").strip()
             for l in text.split("\n") if l.strip()]
    
    headlines = []
    copies = []
    buffer = []
    
    for line in lines:
        # Lines with 10 words or less are likely headlines
        if len(line.split()) <= 10 and len(line) > 5:
            headlines.append(line)
        elif len(line) > 10:  # Longer lines are body copy
            buffer.append(line)
    
    # Combine body copy lines
    if buffer:
        copies.append(" ".join(buffer[:2]))  # Take first 2 sentences
    
    return headlines, copies

def select_best_headline(headlines):
    """Select headline with best emotional score"""
    if not headlines:
        return "Transform Your Business with Innovation"
    
    scored = []
    for h in headlines:
        emotion = max(score_emotions(h).values())
        clarity = score_copy_quality(h)
        scored.append((h, emotion + clarity))
    
    return max(scored, key=lambda x: x[1])[0]

def select_best_copy(copies):
    """Select copy with best overall score"""
    if not copies:
        return "Discover a smarter way to achieve your goals with cutting-edge solutions designed for success."
    
    scored = []
    for c in copies:
        emotion = max(score_emotions(c).values())
        clarity = score_copy_quality(c)
        scored.append((c, emotion + clarity))
    
    return max(scored, key=lambda x: x[1])[0]

# =============================================================================
# PHASE 1: BASIC CAMPAIGN GENERATION
# =============================================================================

def generate_campaign(
    brand_name,
    brand_description,
    industry,
    target_audience,
    objective,
    tone,
    platforms
):
    """Generate initial campaign with improved prompts"""
    
    logger.info(f"🚀 Starting campaign generation for {brand_name}")
    
    try:
        # Structure campaign data
        campaign = structure_campaign(
            brand_name,
            brand_description,
            industry,
            target_audience,
            objective,
            tone,
            platforms=[p.strip() for p in platforms.split(",")]
        )
        
        # Generate strategic foundation first
        strategy_prompt = f"""You are a senior marketing strategist.

Brand: {brand_name}
Description: {brand_description}
Industry: {industry}
Target Audience: {target_audience}
Objective: {objective}
Tone: {tone}

Create a focused marketing strategy including:
1. Core campaign concept (one clear idea)
2. Key audience pain points (3 main points)
3. Primary emotional driver
4. Main messaging angle

Keep it concise and actionable."""

        logger.info("📊 Generating strategy...")
        strategy = generate_strategy(strategy_prompt, max_new_tokens=400)
        
        # Generate creative content based on strategy
        creative_prompt = f"""You are an award-winning advertising copywriter.

STRATEGY:
{strategy}

Brand: {brand_name}
Tone: {tone}
Platforms: {platforms}

Write EXACTLY:
1. Five punchy headlines (5-8 words each)
2. Five ad copy variations (20-30 words each)
3. Three strong CTAs (3-5 words each)

Make them compelling, benefit-focused, and action-oriented.
NO explanations, just the copy."""

        logger.info("✍️ Generating creative content...")
        raw_output = generate_creative(creative_prompt, max_new_tokens=500)
        
        # Clean up output
        output = raw_output.strip()
        
        # Save to memory
        save_campaign(campaign, output)
        
        # Analyze emotions
        emotion_result = rank_output(output)
        emotion_summary = "\n".join(
            [f"{e}: {s:.3f}" for e, s in emotion_result["emotion_scores"].items()]
        )
        
        summary = f"""🎯 Dominant Emotion: {emotion_result['dominant_emotion']}

📊 Emotional Scores:
{emotion_summary}

⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        logger.info("✅ Campaign generated successfully")
        return output, summary
        
    except Exception as e:
        logger.error(f"❌ Error in generate_campaign: {str(e)}")
        return f"Error: {str(e)}", "Generation failed"

# =============================================================================
# PHASE 2: ADVANCED MULTI-VARIANT GENERATION
# =============================================================================

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
    """Advanced campaign with variants, scoring, and posters"""
    
    logger.info(f"🚀 Starting Phase 2 for {brand_name}")
    
    try:
        # Structure campaign
        campaign = structure_campaign(
            brand_name,
            brand_description,
            industry,
            target_audience,
            objective,
            tone,
            platforms.split(",")
        )
        
        # ======== STEP 1: Generate Variants ========
        logger.info("🎨 Generating creative variants...")
        
        variants = {}
        variant_types = ["Emotional", "Trust", "Bold"]
        
        for vtype in variant_types:
            prompt = build_variant_prompt(campaign, vtype)
            logger.info(f"  ↳ Generating {vtype} variant...")
            
            raw = generate_creative(prompt, max_new_tokens=400)
            variants[vtype] = raw.split("### VARIANT OUTPUT")[-1].strip()
        
        # ======== STEP 2: Score Variants ========
        logger.info("📊 Scoring variants...")
        
        scored_variants = []
        for vtype, text in variants.items():
            scores = score_emotions(text)
            
            scored_variants.append({
                "variant": vtype,
                "text": text,
                "emotion_score": max(scores.values()),
                "clarity_score": score_copy_quality(text),
                "visual_score": score_platform_fit(decision_platform, vtype)
            })
        
        # ======== STEP 3: Select Best Variant ========
        logger.info("🏆 Selecting best variant...")
        
        best_variant, final_score = select_best_variant(
            scored_variants, decision_platform
        )
        
        best_text = next(
            v["text"] for v in scored_variants if v["variant"] == best_variant
        )
        
        # ======== STEP 4: Extract Best Content ========
        logger.info("✂️ Extracting headlines and copy...")
        
        headlines, copies = extract_headlines_and_copies(best_text)
        best_headline = select_best_headline(headlines)
        best_copy = select_best_copy(copies)
        
        logger.info(f"  ↳ Selected headline: {best_headline[:50]}...")
        logger.info(f"  ↳ Selected copy: {best_copy[:50]}...")
        
        # ======== STEP 5: Generate Poster Visuals ========
        logger.info("🎨 Generating poster visuals...")
        
        posters = []
        poster_prompt = build_poster_prompt(campaign)
        
        # Generate 3 poster backgrounds
        for i in range(3):
            logger.info(f"  ↳ Generating poster {i+1}/3...")
            
            try:
                # Generate clean background
                background = generate_poster(poster_prompt)
                
                # Add text overlay
                final_poster = overlay_text(
                    background,
                    best_headline,
                    best_copy
                )
                
                posters.append(final_poster)
                logger.info(f"  ✅ Poster {i+1} complete")
                
            except Exception as e:
                logger.error(f"  ❌ Poster {i+1} failed: {str(e)}")
                # Continue with other posters
        
        # ======== STEP 6: Format Output ========
        logger.info("📝 Formatting results...")
        
        # Variants summary
        copy_block = "=" * 60 + "\n"
        copy_block += "ALL CREATIVE VARIANTS\n"
        copy_block += "=" * 60 + "\n\n"
        
        for v in scored_variants:
            copy_block += f"\n{'='*60}\n"
            copy_block += f"📌 {v['variant'].upper()} VARIANT\n"
            copy_block += f"{'='*60}\n"
            copy_block += f"{v['text']}\n\n"
            copy_block += f"📊 Scores:\n"
            copy_block += f"  • Emotion: {v['emotion_score']:.3f}\n"
            copy_block += f"  • Clarity: {v['clarity_score']:.3f}\n"
            copy_block += f"  • Platform Fit: {v['visual_score']:.3f}\n"
        
        # Decision summary
        decision_summary = f"""{'='*60}
🏆 WINNING VARIANT ANALYSIS
{'='*60}

Platform: {decision_platform}
Winner: {best_variant}
Final Score: {final_score:.3f}

{'='*60}
🎯 SELECTED CREATIVE ASSETS
{'='*60}

📰 HEADLINE:
{best_headline}

📝 BODY COPY:
{best_copy}

🖼️ VISUALS:
{len(posters)} poster(s) generated successfully

{'='*60}
⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}"""
        
        logger.info("✅ Phase 2 completed successfully")
        return copy_block, decision_summary, posters
        
    except Exception as e:
        logger.error(f"❌ Error in run_phase2: {str(e)}")
        error_msg = f"Error: {str(e)}\n\nPlease check logs for details."
        return error_msg, error_msg, []

# =============================================================================
# GRADIO INTERFACE
# =============================================================================

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
}
.gr-button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}
"""

with gr.Blocks(
    title="Rookus - AI Creative Campaign Studio",
    theme=gr.themes.Soft(primary_hue="purple"),
    css=custom_css
) as demo:
    
    # Header
    gr.Markdown("""
    # 🚀 Rookus - AI Creative Campaign Studio
    
    **Enterprise-grade campaign generation powered by FREE open-source AI models**
    
    ✨ Features: Multi-variant generation • Emotional scoring • Platform optimization • Visual asset creation
    """)
    
    # ========== PHASE 1: BASIC CAMPAIGN ==========
    with gr.Tab("📝 Phase 1: Basic Campaign"):
        gr.Markdown("""
        ### Generate Your First Campaign
        Fill in your brand details and we'll create a complete campaign strategy with creative assets.
        """)
        
        with gr.Row():
            with gr.Column():
                brand_name = gr.Textbox(
                    label="Brand Name",
                    placeholder="TechForward",
                    info="Your brand or company name"
                )
                brand_description = gr.Textbox(
                    label="Brand Description",
                    placeholder="AI-powered automation platform for SMBs",
                    lines=3,
                    info="Brief description of what you do"
                )
                industry = gr.Textbox(
                    label="Industry",
                    placeholder="Technology / SaaS",
                    info="Your industry or vertical"
                )
                target_audience = gr.Textbox(
                    label="Target Audience",
                    placeholder="Small business owners, 30-50 years old",
                    info="Who are you targeting?"
                )
            
            with gr.Column():
                objective = gr.Dropdown(
                    ["Awareness", "Leads", "Sales", "Engagement"],
                    label="Campaign Objective",
                    value="Leads",
                    info="Primary goal of this campaign"
                )
                tone = gr.Dropdown(
                    ["Premium", "Friendly", "Bold", "Trustworthy", "Professional"],
                    label="Brand Tone",
                    value="Professional",
                    info="How should your brand sound?"
                )
                platforms = gr.Textbox(
                    label="Platforms",
                    placeholder="Meta, Google, LinkedIn",
                    value="Meta, Google",
                    info="Comma-separated list"
                )
        
        generate_btn = gr.Button("🚀 Generate Campaign", variant="primary", size="lg")
        
        with gr.Row():
            output = gr.Textbox(
                label="📄 Generated Campaign",
                lines=20,
                show_copy_button=True
            )
            emotion_output = gr.Textbox(
                label="📊 Emotional Analysis",
                lines=20
            )
        
        generate_btn.click(
            generate_campaign,
            inputs=[
                brand_name, brand_description, industry,
                target_audience, objective, tone, platforms
            ],
            outputs=[output, emotion_output]
        )
    
    # ========== PHASE 2: ADVANCED CAMPAIGN ==========
    with gr.Tab("🎨 Phase 2: Advanced Multi-Variant"):
        gr.Markdown("""
        ### Generate Advanced Multi-Variant Campaign
        
        Creates multiple creative variants, scores them, and generates poster visuals.
        
        **This will:**
        1. Generate 3 different creative variants (Emotional, Trust, Bold)
        2. Score each variant on emotion, clarity, and platform fit
        3. Select the best variant for your chosen platform
        4. Generate 3 poster designs with text overlays
        """)
        
        decision_platform = gr.Dropdown(
            ["Meta", "Google", "LinkedIn"],
            label="Optimize For Platform",
            value="Meta",
            info="Which platform should we optimize for?"
        )
        
        phase2_btn = gr.Button("🎨 Run Advanced Generation", variant="primary", size="lg")
        
        gr.Markdown("### Results")
        
        with gr.Row():
            phase2_output = gr.Textbox(
                label="📄 All Creative Variants",
                lines=25,
                show_copy_button=True
            )
            decision_output = gr.Textbox(
                label="🏆 Winner Analysis",
                lines=25
            )
        
        poster_gallery = gr.Gallery(
            label="🖼️ Generated Posters",
            columns=3,
            height="auto",
            show_label=True
        )
        
        phase2_btn.click(
            run_phase2,
            inputs=[
                brand_name, brand_description, industry,
                target_audience, objective, tone, platforms,
                decision_platform
            ],
            outputs=[phase2_output, decision_output, poster_gallery]
        )
    
    # ========== INFO TAB ==========
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## 🎯 About Rookus
        
        Rookus is an AI-powered creative campaign studio that uses **100% FREE** open-source models from Hugging Face.
        
        ### 🤖 AI Models Used
        
        **Text Generation:**
        - **Strategy**: Meta Llama 3.2 (3B) - Excellent reasoning
        - **Creative**: Mistral 7B v0.3 - Superior creative writing
        
        **Image Generation:**
        - **Primary**: Flux Schnell - Fastest, high quality
        - **Alternative**: SDXL Turbo - Great fallback
        
        ### ✨ Features
        
        - ✅ Multi-variant campaign generation
        - ✅ Emotional resonance scoring
        - ✅ Platform-specific optimization
        - ✅ Visual asset creation
        - ✅ Text overlay on posters
        - ✅ Campaign memory storage
        - ✅ A/B testing support
        
        ### 🚀 Getting Started
        
        1. **Phase 1**: Fill in basic brand info and generate your first campaign
        2. **Phase 2**: Use advanced mode for multi-variant generation with visuals
        3. Review emotional scores and platform fit
        4. Download your favorite posters!
        
        ### 💾 System Requirements
        
        - **Minimum**: 8GB RAM, CPU
        - **Recommended**: 16GB RAM, GPU (NVIDIA GTX 1060+)
        - **Optimal**: 24GB+ RAM, GPU (NVIDIA RTX 3090/4090)
        
        ### 📚 Learn More
        
        - [GitHub Repository](#)
        - [Documentation](#)
        - [API Reference](#)
        
        ---
        
        **Built with ❤️ using free open-source AI**
        """)

# =============================================================================
# LAUNCH APP
# =============================================================================

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("🚀 Launching Rookus Creative Campaign Studio")
    logger.info("="*60)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates public link for sharing
        show_error=True,
        quiet=False
    )
