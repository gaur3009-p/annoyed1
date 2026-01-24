"""
Enhanced Image Generator with FREE Hugging Face Models
Uses Flux, SDXL, and other free models - NO API KEYS NEEDED!
"""

import torch
from diffusers import (
    FluxPipeline,
    StableDiffusionXLPipeline,
    DiffusionPipeline,
    DPMSolverMultistepScheduler
)
from PIL import Image
import logging
from typing import Optional, Literal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# FREE MODEL CONFIGURATIONS
# =============================================================================

class ImageModelConfig:
    """
    Best free image generation models from Hugging Face
    Ranked by quality and speed
    """
    
    # 🥇 BEST: Flux (FREE version) - Highest quality
    FLUX_SCHNELL = "black-forest-labs/FLUX.1-schnell"  # Fast, 4 steps
    FLUX_DEV = "black-forest-labs/FLUX.1-dev"  # Higher quality, slower
    
    # 🥈 GREAT: SDXL Models - Excellent quality
    SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
    SDXL_TURBO = "stabilityai/sdxl-turbo"  # Fast version
    
    # 🥉 GOOD: Specialized models
    REALISTIC_VISION = "SG161222/Realistic_Vision_V6.0_B1_noVAE"  # Photorealistic
    DREAMSHAPER = "Lykon/DreamShaper"  # Artistic, vibrant
    
    # For specific use cases
    PLAYGROUND_V2 = "playgroundai/playground-v2.5-1024px-aesthetic"  # Aesthetic


class ImageGeneratorPipeline:
    """
    Unified image generation pipeline supporting multiple models
    Automatically selects best available model based on hardware
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
        use_fp16: bool = True
    ):
        self.device = self._get_device(device)
        self.use_fp16 = use_fp16 and self.device != "cpu"
        
        # Auto-select best model if not specified
        if model_name is None:
            model_name = self._select_best_model()
        
        self.model_name = model_name
        self.pipe = None
        
        logger.info(f"🎨 Loading image model: {model_name}")
        self._load_pipeline()
    
    def _get_device(self, device: str) -> str:
        """Determine best device"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _select_best_model(self) -> str:
        """Auto-select best model based on available resources"""
        
        if not torch.cuda.is_available():
            logger.warning("⚠️ No GPU detected, using faster SDXL Turbo")
            return ImageModelConfig.SDXL_TURBO
        
        # Check VRAM
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if vram_gb >= 16:
                logger.info("✅ High VRAM detected, using Flux Dev (best quality)")
                return ImageModelConfig.FLUX_DEV
            elif vram_gb >= 10:
                logger.info("✅ Good VRAM detected, using Flux Schnell")
                return ImageModelConfig.FLUX_SCHNELL
            else:
                logger.info("✅ Standard VRAM detected, using SDXL Turbo")
                return ImageModelConfig.SDXL_TURBO
        
        return ImageModelConfig.SDXL_TURBO
    
    def _load_pipeline(self):
        """Load the diffusion pipeline"""
        
        try:
            torch_dtype = torch.float16 if self.use_fp16 else torch.float32
            
            # Load based on model type
            if "flux" in self.model_name.lower():
                self.pipe = FluxPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    use_safetensors=True
                )
            elif "sdxl" in self.model_name.lower():
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    variant="fp16" if self.use_fp16 else None
                )
            else:
                # Generic diffusion pipeline
                self.pipe = DiffusionPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    use_safetensors=True
                )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Optimize for memory
            if self.device == "cuda":
                self.pipe.enable_attention_slicing("max")
                self.pipe.enable_vae_slicing()
                
                # Enable xformers if available (faster)
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    logger.info("✅ xformers enabled for faster generation")
                except:
                    logger.info("ℹ️ xformers not available, using standard attention")
            
            logger.info(f"✅ Pipeline loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Error loading pipeline: {str(e)}")
            raise
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate image from prompt
        
        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid in image
            num_inference_steps: More steps = better quality (4-50)
            guidance_scale: How closely to follow prompt (0-20)
            width: Image width (must be multiple of 8)
            height: Image height (must be multiple of 8)
            seed: Random seed for reproducibility
        """
        
        # Set default negative prompt if not provided
        if negative_prompt is None:
            negative_prompt = self._get_default_negative_prompt()
        
        # Adjust parameters based on model
        if "flux-schnell" in self.model_name.lower():
            num_inference_steps = min(num_inference_steps, 4)
            guidance_scale = 0.0  # Flux Schnell works best without guidance
        elif "turbo" in self.model_name.lower():
            num_inference_steps = min(num_inference_steps, 4)
            guidance_scale = 0.0
        
        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        try:
            # Generate image
            logger.info(f"🎨 Generating: {prompt[:50]}...")
            
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            )
            
            image = output.images[0]
            
            logger.info("✅ Image generated successfully")
            return image
            
        except Exception as e:
            logger.error(f"❌ Generation error: {str(e)}")
            raise
    
    def _get_default_negative_prompt(self) -> str:
        """Get comprehensive negative prompt"""
        return (
            "text, letters, words, typography, watermark, signature, "
            "username, logo, symbols, numbers, captions, labels, "
            "low quality, blurry, distorted, deformed, ugly, "
            "bad anatomy, mutation, extra limbs, missing limbs, "
            "poorly drawn, amateur, sketch, draft"
        )
    
    def generate_batch(
        self,
        prompts: list[str],
        **kwargs
    ) -> list[Image.Image]:
        """Generate multiple images"""
        
        images = []
        for prompt in prompts:
            image = self.generate(prompt, **kwargs)
            images.append(image)
        
        return images


# =============================================================================
# CAMPAIGN-SPECIFIC GENERATOR
# =============================================================================

class CampaignImageGenerator:
    """
    Specialized image generator for marketing campaigns
    Handles brand consistency and campaign-specific requirements
    """
    
    def __init__(self, model_name: Optional[str] = None):
        self.pipeline = ImageGeneratorPipeline(model_name)
    
    def generate_poster(
        self,
        campaign: dict,
        style: Literal["modern", "minimal", "bold", "elegant"] = "modern"
    ) -> Image.Image:
        """
        Generate campaign poster background
        NO TEXT - just beautiful background for text overlay
        """
        
        # Build optimized prompt
        prompt = self._build_campaign_prompt(campaign, style)
        
        # Enhanced negative prompt for marketing
        negative_prompt = (
            "text, letters, words, typography, logos, watermarks, "
            "symbols, numbers, captions, labels, signatures, "
            "people, faces, humans, person, crowd, "  # No people for brand safety
            "low quality, blurry, amateur, sketch, cartoon, anime"
        )
        
        # Generate with optimal settings
        image = self.pipeline.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=6,  # Good balance of speed/quality
            guidance_scale=3.5,     # Moderate guidance
            width=1024,
            height=1024
        )
        
        return image
    
    def _build_campaign_prompt(self, campaign: dict, style: str) -> str:
        """Build optimized prompt for campaign imagery"""
        
        industry = campaign.get('industry', 'technology')
        tone = campaign.get('tone', 'professional')
        emotion = campaign.get('emotion', ['trust'])[0]
        
        # Style-specific elements
        style_elements = {
            "modern": "clean gradients, soft geometric shapes, contemporary design, sleek",
            "minimal": "minimalist, simple, negative space, subtle colors, zen",
            "bold": "vibrant colors, dynamic composition, energetic, striking",
            "elegant": "sophisticated, premium, refined, luxurious, timeless"
        }
        
        # Industry-specific elements
        industry_elements = {
            "technology": "futuristic, digital, innovation, circuit patterns, data flow",
            "finance": "professional, trustworthy, stability, growth charts, upward trend",
            "healthcare": "caring, clean, medical precision, wellness, calm blues",
            "education": "knowledge, growth, inspiration, books, learning journey",
            "retail": "lifestyle, aspirational, product-focused, shopping experience"
        }
        
        # Emotion-based color palette
        emotion_colors = {
            "trust": "blue tones, reliability",
            "urgency": "red and orange, dynamic",
            "aspiration": "purple and gold, premium",
            "curiosity": "teal and yellow, intriguing"
        }
        
        # Construct prompt
        prompt = f"""Professional advertising background, {style_elements.get(style, 'modern')},
{industry_elements.get(industry.lower(), 'professional and clean')},
{emotion_colors.get(emotion.lower(), 'balanced colors')},
high quality commercial photography, studio lighting, professional grade,
product photography aesthetic, clean composition, depth of field,
cinematic lighting, 8K, ultra detailed, photorealistic,
suitable for premium brand advertising"""
        
        # Clean up prompt
        prompt = " ".join(prompt.split())
        
        return prompt
    
    def generate_multiple_variants(
        self,
        campaign: dict,
        num_variants: int = 3
    ) -> list[Image.Image]:
        """Generate multiple style variants"""
        
        styles = ["modern", "minimal", "bold", "elegant"][:num_variants]
        images = []
        
        for style in styles:
            image = self.generate_poster(campaign, style=style)
            images.append(image)
        
        return images


# =============================================================================
# GLOBAL INSTANCE (backward compatible)
# =============================================================================

logger.info("🔥 Initializing image generation pipeline...")

# Create global generator instance
campaign_generator = CampaignImageGenerator()

logger.info("✅ Image generator ready!")


# =============================================================================
# CONVENIENCE FUNCTIONS (backward compatible with old code)
# =============================================================================

def generate_poster(prompt_or_campaign, **kwargs) -> Image.Image:
    """
    Main poster generation function
    Accepts either a prompt string or campaign dict
    """
    
    if isinstance(prompt_or_campaign, dict):
        # Campaign dict provided
        return campaign_generator.generate_poster(prompt_or_campaign, **kwargs)
    else:
        # Raw prompt provided
        return campaign_generator.pipeline.generate(prompt_or_campaign, **kwargs)


# =============================================================================
# TESTING & EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing Enhanced Image Generator")
    print("="*60 + "\n")
    
    # Test 1: Simple generation
    print("🎨 Test 1: Simple prompt...")
    simple_image = generate_poster(
        "Beautiful abstract gradient background, modern, professional, blue and purple tones"
    )
    simple_image.save("test_simple.png")
    print("✅ Saved: test_simple.png")
    
    # Test 2: Campaign-based generation
    print("\n🎨 Test 2: Campaign-based generation...")
    test_campaign = {
        "brand": "TechForward",
        "industry": "Technology",
        "tone": "Professional",
        "emotion": ["Trust", "Innovation"]
    }
    
    campaign_image = generate_poster(test_campaign, style="modern")
    campaign_image.save("test_campaign.png")
    print("✅ Saved: test_campaign.png")
    
    # Test 3: Multiple variants
    print("\n🎨 Test 3: Multiple style variants...")
    variants = campaign_generator.generate_multiple_variants(test_campaign, num_variants=3)
    
    for i, variant in enumerate(variants, 1):
        variant.save(f"test_variant_{i}.png")
        print(f"✅ Saved: test_variant_{i}.png")
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("Check generated images in current directory")
    print("="*60 + "\n")
