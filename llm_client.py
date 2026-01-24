"""
Enhanced LLM Client with FREE Hugging Face Models
No API keys needed - runs locally or uses free inference API
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# FREE MODEL CONFIGURATION - Choose best available
# =============================================================================

class ModelConfig:
    """Free, high-quality models from Hugging Face"""
    
    # For Strategy & Planning (Best reasoning)
    STRATEGY_MODEL = "meta-llama/Llama-3.2-3B-Instruct"  # Latest Llama, great reasoning
    
    # For Creative Writing (Best for marketing copy)
    CREATIVE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"  # Excellent for creative content
    
    # Alternative models you can try:
    # "microsoft/Phi-3-mini-4k-instruct"  # Fast, good quality
    # "google/gemma-2-2b-it"  # Google's latest small model
    # "Qwen/Qwen2.5-7B-Instruct"  # Excellent multilingual


class EnhancedLLMClient:
    """
    Enhanced LLM client with free Hugging Face models
    Optimized for T4 GPU (Google Colab free tier)
    """
    
    def __init__(
        self,
        model_name: str = ModelConfig.CREATIVE_MODEL,
        device: str = "auto",
        load_in_4bit: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        logger.info(f"🚀 Loading model: {model_name}")
        self._load_model(load_in_4bit)
    
    def _load_model(self, load_in_4bit: bool):
        """Load model with optimal configuration"""
        
        try:
            # Quantization config for memory efficiency
            if load_in_4bit and torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            else:
                # CPU or standard GPU loading
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                trust_remote_code=True
            )
            
            # Set padding token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Disable cache for memory efficiency
            self.model.config.use_cache = False
            
            # Create text generation pipeline for easier use
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto"
            )
            
            logger.info(f"✅ Model loaded successfully: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            raise
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.1
    ) -> str:
        """
        Generate text with advanced parameters
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            do_sample: Whether to use sampling
            repetition_penalty: Penalty for repetition
        """
        
        try:
            # Format prompt for instruction-tuned models
            formatted_prompt = self._format_prompt(prompt)
            
            # Generate using pipeline
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            
            # Clean up output
            generated_text = self._clean_output(generated_text)
            
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ Generation error: {str(e)}")
            return f"Error generating text: {str(e)}"
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for instruction-tuned models"""
        
        # Llama 3 format
        if "llama" in self.model_name.lower():
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert marketing strategist and creative director.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Mistral format
        elif "mistral" in self.model_name.lower():
            return f"<s>[INST] {prompt} [/INST]"
        
        # Phi-3 format
        elif "phi" in self.model_name.lower():
            return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        
        # Gemma format
        elif "gemma" in self.model_name.lower():
            return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # Default format
        else:
            return f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    def _clean_output(self, text: str) -> str:
        """Clean up generated output"""
        
        # Remove common artifacts
        text = text.strip()
        
        # Remove instruction markers that might leak through
        markers_to_remove = [
            "<|eot_id|>", "<|end_of_text|>", "</s>", "<end_of_turn>",
            "### Instruction:", "### Response:", "[/INST]", "<|assistant|>"
        ]
        
        for marker in markers_to_remove:
            text = text.replace(marker, "")
        
        return text.strip()
    
    def generate_with_variants(
        self,
        prompt: str,
        num_variants: int = 3,
        **kwargs
    ) -> list[str]:
        """Generate multiple variants with different temperatures"""
        
        variants = []
        temperatures = [0.6, 0.8, 1.0][:num_variants]
        
        for temp in temperatures:
            variant = self.generate_text(
                prompt,
                temperature=temp,
                **kwargs
            )
            variants.append(variant)
        
        return variants


# =============================================================================
# GLOBAL CLIENT INSTANCES
# =============================================================================

# Initialize clients
logger.info("🔥 Initializing LLM clients...")

# Strategy client (better reasoning)
strategy_client = EnhancedLLMClient(
    model_name=ModelConfig.STRATEGY_MODEL,
    load_in_4bit=True
)

# Creative client (better creative writing)
creative_client = EnhancedLLMClient(
    model_name=ModelConfig.CREATIVE_MODEL,
    load_in_4bit=True
)

logger.info("✅ All LLM clients ready!")


# =============================================================================
# CONVENIENCE FUNCTIONS (backward compatible with old code)
# =============================================================================

def generate_text(prompt: str, **kwargs) -> str:
    """
    Main text generation function
    Uses creative model by default
    """
    return creative_client.generate_text(prompt, **kwargs)


def generate_strategy(prompt: str, **kwargs) -> str:
    """
    Generate strategic content
    Uses strategy model (better reasoning)
    """
    return strategy_client.generate_text(prompt, **kwargs)


def generate_creative(prompt: str, **kwargs) -> str:
    """
    Generate creative content
    Uses creative model (better writing)
    """
    return creative_client.generate_text(prompt, **kwargs)


# =============================================================================
# TESTING & EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing Enhanced LLM Client")
    print("="*60 + "\n")
    
    # Test strategy generation
    print("📊 Testing Strategy Model...")
    strategy_prompt = """Create a marketing strategy for a new AI-powered fitness app.
Target audience: Busy professionals aged 25-40.
Goal: Drive app downloads and subscriptions."""
    
    strategy = generate_strategy(strategy_prompt, max_new_tokens=300)
    print("\n" + "="*60)
    print("STRATEGY OUTPUT:")
    print("="*60)
    print(strategy)
    
    # Test creative generation
    print("\n\n✍️ Testing Creative Model...")
    creative_prompt = """Write 3 compelling ad headlines for an AI fitness app.
Make them punchy, benefit-focused, and action-oriented."""
    
    creative = generate_creative(creative_prompt, max_new_tokens=150)
    print("\n" + "="*60)
    print("CREATIVE OUTPUT:")
    print("="*60)
    print(creative)
    
    # Test variants
    print("\n\n🎨 Testing Variant Generation...")
    variants = creative_client.generate_with_variants(
        creative_prompt,
        num_variants=3,
        max_new_tokens=100
    )
    
    for i, variant in enumerate(variants, 1):
        print(f"\n--- Variant {i} ---")
        print(variant)
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60 + "\n")
