import os
import time
import logging
from typing import Tuple, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer

# Configure logging for production tracing
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Registry of supported models to enable modular replacement (Phi-3, Qwen, Gemma)
MODEL_CONFIGS = {
    "microsoft/Phi-3-mini-4k-instruct": {
        "trust_remote_code": False,
        "tokenizer_kwargs": {"use_fast": True},
        # Pass attn_implementation="eager" or "sdpa" to avoid flash-attention requirements on CPU
        "model_kwargs": {"attn_implementation": "sdpa"}
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "trust_remote_code": False,
        "tokenizer_kwargs": {"use_fast": True},
        "model_kwargs": {}
    },
    "google/gemma-2-2b-it": {
        "trust_remote_code": False,
        "tokenizer_kwargs": {},
        "model_kwargs": {}
    }
}

def load_model_and_tokenizer(
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
    **kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Loads and returns a causal language model and its corresponding tokenizer.
    
    This function is optimized for CPU inference compatibility (safely using float32 
    to avoid unsupported half precision operations on CPU) and auto-detects CUDA/GPUs 
    when available.
    
    Args:
        model_name: HuggingFace model hub ID (e.g. Phi-3, Qwen, Gemma models).
        device: Target hardware device ('cpu', 'cuda', etc.). If None, auto-detected.
        dtype: Data precision type (torch.float32, torch.float16, etc.). If None, auto-detected.
        trust_remote_code: Allow execution of custom modeling code on download.
        **kwargs: Additional model arguments passed directly to the Hugging Face loader.
        
    Returns:
        A tuple of (model, tokenizer)
    """
    start_time = time.time()
    
    # 1. Device and precision resolution (CPU Compatibility Design)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    if dtype is None:
        # Crucial for CPU: float16 operations are unsupported or slow on many CPU architectures.
        # Use bfloat16 on CPU to save 50% memory (7.6GB vs 15GB) and prevent OOM, while
        # keeping float16 on GPU.
        dtype = torch.bfloat16 if device == "cpu" else torch.float16
        
    print(f"\n" + "="*50)
    print(f"[ModelLoader] Initializing Model Load Sequence")
    print(f"[ModelLoader] Model: {model_name}")
    print(f"[ModelLoader] Target Device: {device.upper()}")
    print(f"[ModelLoader] Precision Dtype: {dtype}")
    print(f"="*50)
    
    # 2. Extract model config from registry, falling back to defaults if not found
    config = MODEL_CONFIGS.get(model_name, {
        "trust_remote_code": trust_remote_code,
        "tokenizer_kwargs": {},
        "model_kwargs": {}
    })
    
    # Consolidate kwargs (configs merge with arguments)
    tokenizer_kwargs = {
        **config.get("tokenizer_kwargs", {}),
        "trust_remote_code": config.get("trust_remote_code", trust_remote_code)
    }
    
    model_kwargs = {
        **config.get("model_kwargs", {}),
        "trust_remote_code": config.get("trust_remote_code", trust_remote_code)
    }
    
    # Override with explicitly passed kwargs
    model_kwargs.update(kwargs)
    
    # 3. Load Tokenizer
    print(f"[ModelLoader] Loading tokenizer for {model_name}...")
    tok_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    
    # Ensure pad token is configured properly for batching or generation
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
    print(f"[ModelLoader] Tokenizer loaded successfully in {time.time() - tok_start:.2f} seconds.")
    
    # 4. Load Causal Language Model
    print(f"[ModelLoader] Loading model weights for {model_name}...")
    model_start = time.time()
    
    if device == "cpu":
        # Avoid using device_map="auto" on CPU, since it may look for accelerator runtimes.
        # Direct load to CPU with device_map=None is more robust and clean.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,
            low_cpu_mem_usage=True,
            **model_kwargs
        )
        model = model.to("cpu")
    else:
        # Standard accelerate-managed placement on GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            **model_kwargs
        )
        
    print(f"[ModelLoader] Model loaded successfully in {time.time() - model_start:.2f} seconds.")
    
    total_elapsed = time.time() - start_time
    print(f"[ModelLoader] Initialization complete! Total elapsed time: {total_elapsed:.2f} seconds.")
    print("="*50 + "\n")
    
    return model, tokenizer

# Internal cache for lazy loading exports
_model = None
_tokenizer = None

def __getattr__(name: str) -> Any:
    """
    Implements lazy loading for module-level 'model' and 'tokenizer' attributes.
    
    This maintains backward compatibility with files importing them directly (e.g. 
    `from generator.model_loader import model`), while preventing CPU blocking or 
    automatic model downloading when importing this module for unit tests or configuration.
    """
    global _model, _tokenizer
    if name == "model":
        if _model is None:
            _model, _tokenizer = load_model_and_tokenizer()
        return _model
    elif name == "tokenizer":
        if _tokenizer is None:
            _model, _tokenizer = load_model_and_tokenizer()
        return _tokenizer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")