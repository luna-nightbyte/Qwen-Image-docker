"""
Optimized Qwen-Image-Edit Demo Script with Quantization
========================================================

Supports three modes:
1. quantized (recommended) - BitsAndBytes 4-bit, ~5-15s/step
2. standard + model_cpu_offload - ~15-30s/step  
3. standard + sequential_cpu_offload - ~120s/step (fallback)

Environment variables:
- MODEL_PATH: Path to model (default: "/app/models/Qwen-Image-Edit-2511")
- OPTIMIZATION_MODE: "quantized", "standard" (default: "quantized")
- ENABLE_CPU_OFFLOAD: "true"/"false" (default: "true")
- LOW_VRAM_MODE: "true"/"false" - force sequential offload (default: "false")
"""

import gradio as gr
import numpy as np
import random
import torch
import os
import time
import gc

from tools.prompt_utils import polish_edit_prompt

# --- Configuration ---
dtype = torch.bfloat16
gpu_id = 0  # Inside container, mapped GPU is always 0

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model path
model_path = os.environ.get("MODEL_PATH", "/app/models/Qwen-Image-Edit-2511")
print(f"Model path: {model_path}")

# Optimization settings
optimization_mode = os.environ.get("OPTIMIZATION_MODE", "quantized").lower()
enable_cpu_offload = os.environ.get("ENABLE_CPU_OFFLOAD", "true").lower() == "true"
low_vram_mode = os.environ.get("LOW_VRAM_MODE", "false").lower() == "true"

print(f"\nConfiguration:")
print(f"  - Optimization mode: {optimization_mode}")
print(f"  - CPU offload: {enable_cpu_offload}")
print(f"  - Low VRAM mode: {low_vram_mode}")

# Track actual mode used
offload_mode = "unknown"

# --- Clear GPU memory before loading ---
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# --- Model Loading ---
print("\n" + "="*50)
print("Loading model...")
print("="*50)

# ============================================================
# OPTION 1: BitsAndBytes 4-bit Quantization (RECOMMENDED)
# ============================================================
if optimization_mode == "quantized":
    try:
        from diffusers import QwenImageEditPlusPipeline, PipelineQuantizationConfig
        from diffusers import BitsAndBytesConfig as DiffusersBnBConfig
        from transformers import BitsAndBytesConfig as TransformersBnBConfig
        
        print("Using BitsAndBytes 4-bit quantization (recommended)")
        
        quant_config = PipelineQuantizationConfig(
            quant_mapping={
                "transformer": DiffusersBnBConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=[
                        "time_text_embed", "img_in", "norm_out",
                        "proj_out", "img_mod", "txt_mod",
                    ],
                ),
                "text_encoder": TransformersBnBConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                ),
            }
        )
        
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_path,
            quantization_config=quant_config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="cuda",  # Use cuda strategy (all on GPU)
        )
        
        # With 4-bit quantization, model is much smaller (~5GB instead of ~40GB)
        # device_map forces all components on single GPU to avoid CPU dispatch
        print(f"Using device_map (GPU {gpu_id}) for quantized model")
        offload_mode = "quantized + device_map"
        
        print("✓ Quantized model loaded successfully!")
        
    except ImportError as e:
        print(f"✗ BitsAndBytes not available: {e}")
        print("  Install with: pip install bitsandbytes")
        print("  Falling back to standard mode...")
        optimization_mode = "standard"
    except Exception as e:
        print(f"✗ Quantization failed: {e}")
        print("  Falling back to standard mode...")
        optimization_mode = "standard"

# ============================================================
# OPTION 2: Standard mode (no quantization)
# ============================================================
if optimization_mode == "standard":
    from diffusers import QwenImageEditPlusPipeline
    
    print("Using standard mode (no quantization)")
    
    # Load to CPU first to prevent OOM
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    
    if enable_cpu_offload:
        if low_vram_mode:
            print("Enabling SEQUENTIAL CPU offload (slow, ~120s/step)")
            pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
            offload_mode = "sequential"
        else:
            try:
                print("Enabling MODEL CPU offload (faster, ~15-30s/step)")
                pipe.enable_model_cpu_offload(gpu_id=gpu_id)
                offload_mode = "model"
            except Exception as e:
                print(f"model_cpu_offload failed: {e}")
                print("Falling back to sequential...")
                pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
                offload_mode = "sequential (fallback)"
    else:
        print(f"Loading to {device} (no offload)")
        pipe = pipe.to(device)
        offload_mode = "none"

# --- Memory Optimizations ---
print("\nApplying memory optimizations...")

# xformers (disabled - incompatible with Qwen-Image-Edit attention mechanism)
# The model uses custom dual attention outputs that xformers doesn't support
print("  - xformers attention disabled (incompatible with this model)")

# Attention slicing
if hasattr(pipe, 'enable_attention_slicing'):
    slice_size = 1 if low_vram_mode else "auto"
    pipe.enable_attention_slicing(slice_size)
    print(f"  ✓ Attention slicing ({slice_size})")

# VAE optimizations
if hasattr(pipe, 'vae'):
    if hasattr(pipe.vae, 'enable_slicing'):
        pipe.vae.enable_slicing()
        print("  ✓ VAE slicing")
    if hasattr(pipe.vae, 'enable_tiling'):
        pipe.vae.enable_tiling()
        print("  ✓ VAE tiling")

# Clear cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

print(f"\n{'='*50}")
print(f"✓ Model ready! Mode: {offload_mode}")
print(f"{'='*50}\n")

# --- UI Constants ---
MAX_SEED = np.iinfo(np.int32).max

# --- Inference Function ---
def infer(
    image,
    prompt,
    negative_prompt,
    seed=42,
    randomize_seed=False,
    true_guidance_scale=4.0,
    num_inference_steps=50,
    rewrite_prompt=True,
    num_images_per_prompt=1,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate edited image."""
    if image is None:
        raise gr.Error("Please upload an image first!")
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # Generator device
    gen_device = "cpu" if enable_cpu_offload else device
    generator = torch.Generator(device=gen_device).manual_seed(seed)
    
    print(f"\n{'='*50}")
    print(f"Generating edit...")
    print(f"  Prompt: '{prompt}'")
    print(f"  Negative: '{negative_prompt}'")
    print(f"  Seed: {seed}, Steps: {num_inference_steps}")
    print(f"  Mode: {offload_mode}")
    
    # Rewrite prompt if enabled
    if rewrite_prompt:
        try:
            prompt = polish_edit_prompt(prompt, image)
            print(f"  Rewritten: '{prompt}'")
        except Exception as e:
            print(f"  Rewrite skipped: {e}")
    
    # Clear cache before inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    start_time = time.time()
    
    result = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else " ",
        num_inference_steps=num_inference_steps,
        generator=generator,
        true_cfg_scale=true_guidance_scale,
        num_images_per_prompt=num_images_per_prompt
    ).images
    
    elapsed = time.time() - start_time
    per_step = elapsed / num_inference_steps
    
    print(f"  ✓ Done: {elapsed:.1f}s ({per_step:.2f}s/step)")
    print(f"{'='*50}\n")
    
    # Clear cache after
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, seed


# --- Gradio UI ---
css = """
#col-container { margin: 0 auto; max-width: 1024px; }
.status-good { background: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0; }
.status-warn { background: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }
.status-bad { background: #f8d7da; padding: 10px; border-radius: 5px; margin: 10px 0; }
"""

with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML('<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_edit_logo.png" alt="Qwen-Image Logo" width="400" style="display: block; margin: 0 auto;">')
        gr.Markdown("[Learn more](https://github.com/QwenLM/Qwen-Image) about Qwen-Image.")
        
        # Status indicator
        if "quantized" in offload_mode:
            gr.HTML(f'<div class="status-good">✓ <b>Quantized mode</b> (~5-15s/step) - {offload_mode}</div>')
        elif offload_mode == "model":
            gr.HTML(f'<div class="status-good">✓ <b>Model offload</b> (~15-30s/step)</div>')
        elif "sequential" in offload_mode:
            gr.HTML(f'<div class="status-warn">⚠️ <b>Sequential offload</b> (~120s/step) - Set OPTIMIZATION_MODE=quantized for faster inference</div>')
        else:
            gr.HTML(f'<div class="status-bad">Mode: {offload_mode}</div>')
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", show_label=False, type="pil")
            result = gr.Gallery(label="Result", show_label=False, type="pil")
        
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                placeholder="Describe the edit instruction",
                container=False,
            )
            negative_prompt = gr.Text(
                label="Negative Prompt",
                show_label=False,
                placeholder="What to avoid (optional)",
                container=False,
            )
            run_button = gr.Button("Edit!", variant="primary")

        with gr.Accordion("Advanced Settings", open=False):
            seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                true_guidance_scale = gr.Slider(
                    label="Guidance scale", minimum=1.0, maximum=10.0, step=0.1, value=4.0
                )
                num_inference_steps = gr.Slider(
                    label="Steps", minimum=10, maximum=50, step=5, value=50
                )
                num_images_per_prompt = gr.Slider(
                    label="Images", minimum=1, maximum=4, step=1, value=1
                )
                rewrite_prompt = gr.Checkbox(label="Rewrite prompt", value=True)

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            input_image, prompt, negative_prompt, seed, randomize_seed,
            true_guidance_scale, num_inference_steps,
            rewrite_prompt, num_images_per_prompt,
        ],
        outputs=[result, seed],
    )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("QWEN-IMAGE-EDIT SERVER")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Mode: {offload_mode}")
    print()
    if "quantized" in offload_mode:
        print("✓ Using quantized mode (fastest)")
    elif "sequential" in offload_mode:
        print("⚠️ Using sequential offload (slow)")
        print("  Set OPTIMIZATION_MODE=quantized for faster inference")
    print("="*60 + "\n")
    
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7861"))
    demo.launch(css=css, server_name=server_name, server_port=server_port)