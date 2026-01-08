import gradio as gr
import numpy as np
import random
import torch
import os

from diffusers import QwenImageEditPlusPipeline
from tools.prompt_utils import polish_edit_prompt

# --- Model Loading ---
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

# Support local model path or HuggingFace model ID
model_path = os.environ.get("MODEL_PATH", "/app/models/Qwen-Image-Edit-2511")
print(f"Loading edit model from: {model_path}")

# Check for low VRAM mode
enable_cpu_offload = os.environ.get("ENABLE_CPU_OFFLOAD", "false").lower() == "true"
low_vram_mode = os.environ.get("LOW_VRAM_MODE", "false").lower() == "true"
print(f"Loading with CPU offload={enable_cpu_offload}, low_vram={low_vram_mode}")

# Load the model pipeline (2511 uses QwenImageEditPlusPipeline)
pipe = QwenImageEditPlusPipeline.from_pretrained(model_path, torch_dtype=dtype)

if enable_cpu_offload:
    print("Enabling sequential CPU offload for low VRAM")
    pipe.enable_sequential_cpu_offload()
else:
    pipe = pipe.to(device)

# Enable memory optimizations if available
if hasattr(pipe, 'enable_attention_slicing'):
    pipe.enable_attention_slicing(1)
    print("Enabled attention slicing")

if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_slicing'):
    pipe.vae.enable_slicing()
    print("Enabled VAE slicing")

if low_vram_mode and hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):
    pipe.vae.enable_tiling()
    print("Enabled VAE tiling")

print(f"Model loaded successfully on {device}")

# --- UI Constants and Helpers ---
MAX_SEED = np.iinfo(np.int32).max

# --- Main Inference Function (with hardcoded negative prompt) ---
def infer(
    image,
    prompt,
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=50,
    rewrite_prompt=True,
    num_images_per_prompt=1,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generates an image using the local Qwen-Image diffusers pipeline.
    """
    # Hardcode the negative prompt as requested
    negative_prompt = " "
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # Set up the generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print(f"Calling pipeline with prompt: '{prompt}'")
    print(f"Negative Prompt: '{negative_prompt}'")
    print(f"Seed: {seed}, Steps: {num_inference_steps}, Guidance: {true_guidance_scale}")
    if rewrite_prompt:
        prompt = polish_edit_prompt(prompt, image)
        print(f"Rewritten Prompt: {prompt}")

    # Generate the image
    image = pipe(
        image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        true_cfg_scale=true_guidance_scale,
        num_images_per_prompt=num_images_per_prompt
    ).images

    return image, seed

# --- Examples and UI Layout ---
examples = []

css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
#edit_text{margin-top: -62px !important}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML('<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_edit_logo.png" alt="Qwen-Image Logo" width="400" style="display: block; margin: 0 auto;">')
        gr.Markdown("[Learn more](https://github.com/QwenLM/Qwen-Image) about the Qwen-Image series. Try on [Qwen Chat](https://chat.qwen.ai/), or [download model](https://huggingface.co/Qwen/Qwen-Image-Edit) to run locally with ComfyUI or diffusers.")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", show_label=False, type="pil")

            # result = gr.Image(label="Result", show_label=False, type="pil")
            result = gr.Gallery(label="Result", show_label=False, type="pil")
        with gr.Row():
            prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    placeholder="describe the edit instruction",
                    container=False,
            )
            run_button = gr.Button("Edit!", variant="primary")

        with gr.Accordion("Advanced Settings", open=False):
            # Negative prompt UI element is removed here

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():

                true_guidance_scale = gr.Slider(
                    label="True guidance scale",
                    minimum=1.0,
                    maximum=10.0,
                    step=0.1,
                    value=4.0
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=50,
                )
                
                num_images_per_prompt = gr.Slider(
                    label="Number of images per prompt",
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=1,
                )
                
                rewrite_prompt = gr.Checkbox(label="Rewrite prompt", value=True)

        # gr.Examples(examples=examples, inputs=[prompt], outputs=[result, seed], fn=infer, cache_examples=False)

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            input_image,
            prompt,
            seed,
            randomize_seed,
            true_guidance_scale,
            num_inference_steps,
            rewrite_prompt,
            num_images_per_prompt,
        ],
        outputs=[result, seed],
    )

if __name__ == "__main__":
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7861"))
    demo.launch(server_name=server_name, server_port=server_port)